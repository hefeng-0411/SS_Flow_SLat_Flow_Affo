"""Structured SS-flow and differentiable coarse-geometry objectives."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.geometry.kaolin_camera import project_opencv_points_kaolin
from geoss.utils.optional_deps import require_dependency


class FlowMatchingLossBuilder(nn.Module):
    """Build CFM, decoder occupancy, volume reprojection and prior losses."""

    def __init__(
        self,
        *,
        lambda_cfm: float = 1.0,
        lambda_depth: float = 0.1,
        lambda_silhouette: float = 0.1,
        lambda_occupancy: float = 0.5,
        lambda_surface: float = 0.05,
        lambda_prior: float = 0.01,
        use_depth_loss: bool = True,
        use_silhouette_loss: bool = True,
        use_surface_loss: bool = False,
        use_prior_preservation: bool = True,
        surface_backend: str = "geomloss",
        surface_blur: float = 0.05,
        render_resolution: int = 64,
        ray_samples: int = 48,
        extensions_root: Optional[str] = None,
    ) -> None:
        super().__init__()
        if surface_backend not in {"geomloss", "frnn"}:
            raise ValueError(f"Unsupported surface_backend={surface_backend!r}")
        self.weights = {
            "cfm": float(lambda_cfm),
            "depth": float(lambda_depth),
            "silhouette": float(lambda_silhouette),
            "occupancy": float(lambda_occupancy),
            "surface": float(lambda_surface),
            "prior": float(lambda_prior),
        }
        self.use_depth_loss = bool(use_depth_loss)
        self.use_silhouette_loss = bool(use_silhouette_loss)
        self.use_surface_loss = bool(use_surface_loss)
        self.use_prior_preservation = bool(use_prior_preservation)
        self.surface_backend = surface_backend
        self.surface_blur = float(surface_blur)
        self.render_resolution = int(render_resolution)
        self.ray_samples = int(ray_samples)
        self.extensions_root = Path(extensions_root) if extensions_root else None

    def forward(
        self,
        *,
        v_final: torch.Tensor,
        v_target: torch.Tensor,
        v_base: torch.Tensor,
        gate: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        predicted_clean_grid: Optional[torch.Tensor] = None,
        ss_decoder: Optional[nn.Module] = None,
        gt_occ: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        view_valid_mask: Optional[torch.Tensor] = None,
        K_dataset: Optional[torch.Tensor] = None,
        c2w_dataset: Optional[torch.Tensor] = None,
        canonical_center: Optional[torch.Tensor] = None,
        canonical_half_extent: Optional[torch.Tensor] = None,
        vggt_depth: Optional[torch.Tensor] = None,
        vggt_depth_confidence: Optional[torch.Tensor] = None,
        alignment_scale: Optional[torch.Tensor] = None,
        alignment_quality: Optional[torch.Tensor] = None,
        depth_target: Optional[torch.Tensor] = None,
        depth_supervision_weight: Optional[torch.Tensor] = None,
        target_surface_points: Optional[torch.Tensor] = None,
        target_surface_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if v_final.shape != v_target.shape or v_final.shape != v_base.shape:
            raise ValueError("v_final, v_target, and v_base must have identical shapes")
        zero = v_final.new_zeros((), dtype=torch.float32)
        diagnostics: Dict[str, Any] = {}
        if sample_weight is None:
            sample_weight = torch.ones(v_final.shape[0], device=v_final.device)
        sample_weight = sample_weight.float().reshape(-1).clamp(0, 1)
        if sample_weight.shape[0] != v_final.shape[0]:
            raise ValueError(
                f"sample_weight must be [B], got {tuple(sample_weight.shape)}"
            )
        effective_valid_mask = torch.ones(
            v_final.shape[:2], device=v_final.device, dtype=torch.float32
        ) if valid_mask is None else valid_mask.float()
        effective_valid_mask = effective_valid_mask * sample_weight[:, None]
        loss_cfm = _masked_velocity_mse(v_final, v_target, effective_valid_mask)
        loss_base = _masked_velocity_mse(v_base, v_target, effective_valid_mask)
        diagnostics.update(
            {
                "cfm_available": True,
                "optimization_sample_weight": sample_weight.detach().mean(),
                "loss_base_cfm": loss_base.detach(),
                "adapter_cfm_gain": (loss_base - loss_cfm).detach(),
                "residual_target_norm": torch.linalg.vector_norm(
                    (v_target.float() - v_base.float()), dim=-1
                ).mean().detach(),
            }
        )

        occupancy_logits = None
        decoder_sample_valid = torch.ones(
            v_final.shape[0], device=v_final.device, dtype=torch.bool
        )
        if predicted_clean_grid is not None and ss_decoder is not None:
            raw_occupancy_logits = ss_decoder(predicted_clean_grid)
            if raw_occupancy_logits.ndim != 5 or raw_occupancy_logits.shape[1] != 1:
                raise RuntimeError(
                    "TRELLIS SS decoder must produce [B,1,D,H,W], "
                    f"got {tuple(raw_occupancy_logits.shape)}"
                )
            if raw_occupancy_logits.numel() == 0:
                raise RuntimeError("TRELLIS SS decoder violated its dense output-shape contract")
            finite = torch.isfinite(raw_occupancy_logits)
            decoder_sample_valid = finite.flatten(1).all(dim=1)
            decoder_logits_source = (
                raw_occupancy_logits
                if bool(decoder_sample_valid.all())
                else raw_occupancy_logits.detach()
            )
            occupancy_logits = torch.nan_to_num(
                decoder_logits_source.float(), nan=0.0, posinf=20.0, neginf=-20.0
            )
            active_fraction = (occupancy_logits > 0).float().flatten(1).mean(dim=1)
            diagnostics.update(
                {
                    "decoder_numerically_valid": bool(decoder_sample_valid.all().item()),
                    "decoder_finite_fraction": finite.float().mean().detach(),
                    "decoder_active_fraction": active_fraction.mean().detach(),
                    "decoder_empty_active": bool((active_fraction == 0).any().item()),
                    "decoder_logit_min": occupancy_logits.detach().amin(),
                    "decoder_logit_max": occupancy_logits.detach().amax(),
                }
            )
        occupancy_probability = occupancy_logits.sigmoid() if occupancy_logits is not None else None
        if occupancy_probability is not None:
            diagnostics["decoder_mean_occupancy_probability"] = (
                occupancy_probability.detach().float().mean()
            )

        if occupancy_probability is not None and gt_occ is not None:
            target_occ = _resize_occupancy(gt_occ.float(), occupancy_probability.shape[-3:])
            decoder_geometry_weight = decoder_sample_valid.float() * sample_weight
            loss_occ_bce = _weighted_batch_mean(
                F.binary_cross_entropy_with_logits(
                    occupancy_logits.float(), target_occ, reduction="none"
                ).flatten(1).mean(dim=1),
                decoder_geometry_weight,
            )
            loss_occ_dice = _weighted_batch_mean(
                _dice_loss_per_sample(occupancy_probability.float(), target_occ),
                decoder_geometry_weight,
            )
            loss_occupancy = loss_occ_bce + loss_occ_dice
            diagnostics.update(
                {
                    "occupancy_available": True,
                    "loss_occupancy_bce": loss_occ_bce.detach(),
                    "loss_occupancy_dice": loss_occ_dice.detach(),
                }
            )
        else:
            loss_occupancy = zero
            diagnostics["occupancy_available"] = False
            diagnostics["occupancy_unavailable_reason"] = "missing decoded logits or gt_occ"

        render_inputs = (occupancy_logits, masks, K_dataset, c2w_dataset, canonical_center, canonical_half_extent)
        rendered = None
        if (self.use_depth_loss or self.use_silhouette_loss) and all(value is not None for value in render_inputs):
            rendered = render_occupancy_volume(
                occupancy_logits,
                K_dataset,
                c2w_dataset,
                canonical_center,
                canonical_half_extent,
                output_resolution=self.render_resolution,
                ray_samples=self.ray_samples,
            )

        if self.use_silhouette_loss and rendered is not None and masks is not None:
            target_mask = _resize_view_maps(masks.float(), self.render_resolution)[:, :, 0]
            predicted_silhouette = rendered["silhouette"].clamp(1e-6, 1 - 1e-6)
            valid_ray = rendered["ray_box_valid"].float()
            if view_valid_mask is not None:
                valid_ray = valid_ray * view_valid_mask.float()[:, :, None, None]
            decoder_geometry_weight = decoder_sample_valid.float() * sample_weight
            valid_ray = valid_ray * decoder_geometry_weight[:, None, None, None]
            sil_bce = _probability_bce(predicted_silhouette.float(), target_mask.float())
            sil_bce = (sil_bce * valid_ray).sum() / valid_ray.sum().clamp_min(1)
            sil_dice = _weighted_batch_mean(
                _dice_loss_per_sample(
                    predicted_silhouette * valid_ray,
                    target_mask * valid_ray,
                ),
                decoder_geometry_weight,
            )
            loss_silhouette = sil_bce + sil_dice
            diagnostics.update(
                {
                    "silhouette_available": True,
                    "silhouette_path": "differentiable_volume_projection",
                    "nvdiffrast_used": False,
                }
            )
        else:
            loss_silhouette = zero
            diagnostics["silhouette_available"] = False
            diagnostics["silhouette_unavailable_reason"] = (
                "disabled" if not self.use_silhouette_loss else "missing occupancy/camera/mask modality"
            )

        aligned_depth_ready = (
            rendered is not None
            and depth_target is not None
            and depth_supervision_weight is not None
        )
        legacy_depth_ready = (
            rendered is not None
            and vggt_depth is not None
            and vggt_depth_confidence is not None
        )
        depth_ready = aligned_depth_ready or legacy_depth_ready
        if self.use_depth_loss and depth_ready:
            if aligned_depth_ready:
                target_depth_map = _resize_view_maps(
                    depth_target.float(), self.render_resolution
                )[:, :, 0]
                depth_mask = _resize_view_maps(
                    depth_supervision_weight[:, :, None].float(), self.render_resolution
                )[:, :, 0].clamp(0, 1)
            else:
                target_depth_map = _resize_view_maps(
                    vggt_depth.float(), self.render_resolution
                )[:, :, 0]
                target_confidence = _resize_view_maps(
                    vggt_depth_confidence[:, :, None].float(), self.render_resolution
                )[:, :, 0]
                if alignment_scale is None:
                    raise ValueError(
                        "Legacy depth reprojection requires explicit alignment_scale"
                    )
                target_depth_map = target_depth_map * alignment_scale[:, None, None, None]
                target_mask = _resize_view_maps(masks.float(), self.render_resolution)[:, :, 0]
                depth_mask = target_mask * target_confidence.clamp(0, 1)
                if alignment_quality is not None:
                    depth_mask = depth_mask * alignment_quality.float()[:, None, None, None]
            depth_mask = (
                depth_mask
                * (target_depth_map > 0).float()
                * rendered["depth_valid"].float()
                * (decoder_sample_valid.float() * sample_weight)[:, None, None, None]
            )
            if view_valid_mask is not None:
                depth_mask = depth_mask * view_valid_mask.float()[:, :, None, None]
            residual = rendered["depth"] - target_depth_map
            robust = torch.sqrt(residual.square() + 1e-6)
            loss_depth = (robust * depth_mask).sum() / depth_mask.sum().clamp_min(1e-8)
            diagnostics["depth_available"] = bool((depth_mask.sum() > 0).item())
            diagnostics["depth_supervision_weight"] = depth_mask.detach().mean()
        else:
            loss_depth = zero
            diagnostics["depth_available"] = False
            diagnostics["depth_unavailable_reason"] = (
                "disabled" if not self.use_depth_loss else "missing occupancy/camera/VGGT depth modality"
            )

        if (
            self.use_surface_loss
            and occupancy_probability is not None
            and target_surface_points is not None
            and bool(decoder_sample_valid.all())
            and bool((sample_weight > 0).all())
        ):
            loss_surface = self._surface_loss(occupancy_probability, target_surface_points, target_surface_weights)
            diagnostics.update({"surface_available": True, "surface_backend": self.surface_backend})
        else:
            loss_surface = zero
            diagnostics["surface_available"] = False
            diagnostics["surface_unavailable_reason"] = (
                "disabled" if not self.use_surface_loss else "missing occupancy or target point cloud"
            )

        if self.use_prior_preservation:
            gate_tensor = gate.float()
            if gate_tensor.ndim == v_final.ndim - 1:
                gate_tensor = gate_tensor.unsqueeze(-1)
            prior_per_sample = (
                (1.0 - gate_tensor) * (v_final.float() - v_base.float()).square()
            ).flatten(1).mean(dim=1)
            loss_prior = _weighted_batch_mean(prior_per_sample, sample_weight)
            diagnostics["prior_available"] = True
        else:
            loss_prior = zero
            diagnostics["prior_available"] = False

        total = (
            self.weights["cfm"] * loss_cfm
            + self.weights["depth"] * loss_depth
            + self.weights["silhouette"] * loss_silhouette
            + self.weights["occupancy"] * loss_occupancy
            + self.weights["surface"] * loss_surface
            + self.weights["prior"] * loss_prior
        )
        # The trainer makes one rank-synchronous decision before backward. Do
        # not throw locally here: a rank-local exception would strand peers in
        # DDP collectives. A non-finite objective is returned for the global
        # transaction to reject and log on every rank.
        diagnostics["objective_numerically_valid"] = bool(torch.isfinite(total).item())
        return {
            "loss_total": total,
            "loss_cfm": loss_cfm,
            "loss_depth": loss_depth,
            "loss_silhouette": loss_silhouette,
            "loss_occupancy": loss_occupancy,
            "loss_surface": loss_surface,
            "loss_prior": loss_prior,
            "diagnostics": diagnostics,
        }

    def _surface_loss(
        self,
        occupancy_probability: torch.Tensor,
        target_points: torch.Tensor,
        target_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.surface_backend == "geomloss":
            SamplesLoss = self._load_geomloss()
            probability = F.adaptive_avg_pool3d(occupancy_probability.float(), 16).flatten(2)[:, 0]
            centers = _voxel_centers(16, probability.device).expand(probability.shape[0], -1, -1)
            source_weights = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
            if target_weights is None:
                target_weights = torch.ones(target_points.shape[:2], device=target_points.device)
            target_weights = target_weights.float().clamp_min(0)
            target_weights = target_weights / target_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=self.surface_blur, backend="online")
            # GeomLoss/KeOps performs the large pairwise reduction without an NxM tensor.
            value = sinkhorn(
                source_weights[..., None],
                centers,
                target_weights[..., None],
                target_points.float(),
            )
            return value.mean()
        frnn = self._load_frnn()
        centers = _voxel_centers(16, occupancy_probability.device).expand(occupancy_probability.shape[0], -1, -1)
        lengths_source = torch.full((centers.shape[0],), centers.shape[1], device=centers.device, dtype=torch.long)
        lengths_target = torch.full((target_points.shape[0],), target_points.shape[1], device=centers.device, dtype=torch.long)
        distances, _, _, _ = frnn.frnn_grid_points(
            target_points.float(), centers.float(), lengths_target, lengths_source, K=1, r=2.0, return_nn=False
        )
        valid = distances[..., 0] >= 0
        if not bool(valid.any()):
            raise RuntimeError("FRNN surface query found no neighbors")
        return distances[..., 0][valid].mean()

    def _load_geomloss(self):
        if self.extensions_root is not None:
            for path in (
                self.extensions_root / "geomloss" / "src",
                self.extensions_root / "keops" / "keopscore",
                self.extensions_root / "keops" / "pykeops",
            ):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
        try:
            from geomloss import SamplesLoss
        except ImportError as exc:
            raise ImportError("Enabled surface loss requires geomloss (and pykeops for the online backend)") from exc
        return SamplesLoss

    def _load_frnn(self):
        if self.extensions_root is not None and str(self.extensions_root / "FRNN") not in sys.path:
            sys.path.insert(0, str(self.extensions_root / "FRNN"))
        try:
            import frnn
        except ImportError as exc:
            raise ImportError("surface_backend=frnn requires the installed FRNN extension") from exc
        return frnn


def render_occupancy_volume(
    occupancy_logits: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    center: torch.Tensor,
    half_extent: torch.Tensor,
    *,
    output_resolution: int,
    ray_samples: int,
) -> Dict[str, torch.Tensor]:
    """Vectorized differentiable alpha projection of a canonical occupancy grid."""
    B, V = K.shape[:2]
    K_render = K.float().clone()
    # Dataset views are square; scale against their principal point convention.
    source_w = (K_render[..., 0, 2] * 2.0).clamp_min(1.0)
    source_h = (K_render[..., 1, 2] * 2.0).clamp_min(1.0)
    K_render[..., 0, :] *= output_resolution / source_w[..., None]
    K_render[..., 1, :] *= output_resolution / source_h[..., None]
    y, x = torch.meshgrid(
        torch.arange(output_resolution, device=K.device, dtype=torch.float32) + 0.5,
        torch.arange(output_resolution, device=K.device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    x_cam = (x[None, None] - K_render[..., 0, 2, None, None]) / K_render[..., 0, 0, None, None]
    y_cam = (y[None, None] - K_render[..., 1, 2, None, None]) / K_render[..., 1, 1, None, None]
    camera_direction = torch.stack([x_cam, y_cam, torch.ones_like(x_cam)], dim=-1)
    world_direction = torch.einsum("bvij,bvhwj->bvhwi", c2w[..., :3, :3].float(), camera_direction)
    origin = c2w[..., :3, 3].float()[:, :, None, None]
    box_min = (center - half_extent)[:, None, None, None]
    box_max = (center + half_extent)[:, None, None, None]
    safe_direction = torch.where(world_direction.abs() < 1e-8, world_direction.sign() * 1e-8 + 1e-8, world_direction)
    t0 = (box_min - origin) / safe_direction
    t1 = (box_max - origin) / safe_direction
    near = torch.minimum(t0, t1).amax(dim=-1).clamp_min(0)
    far = torch.maximum(t0, t1).amin(dim=-1)
    ray_valid = far > near + 1e-6
    steps = torch.linspace(0, 1, ray_samples, device=K.device, dtype=torch.float32)
    depth_samples = near[..., None] + (far - near).clamp_min(0)[..., None] * steps
    world_samples = origin[..., None, :] + world_direction[..., None, :] * depth_samples[..., None]
    normalized = (world_samples - center[:, None, None, None, None]) / half_extent[:, None, None, None, None]
    grid = normalized.reshape(B * V, output_resolution, output_resolution, ray_samples, 3)
    density_grid = F.softplus(occupancy_logits.float())
    density_views = density_grid[:, None].expand(B, V, *density_grid.shape[1:]).reshape(B * V, *density_grid.shape[1:])
    density = F.grid_sample(
        density_views,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[:, 0].reshape(B, V, output_resolution, output_resolution, ray_samples)
    step_size = ((far - near).clamp_min(0) / max(ray_samples - 1, 1))[..., None]
    alpha = 1.0 - torch.exp(-density * step_size)
    alpha = alpha * ray_valid[..., None]
    transmittance = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), (1.0 - alpha + 1e-7)], dim=-1), dim=-1
    )[..., :-1]
    weights = alpha * transmittance
    silhouette = weights.sum(dim=-1).clamp(0, 1)
    depth = (weights * depth_samples).sum(dim=-1) / silhouette.clamp_min(1e-8)
    return {
        "silhouette": silhouette,
        "depth": depth,
        "ray_box_valid": ray_valid,
        "depth_valid": ray_valid & (silhouette > 1e-5),
    }


def _masked_velocity_mse(prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    error = (prediction.float() - target.float()).square()
    if mask is None:
        return error.mean()
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    weighted = error * mask.float()
    return weighted.sum() / (mask.float().sum() * error.shape[-1]).clamp_min(1)


def _dice_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _dice_loss_per_sample(prediction, target).mean()


def _dice_loss_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().flatten(1)
    target = target.float().flatten(1)
    return 1.0 - (2.0 * (prediction * target).sum(dim=1) + 1e-6) / (
        prediction.sum(dim=1) + target.sum(dim=1) + 1e-6
    )


def _weighted_batch_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Average available samples while preserving a zero-gradient graph if none are valid."""
    values = values.float().reshape(-1)
    weights = weights.to(device=values.device, dtype=values.dtype).reshape(-1).clamp_min(0)
    if values.shape != weights.shape:
        raise ValueError(
            f"Per-sample value/weight mismatch: {tuple(values.shape)} vs {tuple(weights.shape)}"
        )
    return (values * weights).sum() / weights.sum().clamp_min(1)


def _probability_bce(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Elementwise FP32 BCE for non-logit renderer probabilities under AMP."""
    prediction = prediction.float().clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    return -(target * prediction.log() + (1.0 - target) * torch.log1p(-prediction))


def _resize_occupancy(occupancy: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    if occupancy.ndim == 4:
        occupancy = occupancy[:, None]
    if occupancy.shape[-3:] == target_shape:
        return occupancy
    return F.interpolate(occupancy, size=target_shape, mode="nearest")


def _resize_view_maps(tensor: torch.Tensor, resolution: int) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[:, :, None]
    B, V, C, H, W = tensor.shape
    resized = F.interpolate(tensor.reshape(B * V, C, H, W), (resolution, resolution), mode="bilinear", align_corners=False)
    return resized.reshape(B, V, C, resolution, resolution)


def _voxel_centers(resolution: int, device: torch.device) -> torch.Tensor:
    axis = (torch.arange(resolution, device=device, dtype=torch.float32) + 0.5) * (2.0 / resolution) - 1.0
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)


@dataclass(frozen=True)
class DecodedSLatGeometricLossConfig:
    rgb_weight: float = 1.0
    silhouette_weight: float = 0.5
    depth_weight: float = 0.1
    feature_weight: float = 0.25
    cross_view_rgb_weight: float = 0.25
    sinkhorn_weight: float = 0.1
    feature_grid_size: int = 12
    sinkhorn_points: int = 2048
    sinkhorn_blur: float = 0.02

    def __post_init__(self) -> None:
        weights = (
            self.rgb_weight,
            self.silhouette_weight,
            self.depth_weight,
            self.feature_weight,
            self.cross_view_rgb_weight,
            self.sinkhorn_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("decoded SLat loss weights must be non-negative")
        if self.feature_grid_size < 2 or self.sinkhorn_points < 1 or self.sinkhorn_blur <= 0:
            raise ValueError("decoded SLat sampling sizes and Sinkhorn blur must be positive")


class DecodedSLatGeometricLoss(nn.Module):
    """End-to-end decoded SLat rendering, reprojection, and Sinkhorn loss."""

    def __init__(
        self,
        config: Optional[DecodedSLatGeometricLossConfig] = None,
        *,
        extensions_root: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.config = config or DecodedSLatGeometricLossConfig()
        if extensions_root is not None:
            root = Path(extensions_root)
            for path in (
                root / "geomloss" / "src",
                root / "keops" / "keopscore",
                root / "keops" / "pykeops",
            ):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
        require_dependency(
            "geomloss",
            real_mode=True,
            feature="decoded SLat geometric locking",
        )
        from geomloss import SamplesLoss

        self.sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=self.config.sinkhorn_blur,
            backend="online",
        )

    def forward(
        self,
        *,
        rendered_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        rendered_alpha: torch.Tensor,
        target_mask: torch.Tensor,
        rendered_depth: torch.Tensor,
        target_depth: torch.Tensor,
        high_features: torch.Tensor,
        aligned_point_map: torch.Tensor,
        vggt_confidence: torch.Tensor,
        intrinsics: torch.Tensor,
        world_to_camera: torch.Tensor,
        decoded_surface_points: torch.Tensor,
        decoded_surface_weights: Optional[torch.Tensor] = None,
        view_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        predicted_rgb = _render_to_bvchw(rendered_rgb, 3, "rendered_rgb").float()
        predicted_alpha = _render_to_bvchw(rendered_alpha, 1, "rendered_alpha").float()
        predicted_depth = _render_to_bvchw(rendered_depth, 1, "rendered_depth").float()
        target_rgb = _render_to_bvchw(target_rgb, 3, "target_rgb").float()
        target_mask = _render_to_bvchw(target_mask, 1, "target_mask").float()
        target_depth = _render_to_bvchw(target_depth, 1, "target_depth").float()
        batch_size, views = target_rgb.shape[:2]
        if predicted_rgb.shape[:2] != (batch_size, views):
            raise ValueError("rendered and target view axes differ")
        if view_valid_mask is None:
            view_valid_mask = torch.ones(
                batch_size, views, device=target_rgb.device, dtype=torch.bool
            )
        view_weight = view_valid_mask[:, :, None, None, None].float()
        foreground = target_mask.clamp(0.0, 1.0) * view_weight
        rgb_error = (predicted_rgb - target_rgb).abs()
        loss_rgb = (rgb_error * foreground).sum() / (
            foreground.sum() * predicted_rgb.shape[2]
        ).clamp_min(1.0)
        alpha = predicted_alpha.clamp(1.0e-5, 1.0 - 1.0e-5)
        mask_bce = -(target_mask * alpha.log() + (1.0 - target_mask) * torch.log1p(-alpha))
        loss_silhouette = (mask_bce * view_weight).sum() / view_weight.expand_as(mask_bce).sum().clamp_min(1.0)
        depth_valid = (
            (target_depth > 1.0e-5)
            & torch.isfinite(target_depth)
            & (foreground > 0.5)
        )
        depth_robust = torch.sqrt((predicted_depth - target_depth).square() + 1.0e-6)
        loss_depth = depth_robust.masked_select(depth_valid).mean() if bool(depth_valid.any()) else predicted_depth.sum() * 0.0

        consistency = decoded_surface_cross_view_consistency(
            images=target_rgb,
            high_features=high_features,
            confidence=vggt_confidence,
            masks=target_mask,
            intrinsics=intrinsics,
            world_to_camera=world_to_camera,
            decoded_surface_points=decoded_surface_points,
            decoded_surface_weights=decoded_surface_weights,
            view_valid_mask=view_valid_mask,
            max_points=self.config.feature_grid_size**2,
        )
        loss_sinkhorn = self._sinkhorn_geometry(
            decoded_surface_points,
            decoded_surface_weights,
            aligned_point_map,
            vggt_confidence,
            target_mask,
        )
        total = (
            self.config.rgb_weight * loss_rgb
            + self.config.silhouette_weight * loss_silhouette
            + self.config.depth_weight * loss_depth
            + self.config.feature_weight * consistency["feature"]
            + self.config.cross_view_rgb_weight * consistency["rgb"]
            + self.config.sinkhorn_weight * loss_sinkhorn
        )
        return {
            "loss": total,
            "loss_rgb": loss_rgb,
            "loss_silhouette": loss_silhouette,
            "loss_depth": loss_depth,
            "loss_feature_reprojection": consistency["feature"],
            "loss_rgb_reprojection": consistency["rgb"],
            "loss_sinkhorn": loss_sinkhorn,
            "reprojection_support": consistency["support"],
        }

    def _sinkhorn_geometry(
        self,
        decoded_points: torch.Tensor,
        decoded_weights: Optional[torch.Tensor],
        point_map: torch.Tensor,
        confidence: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if decoded_points.ndim != 3 or decoded_points.shape[-1] != 3:
            raise ValueError(
                f"decoded_surface_points must be [B,P,3], got {tuple(decoded_points.shape)}"
            )
        batch_size, predicted_count = decoded_points.shape[:2]
        point_map = _render_to_bvchw(point_map, 3, "aligned_point_map").float()
        confidence = _render_to_bvchw(confidence, 1, "vggt_confidence").float()
        masks = _render_to_bvchw(masks, 1, "target_mask").float()
        target_points = point_map.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 3)
        target_score = (confidence * masks).reshape(batch_size, -1)
        target_score = target_score * torch.isfinite(target_points).all(dim=-1).float()
        target_count = min(self.config.sinkhorn_points, target_points.shape[1])
        target_values, target_index = target_score.topk(target_count, dim=1, largest=True, sorted=False)
        target = torch.gather(
            target_points,
            1,
            target_index[..., None].expand(-1, -1, 3),
        )

        if decoded_weights is None:
            decoded_weights = torch.ones(
                batch_size, predicted_count, device=decoded_points.device, dtype=decoded_points.dtype
            )
        if decoded_weights.shape != (batch_size, predicted_count):
            raise ValueError(
                f"decoded_surface_weights must be [B,P], got {tuple(decoded_weights.shape)}"
            )
        predicted_score = decoded_weights.float().clamp_min(0.0)
        predicted_score = predicted_score * torch.isfinite(decoded_points).all(dim=-1).float()
        selected_count = min(self.config.sinkhorn_points, predicted_count)
        predicted_values, predicted_index = predicted_score.topk(
            selected_count, dim=1, largest=True, sorted=False
        )
        predicted = torch.gather(
            decoded_points.float(),
            1,
            predicted_index[..., None].expand(-1, -1, 3),
        )
        target_values = _normalize_positive_measure(target_values)
        predicted_values = _normalize_positive_measure(predicted_values)
        loss = self.sinkhorn(
            predicted_values[..., None],
            torch.nan_to_num(predicted),
            target_values[..., None],
            torch.nan_to_num(target),
        )
        valid = (target_score.sum(dim=1) > 0) & (predicted_score.sum(dim=1) > 0)
        return (loss.reshape(-1) * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def pixel_aligned_cross_view_consistency(
    *,
    images: torch.Tensor,
    high_features: torch.Tensor,
    aligned_point_map: torch.Tensor,
    confidence: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
    view_valid_mask: torch.Tensor,
    grid_size: int,
) -> Dict[str, torch.Tensor]:
    """Compare RGB and high-level features at shared reprojected 3-D points."""

    images = _render_to_bvchw(images, 3, "images").float()
    high_features = _render_to_bvchw(
        high_features, high_features.shape[2], "high_features"
    ).float()
    point_map = _render_to_bvchw(aligned_point_map, 3, "aligned_point_map").float()
    confidence = _render_to_bvchw(confidence, 1, "confidence").float()
    masks = _render_to_bvchw(masks, 1, "masks").float()
    batch_size, views = images.shape[:2]
    size = (int(grid_size), int(grid_size))
    source_points = _resize_bv(point_map, size).permute(0, 1, 3, 4, 2).reshape(
        batch_size, views * grid_size * grid_size, 3
    )
    source_confidence = (
        _resize_bv(confidence * masks, size)
        .permute(0, 1, 3, 4, 2)
        .reshape(batch_size, views * grid_size * grid_size, 1)
    )
    source_rgb = _resize_bv(images, size).permute(0, 1, 3, 4, 2).reshape(
        batch_size, views * grid_size * grid_size, 3
    )
    source_features = _resize_bv(high_features, size).permute(0, 1, 3, 4, 2).reshape(
        batch_size, views * grid_size * grid_size, high_features.shape[2]
    )
    projection = project_opencv_points_kaolin(
        source_points,
        intrinsics,
        world_to_camera,
        images.shape[-2],
        images.shape[-1],
    )
    target_rgb = _sample_bv_at_projection(images, projection.grid)
    target_features = _sample_bv_at_projection(high_features, projection.grid)
    target_mask = _sample_bv_at_projection(masks, projection.grid)
    points_per_view = grid_size * grid_size
    source_view = torch.arange(views, device=images.device).repeat_interleave(points_per_view)
    target_view = torch.arange(views, device=images.device)
    cross_view = source_view[:, None] != target_view[None]
    support = (
        source_confidence[:, :, None]
        * projection.valid.float()
        * (target_mask > 0.5).float()
        * cross_view[None, :, :, None].float()
        * view_valid_mask[:, source_view, None, None].float()
        * view_valid_mask[:, None, :, None].float()
    )
    feature_error = (
        target_features - source_features[:, :, None]
    ).square().mean(dim=-1, keepdim=True)
    rgb_error = (
        target_rgb - source_rgb[:, :, None]
    ).square().mean(dim=-1, keepdim=True)
    denominator = support.sum().clamp_min(1.0)
    return {
        "feature": (feature_error * support).sum() / denominator,
        "rgb": (rgb_error * support).sum() / denominator,
        "support": support.mean(),
    }


def decoded_surface_cross_view_consistency(
    *,
    images: torch.Tensor,
    high_features: torch.Tensor,
    confidence: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
    decoded_surface_points: torch.Tensor,
    decoded_surface_weights: Optional[torch.Tensor],
    view_valid_mask: torch.Tensor,
    max_points: int,
) -> Dict[str, torch.Tensor]:
    """Compare view observations at projections of the predicted surface.

    Unlike a consistency statistic built only from fixed VGGT point maps, this
    objective is differentiable with respect to the decoded surface positions.
    It therefore supplies an actual training signal to the SLat decoder path.
    """

    images = _render_to_bvchw(images, 3, "images").float()
    high_features = _render_to_bvchw(
        high_features, high_features.shape[2], "high_features"
    ).float()
    confidence = _render_to_bvchw(confidence, 1, "confidence").float()
    masks = _render_to_bvchw(masks, 1, "masks").float()
    points = decoded_surface_points.float()
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(
            f"decoded_surface_points must be [B,P,3], got {tuple(points.shape)}"
        )
    batch_size, point_count = points.shape[:2]
    if images.shape[0] != batch_size or max_points < 1:
        raise ValueError("decoded surface batch and positive max_points are required")
    selected_count = min(int(max_points), point_count)
    if decoded_surface_weights is None:
        selection_score = torch.ones(
            batch_size, point_count, device=points.device, dtype=points.dtype
        )
    else:
        if decoded_surface_weights.shape != (batch_size, point_count):
            raise ValueError(
                "decoded_surface_weights must match decoded_surface_points [B,P]"
            )
        selection_score = decoded_surface_weights.float().clamp_min(0.0)
    selection_score = selection_score * torch.isfinite(points).all(dim=-1).float()
    selected_score, selected_index = selection_score.topk(
        selected_count, dim=1, largest=True, sorted=False
    )
    selected_points = torch.gather(
        points,
        1,
        selected_index[..., None].expand(-1, -1, 3),
    )
    projection = project_opencv_points_kaolin(
        torch.nan_to_num(selected_points),
        intrinsics,
        world_to_camera,
        images.shape[-2],
        images.shape[-1],
    )
    sampled_rgb = _sample_bv_at_projection(images, projection.grid)
    sampled_features = F.normalize(
        _sample_bv_at_projection(high_features, projection.grid),
        dim=-1,
        eps=1.0e-6,
    )
    sampled_confidence = _sample_bv_at_projection(confidence, projection.grid)
    sampled_mask = _sample_bv_at_projection(masks, projection.grid)
    view_support = (
        sampled_confidence.clamp_min(0.0)
        * sampled_mask.clamp(0.0, 1.0)
        * projection.valid.float()
        * view_valid_mask[:, None, :, None].float()
        * selected_score[:, :, None, None].clamp(0.0, 1.0)
    )
    views = images.shape[1]
    cross_view = (~torch.eye(views, device=images.device, dtype=torch.bool))[None, None, :, :, None]
    pair_support = (
        view_support[:, :, :, None]
        * view_support[:, :, None, :]
        * cross_view.float()
    )
    feature_error = (
        sampled_features[:, :, :, None] - sampled_features[:, :, None, :]
    ).square().mean(dim=-1, keepdim=True)
    rgb_error = (
        sampled_rgb[:, :, :, None] - sampled_rgb[:, :, None, :]
    ).square().mean(dim=-1, keepdim=True)
    denominator = pair_support.sum().clamp_min(1.0e-8)
    return {
        "feature": (feature_error * pair_support).sum() / denominator,
        "rgb": (rgb_error * pair_support).sum() / denominator,
        "support": pair_support.mean(),
    }


def _resize_bv(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch_size, views, channels = value.shape[:3]
    resized = F.interpolate(
        value.reshape(batch_size * views, channels, *value.shape[-2:]),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch_size, views, channels, *size)


def _sample_bv_at_projection(maps: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    batch_size, views, channels, height, width = maps.shape
    point_count = grid.shape[1]
    flat_maps = maps.reshape(batch_size * views, channels, height, width)
    flat_grid = grid.permute(0, 2, 1, 3).reshape(batch_size * views, point_count, 1, 2)
    sampled = F.grid_sample(
        flat_maps,
        flat_grid.to(flat_maps.dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[..., 0].transpose(1, 2).reshape(
        batch_size, views, point_count, channels
    ).permute(0, 2, 1, 3).contiguous()


def _render_to_bvchw(value: torch.Tensor, channels: int, name: str) -> torch.Tensor:
    if value.ndim == 4 and channels == 1:
        value = value[:, :, None]
    if value.ndim != 5:
        raise ValueError(f"{name} must be five-dimensional, got {tuple(value.shape)}")
    if value.shape[2] == channels:
        return value
    if value.shape[-1] == channels:
        return value.permute(0, 1, 4, 2, 3).contiguous()
    raise ValueError(f"{name} must have {channels} channels, got {tuple(value.shape)}")


def _normalize_positive_measure(weights: torch.Tensor) -> torch.Tensor:
    positive = weights.float().clamp_min(0.0)
    missing = positive.sum(dim=1, keepdim=True) <= 0
    fallback = torch.zeros_like(positive)
    fallback[:, 0] = 1.0
    positive = torch.where(missing, fallback, positive)
    return positive / positive.sum(dim=1, keepdim=True).clamp_min(1.0e-8)


__all__ = [
    "DecodedSLatGeometricLoss",
    "DecodedSLatGeometricLossConfig",
    "FlowMatchingLossBuilder",
    "decoded_surface_cross_view_consistency",
    "pixel_aligned_cross_view_consistency",
    "render_occupancy_volume",
]
