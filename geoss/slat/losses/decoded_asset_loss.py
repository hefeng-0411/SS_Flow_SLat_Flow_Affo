"""Decoded TRELLIS SLat supervision through gsplat, Kaolin, and GeomLoss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch

from geoss.losses.geometric_loss import (
    DecodedSLatGeometricLoss,
    DecodedSLatGeometricLossConfig,
)
from geoss.renderers.gsplat_renderer import render_gaussians
from geoss.slat.utils.normalization import denormalize_slat


def flow_x0_from_velocity(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor,
    sigma_min: float,
) -> torch.Tensor:
    """Invert TRELLIS' linear flow path: x0=(1-sigma_min)x_t-s(t)v."""

    if x_t.shape != velocity.shape:
        raise ValueError("x_t and velocity must have the same shape")
    t = timestep.to(device=x_t.device, dtype=x_t.dtype).reshape(
        x_t.shape[0], *([1] * (x_t.ndim - 1))
    )
    sigma = float(sigma_min) + (1.0 - float(sigma_min)) * t
    return (1.0 - float(sigma_min)) * x_t - sigma * velocity


@dataclass
class DecodedAssetLossConfig:
    enabled: bool = False
    every: int = 4
    views_per_object: int = 2
    rgb_weight: float = 1.0
    foreground_rgb_weight: float = 0.0
    ssim_weight: float = 0.0
    lpips_weight: float = 0.0
    mask_weight: float = 0.5
    depth_weight: float = 0.1
    geometry_weight: float = 0.1
    feature_weight: float = 0.25
    cross_view_rgb_weight: float = 0.25
    geometry_points: int = 2048
    feature_grid_size: int = 12
    sinkhorn_blur: float = 0.02
    extensions_root: str = "/mnt/sda/hf/MVG/Base/extensions"
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.every < 1 or self.views_per_object < 1 or self.geometry_points < 1:
            raise ValueError("decoded supervision cadence and sample counts must be positive")
        if self.feature_grid_size < 2 or self.sinkhorn_blur <= 0:
            raise ValueError("feature grid and Sinkhorn blur must be positive")
        weights = (
            self.rgb_weight,
            self.foreground_rgb_weight,
            self.ssim_weight,
            self.lpips_weight,
            self.mask_weight,
            self.depth_weight,
            self.geometry_weight,
            self.feature_weight,
            self.cross_view_rgb_weight,
        )
        if any(float(value) < 0 for value in weights):
            raise ValueError("decoded supervision weights must be non-negative")
        if len(self.background) != 3 or any(
            not 0.0 <= float(value) <= 1.0 for value in self.background
        ):
            raise ValueError("decoded supervision background must contain three values in [0,1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "DecodedAssetLossConfig":
        value = dict(value or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise KeyError(f"Unknown decoded-supervision options: {sorted(unknown)}")
        if "background" in value:
            value["background"] = tuple(float(v) for v in value["background"])
        return cls(**value)


class DecodedAssetSupervisor:
    """Decode SLat, render all selected views, and apply geometric locking."""

    def __init__(self, pipeline, config: Mapping[str, Any] | None = None) -> None:
        self.pipeline = pipeline
        self.config = DecodedAssetLossConfig.from_mapping(config)
        self.objective = None
        if self.config.enabled:
            if "slat_decoder_gs" not in pipeline.models:
                raise RuntimeError("Decoded asset supervision requires TRELLIS slat_decoder_gs")
            self.objective = DecodedSLatGeometricLoss(
                DecodedSLatGeometricLossConfig(
                    rgb_weight=self.config.rgb_weight,
                    silhouette_weight=self.config.mask_weight,
                    depth_weight=self.config.depth_weight,
                    feature_weight=self.config.feature_weight,
                    cross_view_rgb_weight=self.config.cross_view_rgb_weight,
                    sinkhorn_weight=self.config.geometry_weight,
                    feature_grid_size=self.config.feature_grid_size,
                    sinkhorn_points=self.config.geometry_points,
                    sinkhorn_blur=self.config.sinkhorn_blur,
                ),
                extensions_root=self.config.extensions_root,
            )

    def __call__(self, out: dict, batch: dict, step: int) -> Dict[str, torch.Tensor]:
        velocity = out.get("v_slat_geo", out.get("velocity"))
        velocity = getattr(velocity, "feats", velocity)
        if not isinstance(velocity, torch.Tensor):
            raise TypeError("Decoded supervision requires a tensor SLat velocity")
        anchor = velocity.sum() * 0.0
        if not self.config.enabled or step % self.config.every != 0:
            return {
                "loss": anchor,
                "enabled": torch.tensor(False, device=anchor.device),
                "applied": torch.tensor(False, device=anchor.device),
            }
        if self.objective is None:
            raise RuntimeError("Decoded SLat objective was not initialized")

        batch_size = int(batch["slat_latent_tokens"].shape[0])
        object_index = (step // self.config.every) % batch_size
        x0_pred = flow_x0_from_velocity(
            batch["slat_latent_tokens"],
            velocity,
            batch["timestep"],
            float(batch["flow_sigma_min"]),
        )
        raw_pred = denormalize_slat(x0_pred, self.pipeline.slat_normalization)
        sparse = _padded_to_sparse(
            raw_pred[object_index : object_index + 1],
            batch["slat_indices"][object_index : object_index + 1],
            batch.get("slat_token_valid_mask")[object_index : object_index + 1]
            if isinstance(batch.get("slat_token_valid_mask"), torch.Tensor)
            else None,
        )
        decoded = self.pipeline.models["slat_decoder_gs"](sparse)
        if len(decoded) != 1:
            raise RuntimeError(f"TRELLIS Gaussian decoder returned {len(decoded)} objects for microbatch 1")
        gaussian = decoded[0]
        available_views = int(batch["images"].shape[1])
        view_valid = batch.get("view_valid_mask")
        if isinstance(view_valid, torch.Tensor):
            available = torch.nonzero(
                view_valid[object_index].bool(), as_tuple=False
            ).flatten()
        else:
            available = torch.arange(available_views, device=batch["images"].device)
        if available.numel() == 0:
            raise RuntimeError(
                f"Decoded supervision object {object_index} has no valid camera views"
            )
        selected_count = min(self.config.views_per_object, int(available.numel()))
        selection_positions = torch.linspace(
            0,
            int(available.numel()) - 1,
            selected_count,
            device=batch["images"].device,
        ).round().long()
        selected = available.index_select(0, selection_positions)
        cameras = {
            "K": batch["K"][object_index, selected],
            "w2c": batch["w2c"][object_index, selected],
        }
        backgrounds = batch["images"].new_tensor(self.config.background).view(1, 3).expand(
            selected_count, -1
        )
        rendered = render_gaussians(
            gaussian,
            cameras,
            tuple(batch["images"].shape[-2:]),
            backgrounds=backgrounds,
            return_alpha=True,
            return_depth=True,
        )
        high_features = batch.get("dino_spatial_features", batch.get("vggt_features"))
        if not isinstance(high_features, torch.Tensor):
            raise KeyError("Decoded SLat supervision requires DINO or VGGT spatial features")
        point_map = batch.get("aligned_pointmap")
        target_depth = batch.get("aligned_depth", batch.get("depths"))
        confidence = batch.get("vggt_confidence", batch.get("alignment_confidence"))
        if not all(isinstance(value, torch.Tensor) for value in (point_map, target_depth, confidence)):
            raise KeyError("Decoded SLat supervision requires aligned point/depth/confidence maps")
        terms = self.objective(
            rendered_rgb=rendered["rendered_rgb"][None],
            target_rgb=batch["images"][object_index : object_index + 1, selected],
            rendered_alpha=rendered["rendered_alpha"][None],
            target_mask=batch["masks"][object_index : object_index + 1, selected],
            rendered_depth=rendered["rendered_depth"][None],
            target_depth=target_depth[object_index : object_index + 1, selected],
            high_features=high_features[object_index : object_index + 1, selected],
            aligned_point_map=point_map[object_index : object_index + 1, selected],
            vggt_confidence=confidence[object_index : object_index + 1, selected],
            intrinsics=batch["K"][object_index : object_index + 1, selected],
            world_to_camera=batch["w2c"][object_index : object_index + 1, selected],
            decoded_surface_points=gaussian.get_xyz[None],
            decoded_surface_weights=gaussian.get_opacity.reshape(1, -1),
            view_valid_mask=torch.ones(
                1, selected_count, device=anchor.device, dtype=torch.bool
            ),
        )
        return {
            **terms,
            "render_loss": terms["loss_rgb"] + terms["loss_silhouette"] + terms["loss_depth"],
            "geometry_loss": terms["loss_sinkhorn"],
            "enabled": torch.tensor(True, device=anchor.device),
            "applied": torch.tensor(True, device=anchor.device),
        }


def _padded_to_sparse(
    feats: torch.Tensor,
    indices: torch.Tensor,
    valid_mask: torch.Tensor | None,
):
    from trellis.modules import sparse as sp

    if feats.ndim != 3 or indices.shape != (*feats.shape[:2], 3):
        raise ValueError("Decoded SLat tensors must be feats [B,L,C] and indices [B,L,3]")
    if feats.shape[0] != 1:
        raise RuntimeError("Sparse decoder conversion requires per-rank microbatch size 1")
    if valid_mask is None:
        valid_mask = torch.ones(*feats.shape[:2], 1, device=feats.device, dtype=torch.bool)
    valid = valid_mask[0, :, 0].bool()
    if not bool(valid.any()):
        raise RuntimeError("Decoded supervision received no valid SLat tokens")
    selected_features = feats[0, valid]
    batch_column = torch.zeros(
        selected_features.shape[0], 1, device=indices.device, dtype=indices.dtype
    )
    coords = torch.cat([batch_column, indices[0, valid]], dim=-1).int().contiguous()
    return sp.SparseTensor(feats=selected_features.contiguous(), coords=coords)


__all__ = [
    "DecodedAssetLossConfig",
    "DecodedAssetSupervisor",
    "flow_x0_from_velocity",
]
