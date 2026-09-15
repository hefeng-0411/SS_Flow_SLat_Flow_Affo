"""Pixel-aligned symmetric conditioning for the complete TRELLIS SLat flow."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import types
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.geometry.kaolin_camera import project_opencv_points_kaolin
from geoss.models.affostruction_conditioning import (
    VoxelPositionalEncoding3D,
    compact_condition_tokens,
)
from geoss.slat.utils.active_voxel_utils import indices_to_active_xyz


@dataclass(frozen=True)
class SLatConditioningOutput:
    condition: torch.Tensor
    condition_valid: torch.Tensor
    active_xyz: torch.Tensor
    projection_grid: torch.Tensor
    sampled_high: torch.Tensor
    sampled_low: torch.Tensor
    view_weights: torch.Tensor
    confidence: torch.Tensor
    occlusion_mask: torch.Tensor
    agreement: torch.Tensor


@dataclass(frozen=True)
class AffostructionSLatFlowOutput:
    velocity: object
    base_velocity: object
    raw_correction: object
    correction: object
    conditioning: SLatConditioningOutput


AFFOSTRUCTION_IMAGE_SLAT_ARCHITECTURE = (
    "vggt_affostruction_sparse_residual_slat_flow_768x12_v1"
)
AFFOSTRUCTION_IMAGE_SLAT_CONFIG = {
    "resolution": 64,
    "model_channels": 768,
    "num_blocks": 12,
    "num_heads": 12,
    "mlp_ratio": 4,
    "patch_size": 2,
    "num_io_res_blocks": 2,
    "io_block_channels": [128],
    "pe_mode": "ape",
    "qk_rms_norm": True,
}


def build_affostruction_image_slat_denoiser(
    affostruction_root: str | Path,
    *,
    slat_channels: int,
    condition_channels: int,
    gradient_checkpointing: bool,
) -> nn.Module:
    root = Path(affostruction_root).resolve()
    source = root / "affostruction/models/structured_latent_flow.py"
    if not source.is_file():
        raise FileNotFoundError(f"Affostruction sparse-flow source not found: {source}")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPCONV_ALGO", "native")
    if "affostruction" not in sys.modules:
        package = types.ModuleType("affostruction")
        package.__path__ = [str(root / "affostruction")]
        package.__package__ = "affostruction"
        sys.modules["affostruction"] = package
    module = importlib.import_module("affostruction.models.structured_latent_flow")
    model = module.SLatFlowModel(
        **AFFOSTRUCTION_IMAGE_SLAT_CONFIG,
        in_channels=int(slat_channels),
        out_channels=int(slat_channels),
        cond_channels=int(condition_channels),
        use_fp16=True,
        use_checkpoint=bool(gradient_checkpointing),
    )
    model.convert_to_fp32()
    return model


class ShallowRGBEncoder(nn.Module):
    """Low-frequency geometry and texture map used beside DINO semantics."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        hidden = max(32, int(channels))
        self.network = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class SymmetricSLatConditioner(nn.Module):
    """Reproject, sample, and confidence-fuse multi-view SLat evidence."""

    def __init__(
        self,
        *,
        resolution: int = 64,
        high_feature_dim: int = 1024,
        low_feature_dim: int = 64,
        condition_dim: int = 1024,
        depth_temperature: float = 0.04,
        occlusion_margin: float = 0.03,
        minimum_incidence: float = 0.05,
    ) -> None:
        super().__init__()
        if high_feature_dim < 1 or low_feature_dim < 1 or condition_dim < 1:
            raise ValueError("feature and condition dimensions must be positive")
        if depth_temperature <= 0 or occlusion_margin < 0:
            raise ValueError("depth_temperature must be positive and occlusion_margin non-negative")
        self.resolution = int(resolution)
        self.high_feature_dim = int(high_feature_dim)
        self.low_feature_dim = int(low_feature_dim)
        self.condition_dim = int(condition_dim)
        self.depth_temperature = float(depth_temperature)
        self.occlusion_margin = float(occlusion_margin)
        self.minimum_incidence = float(minimum_incidence)
        self.low_encoder = ShallowRGBEncoder(low_feature_dim)
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(high_feature_dim + low_feature_dim),
            nn.Linear(high_feature_dim + low_feature_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.position = VoxelPositionalEncoding3D(condition_dim, resolution)

    def forward(
        self,
        *,
        active_indices: torch.Tensor,
        images: torch.Tensor,
        high_features: torch.Tensor,
        intrinsics: torch.Tensor,
        world_to_camera: torch.Tensor,
        foreground_masks: torch.Tensor,
        aligned_depth: torch.Tensor,
        vggt_confidence: torch.Tensor,
        active_valid_mask: Optional[torch.Tensor] = None,
        view_valid_mask: Optional[torch.Tensor] = None,
        aligned_point_map: Optional[torch.Tensor] = None,
    ) -> SLatConditioningOutput:
        if active_indices.ndim != 3 or active_indices.shape[-1] not in {3, 4}:
            raise ValueError(f"active_indices must be [B,L,3/4], got {tuple(active_indices.shape)}")
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"images must be [B,V,3,H,W], got {tuple(images.shape)}")
        batch_size, views, _, height, width = images.shape
        active_xyz = indices_to_active_xyz(active_indices, self.resolution).to(
            device=images.device, dtype=torch.float32
        )
        if high_features.ndim != 5 or high_features.shape[:2] != (batch_size, views):
            raise ValueError(
                f"high_features must be [B,V,C,Hf,Wf], got {tuple(high_features.shape)}"
            )
        if high_features.shape[2] != self.high_feature_dim:
            raise ValueError(
                f"high feature width {high_features.shape[2]} != configured {self.high_feature_dim}"
            )
        masks = _as_bvchw(foreground_masks, channels=1, name="foreground_masks").float()
        depths = _as_bvchw(aligned_depth, channels=1, name="aligned_depth").float()
        confidence_maps = _as_bvchw(vggt_confidence, channels=1, name="vggt_confidence").float()
        if masks.shape[:2] != (batch_size, views) or depths.shape[:2] != (batch_size, views):
            raise ValueError("mask/depth view axes must match images")

        projection = project_opencv_points_kaolin(
            active_xyz,
            intrinsics,
            world_to_camera,
            height,
            width,
        )
        grid = projection.grid
        flat_images = images.float().reshape(batch_size * views, 3, height, width)
        low_maps = self.low_encoder(flat_images).reshape(
            batch_size, views, self.low_feature_dim, height, width
        )
        sampled_high = _sample_view_maps(high_features.float(), grid)
        sampled_low = _sample_view_maps(low_maps, grid)
        sampled_mask = _sample_view_maps(masks, grid).clamp(0.0, 1.0)
        sampled_depth = _sample_view_maps(depths, grid)
        sampled_confidence = _sample_view_maps(confidence_maps, grid).clamp(0.0, 1.0)

        depth_scale = sampled_depth.abs().clamp_min(1.0e-3)
        signed_depth = projection.depth - sampled_depth
        depth_residual = signed_depth.abs()
        depth_valid = torch.isfinite(sampled_depth) & (sampled_depth > 1.0e-5)
        not_behind_surface = signed_depth <= self.occlusion_margin * depth_scale
        occlusion_mask = (
            projection.valid
            & depth_valid
            & not_behind_surface
            & (sampled_mask > 0.5)
        )
        depth_agreement = torch.exp(
            -depth_residual / (self.depth_temperature * depth_scale).clamp_min(1.0e-6)
        )

        if aligned_point_map is not None:
            point_maps = _as_bvchw(aligned_point_map, channels=3, name="aligned_point_map").float()
            normal_maps = _normal_map(point_maps)
            sampled_normals = F.normalize(_sample_view_maps(normal_maps, grid), dim=-1, eps=1.0e-6)
            camera_centers = torch.linalg.inv(world_to_camera.float())[..., :3, 3]
            view_direction = F.normalize(
                camera_centers[:, None] - active_xyz[:, :, None], dim=-1, eps=1.0e-6
            )
        else:
            camera_points = _camera_point_map(depths, intrinsics)
            normal_maps = _normal_map(camera_points)
            sampled_normals = F.normalize(_sample_view_maps(normal_maps, grid), dim=-1, eps=1.0e-6)
            view_direction = F.normalize(-projection.camera_points, dim=-1, eps=1.0e-6)
        incidence = (sampled_normals * view_direction).sum(dim=-1, keepdim=True).abs()
        angle_agreement = incidence.clamp(min=self.minimum_incidence, max=1.0)
        agreement = depth_agreement * angle_agreement

        if view_valid_mask is None:
            view_valid_mask = torch.ones(
                batch_size, views, device=images.device, dtype=torch.bool
            )
        if view_valid_mask.shape != (batch_size, views):
            raise ValueError(f"view_valid_mask must be [B,V], got {tuple(view_valid_mask.shape)}")
        if active_valid_mask is None:
            active_valid_mask = torch.ones(
                *active_indices.shape[:2], 1, device=images.device, dtype=torch.bool
            )
        if active_valid_mask.ndim == 2:
            active_valid_mask = active_valid_mask[..., None]
        if active_valid_mask.shape != (*active_indices.shape[:2], 1):
            raise ValueError(
                f"active_valid_mask must be [B,L,1], got {tuple(active_valid_mask.shape)}"
            )

        c_v = sampled_confidence
        m_v = occlusion_mask.to(sampled_confidence.dtype)
        w_v = agreement
        weights = (
            c_v
            * m_v
            * w_v
            * view_valid_mask[:, None, :, None].to(c_v.dtype)
            * active_valid_mask[:, :, None].to(c_v.dtype)
        )
        concatenated = torch.cat([sampled_high, sampled_low], dim=-1)
        denominator = weights.sum(dim=2)
        fused = (weights * concatenated).sum(dim=2) / denominator.clamp_min(1.0e-8)
        condition = self.feature_projection(fused) + self.position(active_indices).to(fused)
        condition_valid = active_valid_mask[..., 0].bool() & (denominator[..., 0] > 0)
        condition = condition * condition_valid[..., None].to(condition.dtype)
        available_views = view_valid_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        fused_confidence = (denominator[..., 0] / available_views).clamp(0.0, 1.0)
        if not torch.isfinite(condition).all():
            raise FloatingPointError("SLat conditioning produced NaN or Inf")
        return SLatConditioningOutput(
            condition=condition,
            condition_valid=condition_valid,
            active_xyz=active_xyz,
            projection_grid=grid,
            sampled_high=sampled_high,
            sampled_low=sampled_low,
            view_weights=weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0e-8),
            confidence=fused_confidence[..., None],
            occlusion_mask=occlusion_mask,
            agreement=agreement,
        )


class AffostructionSLatFlow(nn.Module):
    """Independent image-conditioned sparse flow correcting a frozen TRELLIS prior."""

    architecture_version = AFFOSTRUCTION_IMAGE_SLAT_ARCHITECTURE

    def __init__(
        self,
        spatial_flow: nn.Module,
        conditioner: SymmetricSLatConditioner,
        *,
        classifier_free_dropout: float = 0.1,
        residual_limit: float = 1.0,
    ) -> None:
        super().__init__()
        expected = int(getattr(spatial_flow, "cond_channels", -1))
        if expected != conditioner.condition_dim:
            raise ValueError(
                f"Sparse flow cond_channels={expected}, conditioner={conditioner.condition_dim}"
            )
        if not 0.0 <= classifier_free_dropout < 1.0:
            raise ValueError("classifier_free_dropout must lie in [0,1)")
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        self.spatial_flow = spatial_flow.train().requires_grad_(True)
        self.conditioner = conditioner
        self.classifier_free_dropout = float(classifier_free_dropout)
        self.residual_limit = float(residual_limit)

    def forward(
        self,
        sparse_state,
        timestep: torch.Tensor,
        *,
        base_velocity,
        **condition_inputs,
    ) -> AffostructionSLatFlowOutput:
        conditioning = self.conditioner(**condition_inputs)
        compact = compact_condition_tokens(
            conditioning.condition, conditioning.condition_valid
        )
        batch_size = conditioning.condition.shape[0]
        empty = torch.nonzero(compact.counts == 0, as_tuple=False).flatten()
        if empty.numel() != 0:
            raise RuntimeError(
                "No active SLat voxel has valid multi-view VGGT evidence for "
                f"batch objects {empty.tolist()}"
            )
        condition = compact.tokens
        condition_mask = compact.valid_mask
        if self.training and self.classifier_free_dropout > 0:
            dropped = torch.rand(batch_size, device=condition.device) < self.classifier_free_dropout
            condition = torch.where(dropped[:, None, None], torch.zeros_like(condition), condition)
        raw_sparse = self.spatial_flow(
            sparse_state,
            timestep,
            condition,
            cond_mask=condition_mask,
        )
        raw_features = _align_sparse_features(
            raw_sparse.feats,
            raw_sparse.coords,
            base_velocity.coords,
            self.conditioner.resolution,
        )
        confidence = _confidence_for_sparse_coords(
            conditioning.confidence,
            condition_inputs["active_indices"],
            condition_inputs.get("active_valid_mask"),
            base_velocity.coords,
            self.conditioner.resolution,
        ).to(raw_features)
        batch_index = base_velocity.coords[:, 0].long()
        base_features = base_velocity.feats.float()
        energy = torch.zeros(
            batch_size,
            device=base_features.device,
            dtype=base_features.dtype,
        )
        counts = torch.zeros_like(energy)
        energy.scatter_add_(0, batch_index, base_features.square().mean(dim=-1))
        counts.scatter_add_(0, batch_index, torch.ones_like(batch_index, dtype=energy.dtype))
        base_rms = torch.sqrt(energy / counts.clamp_min(1.0) + 1.0e-4)
        limit = self.residual_limit * base_rms.index_select(0, batch_index).unsqueeze(-1)
        bounded = limit * torch.tanh(raw_features.float() / limit)
        correction_features = confidence * bounded
        velocity_features = base_features + correction_features
        velocity = base_velocity.replace(velocity_features.to(base_velocity.feats.dtype))
        raw_correction = base_velocity.replace(raw_features.to(base_velocity.feats.dtype))
        correction = base_velocity.replace(correction_features.to(base_velocity.feats.dtype))
        return AffostructionSLatFlowOutput(
            velocity=velocity,
            base_velocity=base_velocity,
            raw_correction=raw_correction,
            correction=correction,
            conditioning=conditioning,
        )


def _coordinate_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    coords = coords.long()
    return (
        ((coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2])
        * resolution
        + coords[:, 3]
    )


def _align_sparse_features(
    features: torch.Tensor,
    source_coords: torch.Tensor,
    target_coords: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    source_keys = _coordinate_keys(source_coords, resolution)
    target_keys = _coordinate_keys(target_coords, resolution)
    sorted_keys, order = source_keys.sort()
    lookup = torch.searchsorted(sorted_keys, target_keys)
    if bool((lookup >= sorted_keys.numel()).any()) or not torch.equal(
        sorted_keys[lookup], target_keys
    ):
        raise RuntimeError("Sparse image flow changed the active voxel support")
    return features[order[lookup]]


def _confidence_for_sparse_coords(
    confidence: torch.Tensor,
    active_indices: torch.Tensor,
    active_valid_mask: Optional[torch.Tensor],
    target_coords: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    batch_size, active_count = active_indices.shape[:2]
    if active_valid_mask is None:
        active_valid_mask = torch.ones(
            batch_size,
            active_count,
            1,
            device=active_indices.device,
            dtype=torch.bool,
        )
    valid = active_valid_mask[..., 0].bool()
    batch_column = torch.arange(
        batch_size,
        device=active_indices.device,
        dtype=active_indices.dtype,
    )[:, None, None].expand(-1, active_count, 1)
    coords = torch.cat([batch_column, active_indices[..., -3:]], dim=-1)[valid]
    values = confidence[valid]
    return _align_sparse_features(values, coords, target_coords, resolution)


def _sample_view_maps(maps: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    batch_size, views, channels, height, width = maps.shape
    if grid.shape[0] != batch_size or grid.shape[2] != views or grid.shape[-1] != 2:
        raise ValueError(f"grid must be [B,L,V,2], got {tuple(grid.shape)}")
    active_count = grid.shape[1]
    flat_maps = maps.reshape(batch_size * views, channels, height, width)
    flat_grid = grid.permute(0, 2, 1, 3).reshape(
        batch_size * views, active_count, 1, 2
    ).to(maps.dtype)
    sampled = F.grid_sample(
        flat_maps,
        flat_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled[..., 0].transpose(1, 2).reshape(
        batch_size, views, active_count, channels
    ).permute(0, 2, 1, 3).contiguous()


def _as_bvchw(value: torch.Tensor, *, channels: int, name: str) -> torch.Tensor:
    if value.ndim == 4 and channels == 1:
        value = value[:, :, None]
    if value.ndim != 5 or value.shape[2] != channels:
        raise ValueError(f"{name} must be [B,V,{channels},H,W], got {tuple(value.shape)}")
    return value


def _camera_point_map(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    batch_size, views, _, height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    z = depth[:, :, 0]
    x_camera = (x - intrinsics[..., 0, 2, None, None]) * z / intrinsics[..., 0, 0, None, None].clamp_min(1.0e-6)
    y_camera = (y - intrinsics[..., 1, 2, None, None]) * z / intrinsics[..., 1, 1, None, None].clamp_min(1.0e-6)
    return torch.stack([x_camera, y_camera, z], dim=2).reshape(
        batch_size, views, 3, height, width
    )


def _normal_map(points: torch.Tensor) -> torch.Tensor:
    dx = F.pad(points[..., 2:] - points[..., :-2], (1, 1, 0, 0))
    dy = F.pad(points[..., 2:, :] - points[..., :-2, :], (0, 0, 1, 1))
    normals = torch.linalg.cross(dx, dy, dim=2)
    finite = torch.isfinite(normals).all(dim=2, keepdim=True)
    return torch.where(finite, F.normalize(normals, dim=2, eps=1.0e-6), torch.zeros_like(normals))


__all__ = [
    "AFFOSTRUCTION_IMAGE_SLAT_ARCHITECTURE",
    "AFFOSTRUCTION_IMAGE_SLAT_CONFIG",
    "AffostructionSLatFlow",
    "AffostructionSLatFlowOutput",
    "SLatConditioningOutput",
    "ShallowRGBEncoder",
    "SymmetricSLatConditioner",
    "build_affostruction_image_slat_denoiser",
]
