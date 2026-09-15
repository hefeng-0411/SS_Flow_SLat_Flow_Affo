"""Exact Affostruction Stage-1 RGBD conditioning and model construction.

This module deliberately contains no adapter, LoRA layer, confidence head, or
auxiliary geometry network.  It implements Eq. (2) of Affostruction: DINOv2
features are attached to depth-unprojected voxels, averaged over views, and
added to a fixed 3D sinusoidal positional encoding before native cross-attention.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
import types
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.models.affostruction_conditioning import VoxelPositionalEncoding3D


OFFICIAL_SS_ARCHITECTURE = "affostruction_official_ss_flow_768x12_v1"
OFFICIAL_SS_CONFIG = {
    "resolution": 16,
    "in_channels": 8,
    "out_channels": 8,
    "model_channels": 768,
    "cond_channels": 1024,
    "num_blocks": 12,
    "num_heads": 12,
    "mlp_ratio": 4,
    "patch_size": 1,
    "pe_mode": "ape",
    "qk_rms_norm": True,
}


@dataclass(frozen=True)
class AffostructionCondition:
    tokens: torch.Tensor
    valid_mask: torch.Tensor
    voxel_counts: torch.Tensor


def build_official_ss_denoiser(
    affostruction_root: str | Path,
    *,
    gradient_checkpointing: bool,
) -> nn.Module:
    """Construct the authors' released Stage-1 class with Table-A1 dimensions."""

    root = Path(affostruction_root).resolve()
    if not (root / "affostruction/models/sparse_structure_flow.py").is_file():
        raise FileNotFoundError(f"Official Affostruction source tree not found: {root}")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    if "affostruction" not in sys.modules:
        package = types.ModuleType("affostruction")
        package.__path__ = [str(root / "affostruction")]
        package.__package__ = "affostruction"
        sys.modules["affostruction"] = package
    module = importlib.import_module("affostruction.models.sparse_structure_flow")
    SparseStructureFlowModel = module.SparseStructureFlowModel

    model = SparseStructureFlowModel(
        **OFFICIAL_SS_CONFIG,
        use_fp16=True,
        use_checkpoint=bool(gradient_checkpointing),
    )
    model.convert_to_fp32()
    return model


class OfficialAffostructionRGBDConditioner(nn.Module):
    """Batched implementation of the released sparse RGBD voxel fusion.

    Inputs use OpenCV camera convention and canonical world coordinates in
    ``[-0.5, 0.5]^3``.  Missing/padded views are excluded with
    ``view_valid_mask``; numbered frame gaps therefore cannot shift cameras,
    depths, or images relative to one another.
    """

    def __init__(
        self,
        image_model: nn.Module,
        image_transform: Callable[[torch.Tensor], torch.Tensor],
        *,
        image_size: int = 224,
        voxel_resolution: int = 16,
        feature_dim: int = 1024,
    ) -> None:
        super().__init__()
        if image_size != 224 or voxel_resolution != 16 or feature_dim != 1024:
            raise ValueError(
                "Official Affostruction Stage-1 requires image_size=224, "
                "voxel_resolution=16, and feature_dim=1024"
            )
        self.image_model = image_model.eval().requires_grad_(False)
        self.image_transform = image_transform
        self.image_size = int(image_size)
        self.voxel_resolution = int(voxel_resolution)
        self.feature_dim = int(feature_dim)
        self.positional_encoding = VoxelPositionalEncoding3D(
            feature_dim, voxel_resolution
        )

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        masks: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_to_world: torch.Tensor,
        view_valid_mask: torch.Tensor,
        *,
        amp_dtype: torch.dtype = torch.float16,
    ) -> AffostructionCondition:
        self._validate_inputs(
            images, depths, masks, intrinsics, camera_to_world, view_valid_mask
        )
        batch_size, views, _, source_height, source_width = images.shape
        flat_views = batch_size * views
        resolution = self.voxel_resolution
        voxel_count = resolution**3

        rgb = F.interpolate(
            images.float().reshape(flat_views, 3, source_height, source_width),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)
        alpha = F.interpolate(
            masks.float().reshape(flat_views, 1, source_height, source_width),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).reshape(batch_size, views, self.image_size, self.image_size)
        depth_height, depth_width = depths.shape[-2:]
        depth = F.interpolate(
            depths.float().reshape(flat_views, 1, depth_height, depth_width),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).reshape(batch_size, views, self.image_size, self.image_size)

        normalized = self.image_transform(rgb)
        autocast_enabled = normalized.device.type == "cuda"
        with torch.autocast(
            device_type=normalized.device.type,
            dtype=amp_dtype,
            enabled=autocast_enabled,
        ):
            features_dict = self.image_model(normalized, is_training=True)
        features = self._spatial_features(features_dict, flat_views).float()

        scale_x = self.image_size / float(source_width)
        scale_y = self.image_size / float(source_height)
        K = intrinsics.float().clone()
        K[..., 0, :] *= scale_x
        K[..., 1, :] *= scale_y
        K[..., 2, :] = K[..., 2, :] * 0.0
        K[..., 2, 2] = 1.0

        y, x = torch.meshgrid(
            torch.arange(self.image_size, device=images.device, dtype=torch.float32),
            torch.arange(self.image_size, device=images.device, dtype=torch.float32),
            indexing="ij",
        )
        z = depth
        x_camera = (x[None, None] - K[..., 0, 2, None, None]) * z / K[
            ..., 0, 0, None, None
        ]
        y_camera = (y[None, None] - K[..., 1, 2, None, None]) * z / K[
            ..., 1, 1, None, None
        ]
        camera = torch.stack((x_camera, y_camera, z, torch.ones_like(z)), dim=2)
        world = torch.einsum("bvij,bvjhw->bvihw", camera_to_world.float(), camera)[
            :, :, :3
        ]
        finite = torch.isfinite(world).all(dim=2) & torch.isfinite(z)
        inside = (world >= -0.5).all(dim=2) & (world <= 0.5).all(dim=2)
        valid = (
            finite
            & inside
            & (z > 0.0)
            & (alpha > 0.0)
            & view_valid_mask.bool()[..., None, None]
        )

        indices = torch.floor((world + 0.5) * resolution).long().clamp_(0, resolution - 1)
        local_hash = (
            indices[:, :, 0] * resolution * resolution
            + indices[:, :, 1] * resolution
            + indices[:, :, 2]
        )
        view_offset = torch.arange(flat_views, device=images.device).reshape(
            batch_size, views, 1, 1
        ) * voxel_count
        grouped_hash = (local_hash + view_offset).reshape(-1)
        valid_flat = valid.reshape(-1)
        safe_hash = torch.where(valid_flat, grouped_hash, torch.zeros_like(grouped_hash))

        uv = torch.stack(
            (
                x / float(self.image_size) * 2.0 - 1.0,
                y / float(self.image_size) * 2.0 - 1.0,
            ),
            dim=-1,
        )
        uv = uv[None, None].expand(batch_size, views, -1, -1, -1).reshape(-1, 2)
        weight = valid_flat.float()
        uv_sum = torch.zeros(
            flat_views * voxel_count, 2, device=images.device, dtype=torch.float32
        )
        per_view_count = torch.zeros(
            flat_views * voxel_count, device=images.device, dtype=torch.float32
        )
        uv_sum.scatter_add_(0, safe_hash[:, None].expand(-1, 2), uv * weight[:, None])
        per_view_count.scatter_add_(0, safe_hash, weight)
        mean_uv = uv_sum / per_view_count.clamp_min(1.0)[:, None]

        sampled = F.grid_sample(
            features,
            mean_uv.reshape(flat_views, voxel_count, 1, 2),
            mode="bilinear",
            align_corners=False,
        )
        sampled = sampled[..., 0].transpose(1, 2).reshape(
            batch_size, views, voxel_count, self.feature_dim
        )
        per_view_valid = per_view_count.reshape(batch_size, views, voxel_count) > 0
        summed = (sampled * per_view_valid[..., None]).sum(dim=1)
        observation_count = per_view_valid.sum(dim=1)
        averaged = summed / observation_count.clamp_min(1)[..., None]

        coords = torch.stack(
            torch.meshgrid(
                *(torch.arange(resolution, device=images.device) for _ in range(3)),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(voxel_count, 3)
        averaged = F.layer_norm(averaged, (self.feature_dim,))
        dense_tokens = averaged + self.positional_encoding(coords)[None].float()
        observed = observation_count > 0
        dense_tokens = dense_tokens * observed[..., None]
        return _compact_conditions(dense_tokens, observed)

    def _spatial_features(self, output: dict, flat_views: int) -> torch.Tensor:
        if not isinstance(output, dict) or "x_prenorm" not in output:
            raise RuntimeError("DINOv2 must return x_prenorm tokens")
        tokens = output["x_prenorm"]
        register_count = int(getattr(self.image_model, "num_register_tokens", 0))
        patch_tokens = tokens[:, register_count + 1 :]
        patch_side = self.image_size // 14
        if patch_tokens.shape != (flat_views, patch_side * patch_side, self.feature_dim):
            raise RuntimeError(
                f"Unexpected DINOv2 patch-token shape {tuple(patch_tokens.shape)}"
            )
        return patch_tokens.transpose(1, 2).reshape(
            flat_views, self.feature_dim, patch_side, patch_side
        )

    @staticmethod
    def _validate_inputs(images, depths, masks, intrinsics, camera_to_world, valid):
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"images must be [B,V,3,H,W], got {tuple(images.shape)}")
        batch_views = images.shape[:2]
        if depths.ndim != 4 or depths.shape[:2] != batch_views:
            raise ValueError("depths must be [B,V,Hd,Wd] and align by batch/view")
        if masks.shape != (*batch_views, 1, *images.shape[-2:]):
            raise ValueError("masks must be [B,V,1,H,W] and align with images")
        if intrinsics.shape != (*batch_views, 3, 3):
            raise ValueError("intrinsics must be [B,V,3,3]")
        if camera_to_world.shape != (*batch_views, 4, 4):
            raise ValueError("camera_to_world must be [B,V,4,4]")
        if valid.shape != batch_views:
            raise ValueError("view_valid_mask must be [B,V]")


def _compact_conditions(tokens: torch.Tensor, valid: torch.Tensor) -> AffostructionCondition:
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise RuntimeError("At least one RGBD ray must occupy a canonical voxel per object")
    length = int(counts.max().item())
    order = torch.argsort(valid.to(torch.int8), dim=1, descending=True, stable=True)
    gather = order[:, :length, None].expand(-1, -1, tokens.shape[-1])
    compact = torch.gather(tokens, 1, gather)
    compact_valid = torch.arange(length, device=tokens.device)[None] < counts[:, None]
    compact = compact * compact_valid[..., None]
    return AffostructionCondition(compact, compact_valid, counts)


__all__ = [
    "AffostructionCondition",
    "OFFICIAL_SS_ARCHITECTURE",
    "OFFICIAL_SS_CONFIG",
    "OfficialAffostructionRGBDConditioner",
    "build_official_ss_denoiser",
]
