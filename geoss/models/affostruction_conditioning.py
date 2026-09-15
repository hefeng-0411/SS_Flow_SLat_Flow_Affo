"""Shared conditioning primitives for Affostruction-style flow models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelPositionalEncoding3D(nn.Module):
    """Affostruction's additive three-axis sinusoidal voxel encoding."""

    def __init__(self, channels: int, resolution: int) -> None:
        super().__init__()
        if channels < 1 or resolution < 1:
            raise ValueError("channels and resolution must be positive")
        original_channels = int(channels)
        axis_channels = int(math.ceil(original_channels / 6.0) * 2)
        if axis_channels % 2:
            axis_channels += 1
        inv_freq = 1.0 / (
            10_000.0
            ** (torch.arange(0, axis_channels, 2, dtype=torch.float32) / float(axis_channels))
        )
        position = torch.arange(resolution, dtype=torch.float32)
        phase = torch.einsum("i,j->ij", position, inv_freq)
        axis = torch.stack([phase.sin(), phase.cos()], dim=-1).flatten(-2)
        x = axis[:, None, None].expand(resolution, resolution, resolution, axis_channels)
        y = axis[None, :, None].expand(resolution, resolution, resolution, axis_channels)
        z = axis[None, None, :].expand(resolution, resolution, resolution, axis_channels)
        encoding = torch.cat([x, y, z], dim=-1)[..., :original_channels].contiguous()
        self.channels = original_channels
        self.resolution = int(resolution)
        self.register_buffer("encoding", encoding, persistent=True)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] == 4:
            indices = indices[..., 1:]
        if indices.shape[-1] != 3:
            raise ValueError(f"indices must end in xyz, got {tuple(indices.shape)}")
        xyz = indices.long()
        if bool(((xyz < 0) | (xyz >= self.resolution)).any()):
            raise ValueError(f"voxel indices must lie in [0,{self.resolution - 1}]")
        return self.encoding[xyz[..., 0], xyz[..., 1], xyz[..., 2]]


@dataclass(frozen=True)
class CompactCondition:
    tokens: torch.Tensor
    valid_mask: torch.Tensor
    counts: torch.Tensor


def compact_condition_tokens(tokens: torch.Tensor, valid_mask: torch.Tensor) -> CompactCondition:
    """Stable, vectorized compaction of padded voxel conditions."""

    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [B,M,C], got {tuple(tokens.shape)}")
    if valid_mask.ndim == 3 and valid_mask.shape[-1] == 1:
        valid_mask = valid_mask[..., 0]
    if valid_mask.shape != tokens.shape[:2]:
        raise ValueError(
            f"valid_mask must be [B,M], got {tuple(valid_mask.shape)} for {tuple(tokens.shape)}"
        )
    valid = valid_mask.bool()
    counts = valid.sum(dim=1)
    compact_length = max(1, int(counts.max().item()))
    order = torch.argsort(valid.to(torch.int8), dim=1, descending=True, stable=True)
    gather_index = order[:, :compact_length, None].expand(-1, -1, tokens.shape[-1])
    compact = torch.gather(tokens, 1, gather_index)
    compact_valid = (
        torch.arange(compact_length, device=tokens.device)[None] < counts[:, None]
    )
    compact = compact * compact_valid[..., None].to(compact.dtype)
    return CompactCondition(compact, compact_valid, counts)


@torch.no_grad()
def extract_dinov2_spatial_features(
    image_model: nn.Module,
    image_transform: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    *,
    image_size: int = 518,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run the frozen TRELLIS DINO encoder and return its patch feature map."""

    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError(f"images must be [B,V,3,H,W], got {tuple(images.shape)}")
    batch_size, views = images.shape[:2]
    resized = F.interpolate(
        images.float().reshape(batch_size * views, 3, *images.shape[-2:]),
        size=(int(image_size), int(image_size)),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)
    normalized = image_transform(resized)
    amp_enabled = normalized.device.type == "cuda"
    with torch.autocast(
        device_type=normalized.device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        output = image_model(normalized, is_training=True)
    if not isinstance(output, dict) or "x_prenorm" not in output:
        raise RuntimeError("TRELLIS DINO encoder did not return x_prenorm tokens")
    all_tokens = output["x_prenorm"]
    register_tokens = int(getattr(image_model, "num_register_tokens", 0))
    patch_tokens = all_tokens[:, register_tokens + 1 :]
    patch_size = int(getattr(image_model, "patch_size", 14))
    grid_height = int(image_size) // patch_size
    grid_width = int(image_size) // patch_size
    if grid_height * grid_width != patch_tokens.shape[1]:
        side = math.isqrt(int(patch_tokens.shape[1]))
        if side * side != patch_tokens.shape[1]:
            raise RuntimeError(
                "DINO patch-token count cannot be mapped to a spatial grid: "
                f"tokens={patch_tokens.shape[1]}, image_size={image_size}, patch_size={patch_size}"
            )
        grid_height = grid_width = side
    channels = patch_tokens.shape[-1]
    features = patch_tokens.transpose(1, 2).reshape(
        batch_size, views, channels, grid_height, grid_width
    )
    finite = torch.isfinite(features).all(dim=2, keepdim=True)
    return torch.where(finite, features.float(), torch.zeros_like(features, dtype=torch.float32))


def enable_trellis_gradient_checkpointing(model: nn.Module, enabled: bool = True) -> None:
    """Enable the checkpoint switches already implemented by TRELLIS blocks."""

    state = bool(enabled)
    if hasattr(model, "use_checkpoint"):
        model.use_checkpoint = state
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = state


__all__ = [
    "CompactCondition",
    "VoxelPositionalEncoding3D",
    "compact_condition_tokens",
    "enable_trellis_gradient_checkpointing",
    "extract_dinov2_spatial_features",
]
