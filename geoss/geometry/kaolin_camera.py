"""Vectorized OpenCV-camera projection through Kaolin's camera primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geoss.utils.optional_deps import require_dependency


@dataclass(frozen=True)
class KaolinProjection:
    """Pixel-aligned projection of batched world points into batched views."""

    grid: torch.Tensor
    pixel: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor
    camera_points: torch.Tensor


def project_opencv_points_kaolin(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
    image_height: int,
    image_width: int,
    *,
    near: float = 1.0e-4,
    far: float = 1.0e4,
) -> KaolinProjection:
    """Project ``[B,L,3]`` points into ``[B,V]`` OpenCV cameras with Kaolin.

    Kaolin cameras look down negative Z with positive Y pointing upward,
    whereas the dataset and VGGT camera contract is OpenCV (positive Z,
    positive Y downward).  A fixed basis change converts the view matrices;
    principal-point offsets are converted relative to the image centre.
    Kaolin then produces the NDC grid consumed by ``grid_sample`` with
    ``align_corners=False``.
    """

    require_dependency(
        "kaolin",
        real_mode=True,
        feature="deterministic active-voxel reprojection",
    )
    from kaolin.render.camera import Camera, CameraExtrinsics, PinholeIntrinsics

    if points_world.ndim != 3 or points_world.shape[-1] != 3:
        raise ValueError(f"points_world must be [B,L,3], got {tuple(points_world.shape)}")
    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics must be [B,V,3,3], got {tuple(intrinsics.shape)}")
    if world_to_camera.shape != (*intrinsics.shape[:2], 4, 4):
        raise ValueError(
            "world_to_camera must be [B,V,4,4] aligned with intrinsics, got "
            f"{tuple(world_to_camera.shape)}"
        )
    batch_size, views = intrinsics.shape[:2]
    if points_world.shape[0] != batch_size:
        raise ValueError(
            f"point batch {points_world.shape[0]} does not match camera batch {batch_size}"
        )
    if image_height < 1 or image_width < 1:
        raise ValueError("image_height and image_width must be positive")

    computation_dtype = torch.float32
    device = points_world.device
    K = intrinsics.to(device=device, dtype=computation_dtype)
    w2c_cv = world_to_camera.to(device=device, dtype=computation_dtype)
    points = points_world.to(device=device, dtype=computation_dtype)
    finite_camera = torch.isfinite(K).all(dim=(-2, -1)) & torch.isfinite(w2c_cv).all(dim=(-2, -1))
    finite_points = torch.isfinite(points).all(dim=-1)

    basis = torch.diag(points.new_tensor([1.0, -1.0, -1.0, 1.0]))
    w2c_gl = basis.view(1, 1, 4, 4) @ torch.nan_to_num(w2c_cv)
    flat_w2c = w2c_gl.reshape(batch_size * views, 4, 4)
    extrinsics = CameraExtrinsics.from_view_matrix(
        flat_w2c,
        dtype=computation_dtype,
        device=device,
    )

    x_offset = K[..., 0, 2] - float(image_width) * 0.5
    y_offset = float(image_height) * 0.5 - K[..., 1, 2]
    params = torch.stack(
        [x_offset, y_offset, K[..., 0, 0], K[..., 1, 1]],
        dim=-1,
    ).reshape(batch_size * views, 4)
    camera_intrinsics = PinholeIntrinsics(
        int(image_width),
        int(image_height),
        torch.nan_to_num(params),
        float(near),
        float(far),
    )
    cameras = Camera(extrinsics, camera_intrinsics)

    point_count = points.shape[1]
    expanded = points[:, None].expand(batch_size, views, point_count, 3)
    flat_points = expanded.reshape(batch_size * views, point_count, 3)
    ndc = cameras.transform(flat_points).reshape(batch_size, views, point_count, 3)
    grid = torch.stack([ndc[..., 0], -ndc[..., 1]], dim=-1)

    homogeneous = torch.cat([flat_points, torch.ones_like(flat_points[..., :1])], dim=-1)
    camera_cv = torch.einsum(
        "bcij,bclj->bcli",
        torch.nan_to_num(w2c_cv),
        homogeneous.reshape(batch_size, views, point_count, 4),
    )
    depth = camera_cv[..., 2:3]
    pixel = torch.stack(
        [
            (grid[..., 0] + 1.0) * (float(image_width) * 0.5),
            (grid[..., 1] + 1.0) * (float(image_height) * 0.5),
        ],
        dim=-1,
    )
    valid = (
        finite_camera[:, :, None, None]
        & finite_points[:, None, :, None]
        & torch.isfinite(grid).all(dim=-1, keepdim=True)
        & torch.isfinite(depth)
        & (depth > float(near))
        & (depth < float(far))
        & (grid[..., 0:1] >= -1.0)
        & (grid[..., 0:1] <= 1.0)
        & (grid[..., 1:2] >= -1.0)
        & (grid[..., 1:2] <= 1.0)
    )
    return KaolinProjection(
        grid=grid.permute(0, 2, 1, 3).contiguous(),
        pixel=pixel.permute(0, 2, 1, 3).contiguous(),
        depth=depth.permute(0, 2, 1, 3).contiguous(),
        valid=valid.permute(0, 2, 1, 3).contiguous(),
        camera_points=camera_cv[..., :3].permute(0, 2, 1, 3).contiguous(),
    )


__all__ = ["KaolinProjection", "project_opencv_points_kaolin"]
