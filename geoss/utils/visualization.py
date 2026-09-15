from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def write_point_cloud_ply(
    path: str | Path,
    xyz: torch.Tensor,
    values: Optional[torch.Tensor] = None,
) -> None:
    """Write an ASCII PLY point cloud with optional scalar values as grayscale color."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz_cpu = xyz.detach().reshape(-1, 3).float().cpu()
    if values is not None:
        v = values.detach().reshape(-1).float().cpu()
        v = ((v - v.min()) / (v.max() - v.min()).clamp_min(1e-6) * 255).byte()
    else:
        v = torch.full((xyz_cpu.shape[0],), 255, dtype=torch.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz_cpu.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz_cpu, v):
            ci = int(c.item())
            f.write(f"{p[0].item()} {p[1].item()} {p[2].item()} {ci} {ci} {ci}\n")


def save_npz(path: str | Path, **arrays) -> None:
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in arrays.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            serializable[key] = tensor.numpy()
        else:
            serializable[key] = value
    np.savez_compressed(path, **serializable)


def write_gt_occ_surface_ply(path: str | Path, gt_occ: torch.Tensor, threshold: float = 0.5) -> None:
    """Write occupied voxel centers as a PLY point cloud."""
    from geoss.utils.coordinates import occ_index_to_anchor_center

    occ = gt_occ.detach()
    if occ.ndim == 4:
        occ = occ[0]
    coords = torch.nonzero(occ > threshold, as_tuple=False)
    if coords.numel() == 0:
        xyz = torch.empty(0, 3)
    else:
        xyz = occ_index_to_anchor_center(coords, occ.shape[0])
    write_point_cloud_ply(path, xyz)


def write_vggt_pointmap_ply(path: str | Path, pointmap: torch.Tensor, confidence: Optional[torch.Tensor] = None, stride: int = 8) -> None:
    """Write a decimated VGGT pointmap `[B,N,3,H,W]` or `[N,3,H,W]` to PLY."""
    pts = pointmap.detach()
    if pts.ndim == 5:
        pts = pts[0]
    if pts.ndim != 4:
        raise ValueError(f"pointmap must be [B,N,3,H,W] or [N,3,H,W], got {tuple(pointmap.shape)}")
    pts = pts[:, :, ::stride, ::stride].permute(0, 2, 3, 1).reshape(-1, 3)
    vals = None
    if confidence is not None:
        conf = confidence.detach()
        if conf.ndim == 4:
            conf = conf[0]
        vals = conf[:, ::stride, ::stride].reshape(-1)
    write_point_cloud_ply(path, pts, vals)


def save_projected_anchor_debug_png(
    path: str | Path,
    image: torch.Tensor,
    uv: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    max_points: int = 2048,
) -> None:
    """Draw projected anchors on an RGB image for visual sanity checks."""
    from PIL import Image, ImageDraw
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = image.detach().cpu().clamp(0, 1)
    if img.ndim == 4:
        img = img[0]
    if img.shape[0] == 3:
        arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
    else:
        arr = (img.numpy() * 255).astype("uint8")
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    uv_cpu = uv.detach().reshape(-1, 2).cpu()
    if valid is not None:
        mask = valid.detach().reshape(-1).cpu() > 0.5
        uv_cpu = uv_cpu[mask]
    step = max(1, uv_cpu.shape[0] // max_points)
    for u, v in uv_cpu[::step]:
        draw.ellipse((float(u) - 1, float(v) - 1, float(u) + 1, float(v) + 1), fill=(255, 40, 40))
    pil.save(path)


def save_ray_free_space_debug_png(path: str | Path, free_score: torch.Tensor, occ_score: torch.Tensor) -> None:
    """Save a compact heatmap comparing free and occupied evidence per anchor."""
    from PIL import Image
    import math
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    score = (free_score.detach().reshape(-1) - occ_score.detach().reshape(-1)).cpu()
    side = int(math.ceil(math.sqrt(score.numel())))
    canvas = torch.zeros(side * side)
    canvas[: score.numel()] = score
    canvas = canvas.view(side, side)
    canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min()).clamp_min(1e-6)
    Image.fromarray((canvas.numpy() * 255).astype(np.uint8)).save(path)
