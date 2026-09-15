#!/usr/bin/env python3
"""Rasterize metric depth aligned to every existing numbered RGB frame."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geoss.datasets.meshfleet_trellis_dataset import _available_render_frame_records
from geoss.utils.coordinates import parse_objaverse_camera


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--render-set", default="renders")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.90)
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Depth caching requires CUDA nvdiffrast")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    import nvdiffrast.torch as dr
    import trimesh

    split_root = Path(args.data_root) / args.split
    render_root = split_root / args.render_set
    mesh_root = split_root / "mesh_normalized"
    output_root = Path(args.output_root) / args.split
    uids = sorted(path.name for path in render_root.iterdir() if path.is_dir())
    if args.max_objects > 0:
        uids = uids[: args.max_objects]
    assigned = uids[rank::world_size]
    context = dr.RasterizeCudaContext(device=device)
    completed = skipped = 0

    for position, uid in enumerate(assigned, 1):
        render_dir = render_root / uid
        mesh_path = mesh_root / uid / "mesh.glb"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Normalized mesh missing for uid={uid}: {mesh_path}")
        transforms = json.loads((render_dir / "transforms.json").read_text(encoding="utf-8"))
        records = _available_render_frame_records(render_dir, transforms.get("frames", []))
        if not records:
            raise RuntimeError(f"No joined RGB/camera frames for uid={uid}")
        destination = output_root / uid
        expected = [destination / f"{record.frame_id}.npz" for record in records]
        if not args.overwrite and all(path.is_file() and path.stat().st_size > 0 for path in expected):
            skipped += 1
            continue

        loaded = trimesh.load(mesh_path, force="scene", process=False)
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
        vertices_gltf = np.asarray(mesh.vertices, dtype=np.float32)
        # glTF is Y-up while transforms.json is serialized in Blender's Z-up
        # world frame.  Undo the exporter basis change before applying cameras.
        vertices_blender = np.stack(
            (vertices_gltf[:, 0], -vertices_gltf[:, 2], vertices_gltf[:, 1]), axis=1
        )
        vertices = torch.as_tensor(vertices_blender, device=device)
        faces = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int32), device=device)
        if vertices.numel() == 0 or faces.numel() == 0:
            raise RuntimeError(f"Empty mesh for uid={uid}: {mesh_path}")

        c2w_values = []
        intrinsics_values = []
        for record in records:
            camera_record = {
                **{key: value for key, value in transforms.items() if key != "frames"},
                **record.frame,
            }
            c2w, K = parse_objaverse_camera(
                camera_record,
                image_size=(args.image_size, args.image_size),
                assume_opengl=True,
            )
            c2w_values.append(c2w)
            intrinsics_values.append(K)
        c2w = torch.stack(c2w_values).to(device)
        K = torch.stack(intrinsics_values).to(device)
        depth, raster_mask = _rasterize_depth(
            context, vertices, faces, c2w, K, args.image_size
        )

        alpha_masks = []
        for record in records:
            with Image.open(record.image_path) as handle:
                alpha = handle.convert("RGBA").getchannel("A").resize(
                    (args.image_size, args.image_size), Image.Resampling.NEAREST
                )
            alpha_masks.append(torch.from_numpy(np.asarray(alpha, dtype=np.uint8) > 0))
        alpha_mask = torch.stack(alpha_masks).to(device)
        intersection = (alpha_mask & raster_mask).flatten(1).sum(dim=1).float()
        union = (alpha_mask | raster_mask).flatten(1).sum(dim=1).float().clamp_min(1.0)
        mask_iou = intersection / union
        if float(mask_iou.mean()) < args.minimum_mask_iou:
            raise RuntimeError(
                f"RGB/depth camera verification failed for uid={uid}: "
                f"mean_mask_iou={float(mask_iou.mean()):.6f}, "
                f"minimum={args.minimum_mask_iou}"
            )

        destination.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(expected):
            temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
            try:
                depth_frame = depth[index].cpu().numpy()
                foreground = depth_frame > 0
                depth_min = float(depth_frame[foreground].min())
                depth_max = float(depth_frame[foreground].max())
                span = max(depth_max - depth_min, np.finfo(np.float32).eps)
                encoded = np.zeros(depth_frame.shape, dtype=np.uint16)
                encoded[foreground] = np.clip(
                    np.rint((depth_frame[foreground] - depth_min) / span * 65534.0) + 1.0,
                    1,
                    65535,
                ).astype(np.uint16)
                np.savez_compressed(
                    temporary,
                    depth_u16=encoded,
                    depth_min=np.float32(depth_min),
                    depth_max=np.float32(depth_max),
                )
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        manifest = {
            "uid": uid,
            "render_set": args.render_set,
            "image_size": args.image_size,
            "frame_ids": [record.frame_id for record in records],
            "metadata_indices": [record.metadata_index for record in records],
            "missing_declared_frames": len(transforms.get("frames", [])) - len(records),
            "mean_mask_iou": float(mask_iou.mean()),
            "minimum_mask_iou": float(mask_iou.min()),
            "depth_units": "canonical_camera_metric",
            "camera_convention": "OpenCV",
        }
        _atomic_json(destination / "manifest.json", manifest)
        completed += 1
        print(
            json.dumps(
                {
                    "event": "depth_cached",
                    "rank": rank,
                    "uid": uid,
                    "progress": f"{position}/{len(assigned)}",
                    "frames": len(records),
                    "mask_iou": float(mask_iou.mean()),
                }
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "depth_cache_complete",
                "rank": rank,
                "assigned": len(assigned),
                "completed": completed,
                "skipped": skipped,
            }
        ),
        flush=True,
    )


def _rasterize_depth(context, vertices, faces, c2w, K, image_size):
    import nvdiffrast.torch as dr

    views = c2w.shape[0]
    w2c = torch.linalg.inv(c2w.float())
    normalized_K = K.float().clone()
    normalized_K[:, 0, :] /= float(image_size)
    normalized_K[:, 1, :] /= float(image_size)
    projection = torch.zeros(views, 4, 4, device=vertices.device, dtype=torch.float32)
    near, far = 0.01, 100.0
    projection[:, 0, 0] = 2.0 * normalized_K[:, 0, 0]
    projection[:, 1, 1] = 2.0 * normalized_K[:, 1, 1]
    projection[:, 0, 2] = 2.0 * normalized_K[:, 0, 2] - 1.0
    projection[:, 1, 2] = -2.0 * normalized_K[:, 1, 2] + 1.0
    projection[:, 2, 2] = far / (far - near)
    projection[:, 2, 3] = near * far / (near - far)
    projection[:, 3, 2] = 1.0
    vertices_h = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=1)
    camera_vertices = torch.einsum("vij,nj->vni", w2c, vertices_h)
    clip_vertices = torch.einsum(
        "vij,vnj->vni", projection, camera_vertices
    ).contiguous()
    raster, _ = dr.rasterize(
        context, clip_vertices, faces, resolution=(image_size, image_size)
    )
    interpolated, _ = dr.interpolate(
        camera_vertices[..., 2:3].contiguous(), raster, faces
    )
    mask = raster[..., 3] > 0
    depth = torch.where(mask, interpolated[..., 0], torch.zeros_like(interpolated[..., 0]))
    return depth.float(), mask


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
