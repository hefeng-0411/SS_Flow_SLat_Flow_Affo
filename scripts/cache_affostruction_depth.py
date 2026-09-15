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
    parser.add_argument("--raster-batch-size", type=int, default=16)
    parser.add_argument("--minimum-foreground-fraction", type=float, default=0.002)
    parser.add_argument("--maximum-foreground-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-valid-frames", type=int, default=1)
    parser.add_argument("--minimum-canonical-pixels", type=int, default=64)
    parser.add_argument(
        "--uid",
        action="append",
        default=[],
        help="Restrict caching to one UID; repeat the option for multiple UIDs.",
    )
    parser.add_argument(
        "--fail-on-insufficient-valid-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.raster_batch_size < 1:
        raise ValueError("raster-batch-size must be positive")
    if not 0.0 <= args.minimum_foreground_fraction < args.maximum_foreground_fraction <= 1.0:
        raise ValueError("foreground fractions must satisfy 0 <= minimum < maximum <= 1")
    if args.minimum_valid_frames < 1:
        raise ValueError("minimum-valid-frames must be positive")
    if args.minimum_canonical_pixels < 1:
        raise ValueError("minimum-canonical-pixels must be positive")

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
    if args.uid:
        requested = set(args.uid)
        available = set(uids)
        missing = sorted(requested - available)
        if missing:
            raise FileNotFoundError(
                f"Requested UIDs are absent from {render_root}: {missing}"
            )
        uids = [uid for uid in uids if uid in requested]
    if args.max_objects > 0:
        uids = uids[: args.max_objects]
    assigned = uids[rank::world_size]
    context = dr.RasterizeCudaContext(device=device)
    completed = skipped = unusable = 0

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
        if not args.overwrite and _quality_cache_complete(destination, records, args):
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
        destination.mkdir(parents=True, exist_ok=True)
        frame_quality = []
        valid_frame_ids = []
        for start in range(0, len(records), args.raster_batch_size):
            stop = min(start + args.raster_batch_size, len(records))
            depth, raster_mask = _rasterize_depth(
                context,
                vertices,
                faces,
                c2w[start:stop],
                K[start:stop],
                args.image_size,
            )
            alpha_masks = []
            alpha_errors = []
            for record in records[start:stop]:
                try:
                    with Image.open(record.image_path) as handle:
                        alpha = handle.convert("RGBA").getchannel("A").resize(
                            (args.image_size, args.image_size), Image.Resampling.NEAREST
                        )
                    alpha_masks.append(
                        torch.from_numpy(
                            np.asarray(alpha, dtype=np.uint8).copy() > 0
                        )
                    )
                    alpha_errors.append(None)
                except (OSError, ValueError) as exc:
                    alpha_masks.append(
                        torch.zeros(args.image_size, args.image_size, dtype=torch.bool)
                    )
                    alpha_errors.append(f"image_decode_error:{type(exc).__name__}")
            alpha_mask = torch.stack(alpha_masks).to(device)
            intersection = (alpha_mask & raster_mask).flatten(1).sum(dim=1).float()
            union = (alpha_mask | raster_mask).flatten(1).sum(dim=1).float().clamp_min(1.0)
            mask_iou = intersection / union
            alpha_fraction = alpha_mask.float().mean(dim=(1, 2))
            raster_fraction = raster_mask.float().mean(dim=(1, 2))
            finite_depth = torch.isfinite(depth).flatten(1).all(dim=1)
            positive_depth = (depth > 0).flatten(1).any(dim=1)
            canonical_pixels = _canonical_support_pixels(
                depth, K[start:stop], c2w[start:stop]
            )

            for local_index, record in enumerate(records[start:stop]):
                quality = {
                    "frame_id": record.frame_id,
                    "metadata_index": record.metadata_index,
                    "mask_iou": float(mask_iou[local_index].cpu()),
                    "alpha_foreground_fraction": float(alpha_fraction[local_index].cpu()),
                    "raster_foreground_fraction": float(raster_fraction[local_index].cpu()),
                    "canonical_support_pixels": int(canonical_pixels[local_index].cpu()),
                }
                reasons = []
                if alpha_errors[local_index] is not None:
                    reasons.append(alpha_errors[local_index])
                if quality["mask_iou"] < args.minimum_mask_iou:
                    reasons.append("mask_iou_below_threshold")
                if not (
                    args.minimum_foreground_fraction
                    <= quality["alpha_foreground_fraction"]
                    <= args.maximum_foreground_fraction
                ):
                    reasons.append("degenerate_rgb_foreground_fraction")
                if not (
                    args.minimum_foreground_fraction
                    <= quality["raster_foreground_fraction"]
                    <= args.maximum_foreground_fraction
                ):
                    reasons.append("degenerate_raster_foreground_fraction")
                if not bool(finite_depth[local_index].cpu()):
                    reasons.append("nonfinite_depth")
                if not bool(positive_depth[local_index].cpu()):
                    reasons.append("empty_depth")
                if quality["canonical_support_pixels"] < args.minimum_canonical_pixels:
                    reasons.append("insufficient_canonical_3d_support")
                quality["valid"] = not reasons
                quality["reasons"] = reasons
                frame_quality.append(quality)
                if quality["valid"]:
                    _write_depth_frame(
                        expected[start + local_index], depth[local_index].cpu().numpy()
                    )
                    valid_frame_ids.append(record.frame_id)

            del depth, raster_mask, alpha_mask

        valid_count = len(valid_frame_ids)
        status = (
            "complete"
            if valid_count >= args.minimum_valid_frames
            else "insufficient_valid_frames"
        )
        valid_ious = [row["mask_iou"] for row in frame_quality if row["valid"]]
        all_ious = [row["mask_iou"] for row in frame_quality]
        mesh_min = vertices.min(dim=0).values.cpu().tolist()
        mesh_max = vertices.max(dim=0).values.cpu().tolist()
        manifest = {
            "quality_protocol": "per_frame_rgb_depth_camera_canonical_intersection_v3",
            "status": status,
            "uid": uid,
            "render_set": args.render_set,
            "image_size": args.image_size,
            "source_frame_ids": [record.frame_id for record in records],
            "declared_frames": len(records),
            "valid_frames": valid_count,
            "rejected_frames": len(records) - valid_count,
            "valid_frame_ids": valid_frame_ids,
            "valid_metadata_indices": [
                row["metadata_index"] for row in frame_quality if row["valid"]
            ],
            "missing_declared_frames": len(transforms.get("frames", [])) - len(records),
            "mean_mask_iou_all_frames": float(np.mean(all_ious)),
            "mean_mask_iou_valid_frames": (
                float(np.mean(valid_ious)) if valid_ious else None
            ),
            "minimum_mask_iou": args.minimum_mask_iou,
            "minimum_foreground_fraction": args.minimum_foreground_fraction,
            "maximum_foreground_fraction": args.maximum_foreground_fraction,
            "minimum_valid_frames": args.minimum_valid_frames,
            "minimum_canonical_pixels": args.minimum_canonical_pixels,
            "raster_batch_size": args.raster_batch_size,
            "depth_units": "canonical_camera_metric",
            "camera_convention": "OpenCV",
            "mesh_vertices": int(vertices.shape[0]),
            "mesh_faces": int(faces.shape[0]),
            "mesh_bounds_after_gltf_basis_conversion": [mesh_min, mesh_max],
            "frame_quality": frame_quality,
        }
        _atomic_json(destination / "manifest.json", manifest)
        if status == "complete":
            completed += 1
        else:
            unusable += 1
        print(
            json.dumps(
                {
                    "event": (
                        "depth_cached"
                        if status == "complete"
                        else "depth_cache_object_unusable"
                    ),
                    "rank": rank,
                    "uid": uid,
                    "progress": f"{position}/{len(assigned)}",
                    "declared_frames": len(records),
                    "valid_frames": valid_count,
                    "rejected_frames": len(records) - valid_count,
                    "mask_iou_valid": (
                        float(np.mean(valid_ious)) if valid_ious else None
                    ),
                    "mask_iou_all": float(np.mean(all_ious)),
                    "status": status,
                }
            ),
            flush=True,
        )
        if status != "complete" and args.fail_on_insufficient_valid_frames:
            raise RuntimeError(
                f"No scientifically usable RGB/depth/camera subset for uid={uid}: "
                f"valid_frames={valid_count}, required={args.minimum_valid_frames}"
            )
        del vertices, faces, c2w, K
        torch.cuda.empty_cache()

    print(
        json.dumps(
            {
                "event": "depth_cache_complete",
                "rank": rank,
                "assigned": len(assigned),
                "completed": completed,
                "skipped": skipped,
                "unusable": unusable,
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


def _canonical_support_pixels(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor):
    height, width = depth.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=torch.float32),
        torch.arange(width, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth.float()
    x_camera = (x[None] - K[:, 0, 2, None, None]) * z / K[:, 0, 0, None, None]
    y_camera = (y[None] - K[:, 1, 2, None, None]) * z / K[:, 1, 1, None, None]
    camera = torch.stack((x_camera, y_camera, z, torch.ones_like(z)), dim=1)
    world = torch.einsum("vij,vjhw->vihw", c2w.float(), camera)[:, :3]
    canonical = (
        torch.isfinite(world).all(dim=1)
        & (world >= -0.5).all(dim=1)
        & (world <= 0.5).all(dim=1)
        & (z > 0)
    )
    return canonical.flatten(1).sum(dim=1)


def _quality_cache_complete(destination: Path, records, args) -> bool:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if (
        manifest.get("quality_protocol")
        != "per_frame_rgb_depth_camera_canonical_intersection_v3"
    ):
        return False
    source_ids = [record.frame_id for record in records]
    if manifest.get("source_frame_ids") != source_ids:
        return False
    expected_configuration = {
        "image_size": int(args.image_size),
        "minimum_mask_iou": float(args.minimum_mask_iou),
        "minimum_foreground_fraction": float(args.minimum_foreground_fraction),
        "maximum_foreground_fraction": float(args.maximum_foreground_fraction),
        "minimum_valid_frames": int(args.minimum_valid_frames),
        "minimum_canonical_pixels": int(args.minimum_canonical_pixels),
    }
    for key, expected in expected_configuration.items():
        if manifest.get(key) != expected:
            return False
    valid_ids = manifest.get("valid_frame_ids")
    if not isinstance(valid_ids, list):
        return False
    source_set = set(source_ids)
    if len(valid_ids) != len(set(valid_ids)) or not set(valid_ids).issubset(source_set):
        return False
    if int(manifest.get("valid_frames", -1)) != len(valid_ids):
        return False
    return all(
        (destination / f"{frame_id}.npz").is_file()
        and (destination / f"{frame_id}.npz").stat().st_size > 0
        for frame_id in valid_ids
    )


def _write_depth_frame(path: Path, depth_frame: np.ndarray) -> None:
    foreground = depth_frame > 0
    if not foreground.any():
        raise ValueError(f"Cannot encode an empty depth frame: {path}")
    depth_min = float(depth_frame[foreground].min())
    depth_max = float(depth_frame[foreground].max())
    span = max(depth_max - depth_min, np.finfo(np.float32).eps)
    encoded = np.zeros(depth_frame.shape, dtype=np.uint16)
    encoded[foreground] = np.clip(
        np.rint((depth_frame[foreground] - depth_min) / span * 65534.0) + 1.0,
        1,
        65535,
    ).astype(np.uint16)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            depth_u16=encoded,
            depth_min=np.float32(depth_min),
            depth_max=np.float32(depth_max),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
