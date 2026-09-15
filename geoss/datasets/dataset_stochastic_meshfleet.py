"""Stochastic multi-view MeshFleet input for production SS-flow training.

The loader performs only CPU artifact I/O.  VGGT is deliberately executed by
the trainer after the batch reaches its rank-local GPU.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from geoss.datasets.meshfleet_trellis_dataset import (
    MeshFleetTrellisDataset,
    _available_render_frame_records,
    _missing_declared_frame_ids,
    _rgba_to_rgb_mask,
)
from geoss.utils.coordinates import c2w_to_w2c, parse_objaverse_camera


class StochasticMeshFleetDataset(MeshFleetTrellisDataset):
    """MeshFleet dataset with deterministic stochastic ``N in [1, 8]`` views.

    A sample's random stream is a stable function of seed, epoch, rank, index
    and UID. Worker identity is deliberately excluded: scheduling/prefetch and
    worker-count changes must not alter the scientific sample definition.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        *,
        min_views: int = 1,
        max_views: int = 8,
        seed: int = 0,
        rank: int = 0,
        use_stochastic_views: bool = True,
        image_size: int = 518,
        load_gt_occupancy: bool = True,
        occupancy_resolution: int = 64,
        meshfleet_root: Optional[str] = None,
        depth_root: Optional[str] = None,
        require_depth: bool = False,
        **kwargs: Any,
    ) -> None:
        if min_views < 1 or max_views < min_views or max_views > 8:
            raise ValueError(f"Expected 1 <= min_views <= max_views <= 8, got {min_views}, {max_views}")
        self.min_views = int(min_views)
        self.max_views = int(max_views)
        self.seed = int(seed)
        self.rank = int(rank)
        self._shared_epoch = mp.Value("q", 0, lock=False)
        self.use_stochastic_views = bool(use_stochastic_views)
        self.load_gt_occupancy = bool(load_gt_occupancy)
        self.occupancy_resolution = int(occupancy_resolution)
        self.depth_root = Path(depth_root) if depth_root is not None else None
        self.require_depth = bool(require_depth)
        if self.require_depth and self.depth_root is None:
            raise ValueError("require_depth=True requires depth_root")
        super().__init__(
            root=root,
            split=split,
            num_views=max_views,
            image_size=image_size,
            occ_resolution=occupancy_resolution,
            repeat_views_if_insufficient=False,
            load_3d_modalities=True,
            **kwargs,
        )
        if meshfleet_root is not None:
            self._apply_meshfleet_manifest(Path(meshfleet_root))
        if not self.samples:
            raise RuntimeError(f"MeshFleet split {split!r} contains no usable objects under {root}")

    def _apply_meshfleet_manifest(self, meshfleet_root: Path) -> None:
        """Reuse MeshFleet's artifact-driven UID traversal and stable byte ordering."""
        if not meshfleet_root.is_dir():
            raise FileNotFoundError(f"MeshFleet source root not found: {meshfleet_root}")
        if str(meshfleet_root) not in sys.path:
            sys.path.insert(0, str(meshfleet_root))
        try:
            from second_jet_contact_measure.data.artifact_schema import discover_artifact_schema
            from second_jet_contact_measure.data.uid_manifest import build_uid_manifest
        except ImportError as exc:
            raise ImportError(f"Could not import MeshFleet traversal from {meshfleet_root}") from exc
        split_path = self.root / self.split if (self.root / self.split).is_dir() else self.root
        manifest = build_uid_manifest(discover_artifact_schema(split_path))
        by_uid = {sample["uid"]: sample for sample in self.samples}
        self.samples = [by_uid[entry.uid] for entry in manifest.entries if entry.uid in by_uid]

    def set_epoch(self, epoch: int) -> None:
        self._shared_epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._shared_epoch.value)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        try:
            return self._load_stochastic_sample(sample, index)
        except Exception as exc:
            raise type(exc)(
                f"Failed to load stochastic MeshFleet uid={sample['uid']!r} at index={index}: {exc}"
            ) from exc

    def _load_stochastic_sample(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        render_dir: Path = sample["render_dir"]
        transform_path = render_dir / "transforms.json"
        transforms = json.loads(transform_path.read_text(encoding="utf-8"))
        frames = transforms.get("frames", [])
        available = _available_render_frame_records(render_dir, frames)
        if not available:
            raise RuntimeError(f"Empty valid view set in {transform_path}")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._sample_seed(index, sample["uid"]))
        requested = (
            int(torch.randint(self.min_views, self.max_views + 1, (1,), generator=generator).item())
            if self.use_stochastic_views
            else self.max_views
        )
        count = min(requested, len(available))
        selected_positions = torch.randperm(len(available), generator=generator)[:count].tolist()
        chosen = [available[position] for position in selected_positions]

        images, masks, depths, intrinsics, c2w = [], [], [], [], []
        for record in chosen:  # one bounded loop over at most eight external image files
            frame, image_path = record.frame, record.image_path
            with Image.open(image_path) as handle:
                rgb, mask = _rgba_to_rgb_mask(
                    handle.convert("RGBA"), self.image_size, self.background_color
                )
            camera_record = {**{k: v for k, v in transforms.items() if k != "frames"}, **frame}
            camera_to_world, intrinsic = parse_objaverse_camera(
                camera_record,
                image_size=(self.image_size, self.image_size),
                assume_opengl=True,
            )
            _assert_camera(camera_to_world, intrinsic, sample["uid"])
            images.append(rgb)
            masks.append(mask)
            if self.depth_root is not None:
                depth_path = self.depth_root / self.split / sample["uid"] / f"{record.frame_id}.npz"
                if not depth_path.is_file():
                    if self.require_depth:
                        raise FileNotFoundError(
                            f"Missing Affostruction metric depth for exact frame {record.frame_id}: {depth_path}"
                        )
                else:
                    with np.load(depth_path, allow_pickle=False) as depth_archive:
                        encoded = np.asarray(depth_archive["depth_u16"], dtype=np.uint16)
                        depth_min = float(depth_archive["depth_min"])
                        depth_max = float(depth_archive["depth_max"])
                    depth = np.zeros(encoded.shape, dtype=np.float32)
                    foreground = encoded > 0
                    if foreground.any():
                        depth[foreground] = depth_min + (
                            (encoded[foreground].astype(np.float32) - 1.0) / 65534.0
                        ) * (depth_max - depth_min)
                    if depth.shape != (self.image_size, self.image_size):
                        raise ValueError(
                            f"Depth/image resolution mismatch for {depth_path}: {depth.shape} != "
                            f"{(self.image_size, self.image_size)}"
                        )
                    if not np.isfinite(depth).all() or np.any(depth < 0):
                        raise ValueError(f"Invalid metric depth values in {depth_path}")
                    depths.append(torch.from_numpy(depth.astype(np.float32, copy=False)))
            intrinsics.append(intrinsic)
            c2w.append(camera_to_world)
        if not all(record.numeric_id is not None for record in chosen):
            invalid = [record.frame_id for record in chosen if record.numeric_id is None]
            raise ValueError(f"MeshFleet view IDs must be decimal, got {invalid}")

        c2w_tensor = torch.stack(c2w)
        K_tensor = torch.stack(intrinsics)
        aabb = _canonical_aabb(transforms)
        center = 0.5 * (aabb[0] + aabb[1])
        half_extent = 0.5 * (aabb[1] - aabb[0])
        if not torch.isfinite(half_extent).all() or bool((half_extent <= 0).any()):
            raise ValueError(f"Malformed canonical AABB for uid={sample['uid']}: {aabb.tolist()}")

        result: Dict[str, Any] = {
            "uid": sample["uid"],
            "images": torch.stack(images),
            "masks": torch.stack(masks),
            "K_dataset": K_tensor,
            "c2w_dataset": c2w_tensor,
            "w2c_dataset": c2w_to_w2c(c2w_tensor),
            # Backward-compatible aliases for the retained sparse-ray ablation.
            "K": K_tensor,
            "c2w": c2w_tensor,
            "w2c": c2w_to_w2c(c2w_tensor),
            "view_ids": torch.tensor([record.numeric_id for record in chosen], dtype=torch.long),
            "view_metadata_indices": torch.tensor(
                [record.metadata_index for record in chosen], dtype=torch.long
            ),
            "num_views": count,
            "canonical_aabb": aabb,
            "canonical_center": center,
            "canonical_half_extent": half_extent,
            "object_scale": torch.as_tensor(transforms.get("scale", 1.0), dtype=torch.float32),
            "object_offset": torch.as_tensor(transforms.get("offset", [0.0, 0.0, 0.0]), dtype=torch.float32),
            "metadata": {
                "coordinate_system": "OpenCV cameras in normalized TRELLIS object frame",
                "source_camera_system": "Blender/OpenGL c2w",
                "canonical_aabb": aabb.tolist(),
                "scale": transforms.get("scale"),
                "offset": transforms.get("offset"),
                "selected_frame_paths": [str(record.image_path) for record in chosen],
                "selected_frame_ids": [record.frame_id for record in chosen],
                "selected_frame_metadata_indices": [record.metadata_index for record in chosen],
                "missing_frame_ids": _missing_declared_frame_ids(frames, available),
                "num_frames_total": len(frames),
                "num_frames_available": len(available),
                "missing_frames_skipped": len(frames) - len(available),
                "epoch": self.epoch,
                "rank": self.rank,
            },
        }
        if self.depth_root is not None and len(depths) == len(chosen):
            result["depths"] = torch.stack(depths)
            result["metadata"]["depth_source"] = "mesh_rasterized_metric_depth"
            result["metadata"]["depth_frame_ids"] = [record.frame_id for record in chosen]
        elif self.require_depth:
            raise RuntimeError(
                f"Depth population is incomplete for uid={sample['uid']}: {len(depths)}/{len(chosen)}"
            )
        self._attach_structure_targets(result, sample)
        return result

    def _attach_structure_targets(self, result: Dict[str, Any], sample: Dict[str, Any]) -> None:
        latent_path = sample.get("ss_latent_path")
        if latent_path is None:
            raise FileNotFoundError(f"Production SS training requires an SS latent for uid={sample['uid']}")
        with np.load(latent_path, allow_pickle=False) as archive:
            if "mean" not in archive:
                raise KeyError(f"SS latent archive lacks 'mean': {latent_path}")
            grid = torch.from_numpy(np.asarray(archive["mean"], dtype=np.float32).copy())
        if tuple(grid.shape) != (8, 16, 16, 16):
            raise ValueError(f"Expected TRELLIS SS latent [8,16,16,16], got {tuple(grid.shape)}")
        result["ss_latent_grid"] = grid
        result["ss_latent_tokens"] = grid.flatten(1).transpose(0, 1).contiguous()

        voxel_path = sample.get("voxel_path")
        if self.load_gt_occupancy and voxel_path is not None:
            points = _read_ply_xyz_vectorized(Path(voxel_path))
            resolution = self.occupancy_resolution
            indices = torch.floor((points + 0.5) * resolution).to(torch.long)
            valid = ((indices >= 0) & (indices < resolution)).all(dim=-1)
            indices = indices[valid]
            occupancy = torch.zeros(resolution, resolution, resolution, dtype=torch.float32)
            if indices.numel() == 0:
                raise RuntimeError(f"Voxel artifact contains no canonical support: {voxel_path}")
            # Tensor indexing is vectorized; PLY point iteration never enters Python.
            occupancy[indices[:, 2], indices[:, 1], indices[:, 0]] = 1.0
            result["gt_occ"] = occupancy.unsqueeze(0)
            result["gt_sparse_indices"] = indices

    def _sample_seed(self, index: int, uid: str) -> int:
        payload = f"{self.seed}|{self.epoch}|{self.rank}|{index}|{uid}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & 0x7FFF_FFFF_FFFF_FFFF


def stochastic_meshfleet_collate(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad only the view dimension; objects remain separate batch elements."""
    if not samples:
        raise ValueError("Cannot collate an empty MeshFleet batch")
    batch_size = len(samples)
    max_views = max(int(sample["num_views"]) for sample in samples)
    image_shape = samples[0]["images"].shape[1:]
    images = samples[0]["images"].new_zeros((batch_size, max_views, *image_shape))
    masks = samples[0]["masks"].new_zeros((batch_size, max_views, *samples[0]["masks"].shape[1:]))
    K = samples[0]["K_dataset"].new_zeros((batch_size, max_views, 3, 3))
    c2w = torch.eye(4, dtype=samples[0]["c2w_dataset"].dtype).view(1, 1, 4, 4).repeat(batch_size, max_views, 1, 1)
    valid = torch.zeros(batch_size, max_views, dtype=torch.bool)
    view_ids = torch.full((batch_size, max_views), -1, dtype=torch.long)
    view_metadata_indices = torch.full((batch_size, max_views), -1, dtype=torch.long)
    depths = None
    if all("depths" in sample for sample in samples):
        depths = samples[0]["depths"].new_zeros(
            (batch_size, max_views, *samples[0]["depths"].shape[-2:])
        )
    for batch_index, sample in enumerate(samples):  # bounded batch metadata loop
        count = int(sample["num_views"])
        images[batch_index, :count] = sample["images"]
        masks[batch_index, :count] = sample["masks"]
        K[batch_index, :count] = sample["K_dataset"]
        c2w[batch_index, :count] = sample["c2w_dataset"]
        valid[batch_index, :count] = True
        view_ids[batch_index, :count] = sample["view_ids"]
        view_metadata_indices[batch_index, :count] = sample["view_metadata_indices"]
        if depths is not None:
            depths[batch_index, :count] = sample["depths"]
    result: Dict[str, Any] = {
        "uid": [sample["uid"] for sample in samples],
        "images": images,
        "masks": masks,
        "K_dataset": K,
        "c2w_dataset": c2w,
        "w2c_dataset": torch.linalg.inv(c2w),
        "view_valid_mask": valid,
        "view_ids": view_ids,
        "view_metadata_indices": view_metadata_indices,
        "num_views": torch.tensor([sample["num_views"] for sample in samples], dtype=torch.long),
        "canonical_aabb": torch.stack([sample["canonical_aabb"] for sample in samples]),
        "canonical_center": torch.stack([sample["canonical_center"] for sample in samples]),
        "canonical_half_extent": torch.stack([sample["canonical_half_extent"] for sample in samples]),
        "object_scale": torch.stack([sample["object_scale"].reshape(()) for sample in samples]),
        "object_offset": torch.stack([sample["object_offset"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }
    if depths is not None:
        result["depths"] = depths
    for key in ("ss_latent_grid", "ss_latent_tokens", "gt_occ"):
        if all(key in sample for sample in samples):
            result[key] = torch.stack([sample[key] for sample in samples])
    result["gt_sparse_indices"] = [sample.get("gt_sparse_indices") for sample in samples]
    return result


def seed_meshfleet_worker(worker_id: int) -> None:
    """Seed incidental third-party loader code without driving view selection."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def _canonical_aabb(transforms: Dict[str, Any]) -> torch.Tensor:
    value = transforms.get("aabb", [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    aabb = torch.as_tensor(value, dtype=torch.float32)
    if aabb.shape != (2, 3):
        raise ValueError(f"Canonical AABB must be [2,3], got {tuple(aabb.shape)}")
    return aabb


def _assert_camera(c2w: torch.Tensor, K: torch.Tensor, uid: str) -> None:
    if c2w.shape != (4, 4) or K.shape != (3, 3) or not torch.isfinite(c2w).all() or not torch.isfinite(K).all():
        raise ValueError(f"Malformed camera matrices for uid={uid}: c2w={tuple(c2w.shape)}, K={tuple(K.shape)}")
    rotation_error = torch.linalg.matrix_norm(c2w[:3, :3].T @ c2w[:3, :3] - torch.eye(3)).item()
    if rotation_error > 1e-4 or float(torch.linalg.det(c2w[:3, :3])) <= 0 or bool((K.diag() <= 0).any()):
        raise ValueError(f"Invalid OpenCV camera for uid={uid}: rotation_error={rotation_error}")


def _read_ply_xyz_vectorized(path: Path) -> torch.Tensor:
    """Read the MeshFleet float XYZ PLY without a Python point loop."""
    with path.open("rb") as handle:
        header_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        count_line = next((line for line in header_lines if line.startswith("element vertex ")), None)
        if count_line is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        count = int(count_line.rsplit(" ", 1)[-1])
        properties = [line.split()[1:] for line in header_lines if line.startswith("property ")]
        names = [parts[-1] for parts in properties]
        if names[:3] != ["x", "y", "z"] or any(parts[0] != "float" for parts in properties[:3]):
            raise ValueError(f"Unsupported MeshFleet PLY vertex schema in {path}: {properties}")
        payload = handle.read()
    if any("binary_little_endian" in line for line in header_lines):
        if len(properties) != 3:
            dtype = np.dtype([(name, "<f4") for name in names])
            records = np.frombuffer(payload, dtype=dtype, count=count)
            array = np.stack([records[axis] for axis in ("x", "y", "z")], axis=-1)
        else:
            array = np.frombuffer(payload, dtype="<f4", count=count * 3).reshape(count, 3)
    elif any("format ascii" in line for line in header_lines):
        array = np.fromstring(payload.decode("ascii"), sep=" ", dtype=np.float32).reshape(count, len(properties))[:, :3]
    else:
        raise ValueError(f"Unsupported PLY encoding: {path}")
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


__all__ = [
    "StochasticMeshFleetDataset",
    "stochastic_meshfleet_collate",
    "seed_meshfleet_worker",
]
