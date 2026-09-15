#!/usr/bin/env python3
"""Distributed, leakage-free inference for Affostruction SS/SLat checkpoints."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.geometry.alignment import align_vggt_batch
from geoss.integration.real_trellis_pipeline import RealTrellisGeoPipeline
from geoss.integration.trellis_hub import (
    configure_trellis_hub,
    resolve_local_hf_snapshot,
)
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper
from geoss.models.affostruction_conditioning import extract_dinov2_spatial_features
from geoss.models.official_affostruction_ss import (
    OFFICIAL_SS_ARCHITECTURE,
    OfficialAffostructionRGBDConditioner,
    build_official_ss_denoiser,
)
from geoss.models.ss_flow_adapter import AffostructionSSFlow
from geoss.models.voxel_fusion_engine import ConfidenceSparseVoxelFusion
from geoss.slat.models.slat_flow_adapter import (
    AffostructionSLatFlow,
    SymmetricSLatConditioner,
    build_affostruction_image_slat_denoiser,
)
from scripts.train_geovis_slat import TrainableSymmetricSLatBranch
from scripts.train_stochastic_voxel_ss_flow import TrainableVoxelSSBranch


SS_ARCHITECTURE = "affostruction_direct_ss_cross_attention_v1"
SLAT_ARCHITECTURE = AffostructionSLatFlow.architecture_version
NUMERICS_VERSION = "frozen_trellis_sparse_image_flow_fp32_master_v1"
METHODS = ("original_trellis", "ss_only", "slat_only", "ss_slat")


def _configured_amp_dtype(name: str) -> torch.dtype | None:
    return {
        "native": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Production Affostruction inference requires CUDA")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds visible devices={torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.device = str(device)
    use_ss = args.method in {"ss_only", "ss_slat"}
    use_slat = args.method in {"slat_only", "ss_slat"}
    if use_slat and args.slat_spconv_algo != "auto":
        # TRELLIS reads this while constructing its sparse convolution layers.
        # A6000/cumm does not provide MaskImplicitGemm kernels for every
        # dynamic support emitted by the fine-tuned full-precision backbone.
        os.environ["SPCONV_ALGO"] = args.slat_spconv_algo

    uids = _read_manifest(Path(args.uid_manifest))
    if args.max_samples > 0:
        uids = uids[: args.max_samples]
    shard = uids[rank::world_size]
    method_root = Path(args.output_root) / args.method
    method_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        method_root / f"worker_rank{rank}.json",
        {
            "status": "starting",
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "method": args.method,
            "assigned_uids": len(shard),
        },
    )

    configure_trellis_hub(args)
    trellis_source = resolve_local_hf_snapshot(
        args.trellis_model_path,
        args.hf_cache_root,
        required_file="pipeline.json",
        validate_trellis_pipeline=True,
    )
    pipeline = RealTrellisGeoPipeline(
        args.trellis_root, trellis_source, device=str(device)
    )
    fusion = _install_ss(args, pipeline, device) if use_ss else None
    conditioner = _install_slat(args, pipeline, device) if use_slat else None
    needs_vggt = use_slat or isinstance(fusion, ConfidenceSparseVoxelFusion)
    vggt = _load_vggt(args, device) if needs_vggt else None

    dataset = MeshFleetTrellisDataset(
        args.data_root,
        split=args.split,
        category=args.category,
        num_views=args.num_views,
        image_size=args.conditioning_image_size,
        render_set=args.conditioning_view_set,
        repeat_views_if_insufficient=False,
        uid_manifest=uids,
        load_3d_modalities=False,
    )
    torch.cuda.reset_peak_memory_stats(device)
    completed = failed = skipped = 0
    for uid in shard:
        out_dir = method_root / uid
        metrics_path = out_dir / "metrics.json"
        if (
            not args.overwrite
            and _completed_inference(
                out_dir, metrics_path, method=args.method, uid=uid
            )
        ):
            skipped += 1
            continue
        _cleanup_incomplete_output(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _require_free_disk(method_root, args.minimum_free_disk_gb)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            sample = dataset.get_by_uid(uid)
            if sample["images"].shape[0] != args.num_views:
                raise RuntimeError(
                    f"UID {uid} has {sample['images'].shape[0]} usable conditioning views; "
                    f"the frozen protocol requires {args.num_views}"
                )
            result = _infer_one(
                args,
                pipeline,
                sample,
                device,
                fusion=fusion,
                conditioner=conditioner,
                vggt=vggt,
            )
            outputs = result.pop("outputs")
            saved = pipeline.save_outputs(outputs, out_dir)
            coords = outputs["coords"].detach().cpu()
            occupancy = torch.zeros(
                args.occ_resolution, args.occ_resolution, args.occ_resolution,
                dtype=torch.uint8,
            )
            indices = coords[:, 1:].long().clamp(0, args.occ_resolution - 1)
            occupancy[indices[:, 2], indices[:, 1], indices[:, 0]] = 1
            occupancy_path = out_dir / "predicted_ss_occ.npz"
            occupancy_temporary = occupancy_path.with_name(
                f".{occupancy_path.stem}.{os.getpid()}.tmp.npz"
            )
            try:
                np.savez_compressed(occupancy_temporary, occ=occupancy.numpy())
                if occupancy_temporary.stat().st_size == 0:
                    raise OSError(f"Empty occupancy artifact: {occupancy_temporary}")
                os.replace(occupancy_temporary, occupancy_path)
            finally:
                occupancy_temporary.unlink(missing_ok=True)
            metrics = {
                "status": "ok",
                "method": args.method,
                "uid": uid,
                "rank": rank,
                "local_rank": local_rank,
                "saved_assets": saved,
                "num_active_voxels": int(coords.shape[0]),
                "conditioning_view_set": args.conditioning_view_set,
                "conditioning_frame_ids": sample["metadata"]["selected_frame_ids"],
                "num_conditioning_views": int(sample["images"].shape[0]),
                "conditioning_image_size": list(sample["images"].shape[-2:]),
                "ss_checkpoint": args.ss_checkpoint if use_ss else None,
                "ss_architecture_version": (
                    getattr(fusion, "checkpoint_architecture", None) if use_ss else None
                ),
                "slat_checkpoint": args.slat_checkpoint if use_slat else None,
                "ss_finetuned": use_ss,
                "slat_finetuned": use_slat,
                "ss_sampler_params": {
                    "steps": args.ss_steps,
                    "cfg_strength": args.ss_cfg_strength,
                },
                "slat_sampler_params": {
                    "steps": args.slat_steps,
                    "cfg_strength": args.slat_cfg_strength,
                },
                "seed": args.seed,
                "multi_image_mode": args.multi_image_mode,
                "latency_seconds": time.perf_counter() - started,
                "peak_allocated_cuda_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                "test_time_ground_truth_latents_used": False,
                "test_time_ground_truth_mesh_used": False,
                "test_time_ground_truth_voxels_used": False,
                "evaluation_views_used": False,
                "inference_context_source": (
                    "conditioning_rgb_masks_cameras_vggt_and_predicted_ss_support"
                    if use_ss or use_slat
                    else "conditioning_multiview_rgb_and_masks_only"
                ),
                "ss_forward_dtype": args.ss_amp_dtype if use_ss else "native",
                "slat_forward_dtype": args.slat_amp_dtype if use_slat else "native",
                "slat_spconv_algo": args.slat_spconv_algo if use_slat else "auto",
                **result,
            }
            _atomic_json(metrics_path, metrics)
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "inference_complete",
                        "rank": rank,
                        "method": args.method,
                        "uid": uid,
                        "progress": f"{completed + failed + skipped}/{len(shard)}",
                        "latency_seconds": metrics["latency_seconds"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            failed += 1
            failure = {
                "status": "failed",
                "method": args.method,
                "uid": uid,
                "rank": rank,
                "local_rank": local_rank,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "latency_seconds": time.perf_counter() - started,
            }
            try:
                _atomic_json(metrics_path, failure)
            except OSError as reporting_error:
                print(
                    json.dumps(
                        {
                            "event": "failure_report_write_failed",
                            "rank": rank,
                            "method": args.method,
                            "uid": uid,
                            "error": str(reporting_error),
                        }
                    ),
                    flush=True,
                )
            print(
                json.dumps(
                    {
                        "event": "inference_failed",
                        "rank": rank,
                        "method": args.method,
                        "uid": uid,
                        "progress": f"{completed + failed + skipped}/{len(shard)}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
            if _is_disk_exhaustion(exc):
                raise RuntimeError(
                    f"Stopping distributed inference because storage was exhausted while writing {uid}. "
                    "Free space and rerun with --no-overwrite; completed objects will be skipped."
                ) from exc
        finally:
            if vggt is not None:
                vggt.clear_cache()
            torch.cuda.empty_cache()

    _atomic_json(
        method_root / f"worker_rank{rank}.json",
        {
            "status": "complete",
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "method": args.method,
            "assigned_uids": len(shard),
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "peak_allocated_cuda_gb": torch.cuda.max_memory_allocated(device) / 2**30,
        },
    )
    print(
        json.dumps(
            {
                "rank": rank,
                "method": args.method,
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
            }
        ),
        flush=True,
    )


@torch.inference_mode()
def _infer_one(args, pipeline, sample, device, *, fusion, conditioner, vggt):
    images = sample["images"].to(device=device, dtype=torch.float32)[None]
    masks = sample["masks"].to(device=device, dtype=torch.float32)[None]
    K = sample["K"].to(device=device, dtype=torch.float32)[None]
    c2w = sample["c2w"].to(device=device, dtype=torch.float32)[None]
    w2c = sample["w2c"].to(device=device, dtype=torch.float32)[None]
    valid_views = torch.ones(1, images.shape[1], device=device, dtype=torch.bool)
    dino_features = None
    geometry = None
    aligned = None
    fused = None
    official_condition = None
    if vggt is not None:
        geometry = vggt.extract(images, valid_view_mask=valid_views, use_cache=True)
        raw_geometry = vggt(images, use_cache=True)
        alignment_batch = {
            "images": images,
            "masks": masks,
            "K": K,
            "c2w": c2w,
            "w2c": w2c,
            "vggt_features": geometry.visual_features,
            **{
                key: raw_geometry[key]
                for key in (
                    "vggt_depth",
                    "vggt_pointmap",
                    "vggt_confidence",
                    "vggt_camera",
                )
            },
        }
        aligned = align_vggt_batch(alignment_batch)
        if isinstance(fusion, ConfidenceSparseVoxelFusion):
            dino_features = extract_dinov2_spatial_features(
                pipeline.pipeline.models["image_cond_model"],
                pipeline.pipeline.image_cond_model_transform,
                images,
                image_size=518,
                amp_dtype=torch.bfloat16,
            )
    if isinstance(fusion, OfficialAffostructionRGBDConditioner):
        depths = _load_exact_depths(
            Path(args.depth_root),
            args.split,
            sample["uid"],
            sample["metadata"]["selected_frame_ids"],
        ).to(device=device)[None]
        official_condition = fusion(
            images,
            depths,
            masks,
            K,
            c2w,
            valid_views,
            amp_dtype=_configured_amp_dtype(args.ss_amp_dtype) or torch.float32,
        )
        ss_context = {
            "voxel_condition": official_condition.tokens,
            "observation_mask": official_condition.valid_mask,
        }
    elif fusion is not None:
        aabb_value = sample["metadata"].get("aabb")
        if aabb_value is None:
            aabb_value = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
        aabb = torch.as_tensor(aabb_value, device=device, dtype=torch.float32)
        fused = fusion(
            geometry,
            foreground_masks=masks,
            dataset_K=K,
            dataset_c2w=c2w,
            canonical_center=(0.5 * (aabb[0] + aabb[1]))[None],
            canonical_half_extent=(0.5 * (aabb[1] - aabb[0]))[None],
            spatial_features=dino_features,
        )
        ss_context = {
            "voxel_condition": fused.dense_tokens,
            "observation_mask": fused.observation_mask,
        }
    else:
        ss_context = None

    def slat_context_factory(coords: torch.Tensor) -> dict[str, torch.Tensor]:
        confidence = aligned["vggt_confidence"]
        if confidence.ndim == 4:
            confidence = confidence[:, :, None]
        confidence = confidence * aligned["alignment_confidence"].to(confidence)
        active = coords[None]
        active_valid = torch.ones(
            1, coords.shape[0], 1, device=device, dtype=torch.bool
        )
        conditioned = conditioner(
            active_indices=active,
            images=images,
            high_features=aligned["vggt_features"],
            intrinsics=K,
            world_to_camera=w2c,
            foreground_masks=masks,
            aligned_depth=aligned["aligned_depth"],
            vggt_confidence=confidence,
            active_valid_mask=active_valid,
            view_valid_mask=valid_views,
            aligned_point_map=aligned["aligned_pointmap"],
        )
        return {
            "condition": conditioned.condition,
            "condition_valid": conditioned.condition_valid,
            "confidence": conditioned.confidence,
            "active_indices": active,
        }

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    official_ss = official_condition is not None
    pipeline_images = images[0, :1] if official_ss else images[0]
    pipeline_masks = masks[:, :1] if official_ss else masks
    outputs = pipeline.run(
        pipeline_images,
        masks=pipeline_masks,
        geoss_context=ss_context,
        geovis_slat_context_factory=(
            slat_context_factory if conditioner is not None else None
        ),
        formats=("gaussian", "mesh"),
        seed=args.seed,
        ss_sampler_params={
            "steps": args.ss_steps,
            "cfg_strength": args.ss_cfg_strength,
        },
        slat_sampler_params={
            "steps": args.slat_steps,
            "cfg_strength": args.slat_cfg_strength,
        },
        multi_image_mode="stochastic" if official_ss else args.multi_image_mode,
        ss_autocast_dtype=(
            _configured_amp_dtype(args.ss_amp_dtype) if fusion is not None else None
        ),
        slat_autocast_dtype=(
            _configured_amp_dtype(args.slat_amp_dtype)
            if conditioner is not None
            else None
        ),
    )
    diagnostics = {}
    if official_condition is not None:
        diagnostics.update(
            {
                "ss_conditioning_mode": "official_rgbd_average_voxel_fusion",
                "ss_conditioning_voxels": int(official_condition.voxel_counts[0].cpu()),
                "ss_observed_percent": float(
                    official_condition.voxel_counts[0].float().cpu() / 4096.0 * 100.0
                ),
                "ss_slat_handoff": "native_trellis_single_first_image",
            }
        )
    if fused is not None:
        diagnostics.update(
            {
                "ss_alignment_mode": fused.alignment_mode,
                "ss_alignment_plausible_fraction": (
                    float(fused.alignment_plausible_fraction.mean().cpu())
                    if fused.alignment_plausible_fraction is not None
                    else None
                ),
                "ss_observed_percent": float(
                    fused.observation_mask.float().mean().cpu() * 100.0
                ),
                "ss_geometry_status": fused.geometry_status,
            }
        )
    return {"outputs": outputs, **diagnostics}


def _install_ss(args, pipeline: RealTrellisGeoPipeline, device: torch.device):
    if not Path(args.ss_checkpoint).is_file():
        raise FileNotFoundError(f"SS checkpoint not found: {args.ss_checkpoint}")
    payload = _load_checkpoint(args.ss_checkpoint)
    architecture = payload.get("architecture_version") if isinstance(payload, dict) else None
    if architecture == OFFICIAL_SS_ARCHITECTURE:
        model = build_official_ss_denoiser(
            args.affostruction_root,
            gradient_checkpointing=False,
        ).to(device=device, dtype=torch.float32)
        state = payload.get("ema_model", payload.get("model"))
        if not isinstance(state, dict):
            raise TypeError(f"Official SS checkpoint lacks model weights: {args.ss_checkpoint}")
        model.load_state_dict(state, strict=True)
        del payload
        model.eval().requires_grad_(False)
        pipeline.install_direct_ss_flow(model)
        image_model = pipeline.pipeline.models["image_cond_model"]
        official_conditioner = OfficialAffostructionRGBDConditioner(
            image_model,
            pipeline.pipeline.image_cond_model_transform,
            image_size=224,
            voxel_resolution=16,
            feature_dim=1024,
        ).to(device).eval()
        official_conditioner.checkpoint_architecture = OFFICIAL_SS_ARCHITECTURE
        return official_conditioner

    base = pipeline.pipeline.models["sparse_structure_flow_model"].float()
    fusion = ConfidenceSparseVoxelFusion(
        grid_resolution=16,
        input_feature_dim=1024,
        projected_feature_dim=1024,
        positional_frequencies=4,
        fusion_mode="confidence",
        use_vggt_depth=True,
        use_vggt_pointmap=True,
        use_confidence_weighting=True,
        use_visibility_weighting=True,
        use_3d_positional_encoding=True,
        require_spconv=True,
        use_spconv_refinement=False,
        conditioning_layout="affostruction",
    ).to(device=device, dtype=torch.float32)
    branch = TrainableVoxelSSBranch(
        fusion,
        AffostructionSSFlow(
            base,
            condition_dim=1024,
            classifier_free_dropout=0.0,
            gradient_checkpointing=False,
        ),
    ).to(device=device, dtype=torch.float32)
    _validate_checkpoint(payload, SS_ARCHITECTURE, args.ss_checkpoint)
    branch.load_state_dict(payload["model"], strict=True)
    del payload
    branch.eval().requires_grad_(False)
    pipeline.install_direct_ss_flow(branch.flow.backbone)
    branch.fusion.checkpoint_architecture = SS_ARCHITECTURE
    return branch.fusion


def _load_exact_depths(
    depth_root: Path,
    split: str,
    uid: str,
    frame_ids: list[str],
) -> torch.Tensor:
    depths = []
    for frame_id in frame_ids:
        path = depth_root / split / uid / f"{frame_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Official RGBD inference depth is missing for exact frame {frame_id}: {path}"
            )
        with np.load(path, allow_pickle=False) as archive:
            encoded = np.asarray(archive["depth_u16"], dtype=np.uint16)
            depth_min = float(archive["depth_min"])
            depth_max = float(archive["depth_max"])
        depth = np.zeros(encoded.shape, dtype=np.float32)
        foreground = encoded > 0
        depth[foreground] = depth_min + (
            (encoded[foreground].astype(np.float32) - 1.0) / 65534.0
        ) * (depth_max - depth_min)
        if not np.isfinite(depth).all() or np.any(depth < 0.0):
            raise ValueError(f"Invalid cached depth values: {path}")
        depths.append(torch.from_numpy(depth))
    return torch.stack(depths)


def _install_slat(args, pipeline: RealTrellisGeoPipeline, device: torch.device):
    if not Path(args.slat_checkpoint).is_file():
        raise FileNotFoundError(f"SLat checkpoint not found: {args.slat_checkpoint}")
    payload = _load_checkpoint(args.slat_checkpoint)
    _validate_checkpoint(payload, SLAT_ARCHITECTURE, args.slat_checkpoint)
    model_cfg = payload.get("config", {}).get("model", {})
    base = pipeline.pipeline.models["slat_flow_model"]
    base.eval().requires_grad_(False)
    spatial_flow = build_affostruction_image_slat_denoiser(
        args.affostruction_root,
        slat_channels=int(base.in_channels),
        condition_channels=1024,
        gradient_checkpointing=False,
    )
    conditioner = SymmetricSLatConditioner(
        resolution=int(model_cfg.get("resolution", 64)),
        high_feature_dim=1024,
        low_feature_dim=int(model_cfg.get("low_feature_dim", 64)),
        condition_dim=1024,
        depth_temperature=float(model_cfg.get("depth_temperature", 0.04)),
        occlusion_margin=float(model_cfg.get("occlusion_margin", 0.03)),
        minimum_incidence=float(model_cfg.get("minimum_incidence", 0.05)),
    )
    branch = TrainableSymmetricSLatBranch(
        AffostructionSLatFlow(
            spatial_flow,
            conditioner,
            classifier_free_dropout=0.0,
            residual_limit=float(model_cfg.get("residual_limit", 1.0)),
        )
    ).to(device=device, dtype=torch.float32)
    branch.load_state_dict(payload["model"], strict=True)
    del payload
    branch.eval().requires_grad_(False)
    pipeline.install_direct_slat_flow(
        branch.flow.spatial_flow,
        residual_limit=branch.flow.residual_limit,
    )
    return branch.flow.conditioner


def _load_vggt(args, device):
    source = None
    if args.vggt_checkpoint is None:
        source = resolve_local_hf_snapshot(
            args.vggt_pretrained,
            args.hf_cache_root,
            required_file="model.safetensors",
        )
    return VGGTGeometryWrapper(
        vggt_root=args.vggt_root,
        checkpoint=args.vggt_checkpoint,
        pretrained_name=source,
        mock=False,
        cache_features=True,
        require_real=True,
        vggt_image_size=518,
    ).to(device).eval()


def _load_checkpoint(path: str):
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _validate_checkpoint(payload, architecture: str, path: str) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError(f"Checkpoint is not a full training payload: {path}")
    if payload.get("architecture_version") != architecture:
        raise RuntimeError(
            f"Checkpoint architecture={payload.get('architecture_version')!r}; expected {architecture!r}"
        )
    if payload.get("optimization_numerics_version") != NUMERICS_VERSION:
        raise RuntimeError(
            f"Checkpoint numerics={payload.get('optimization_numerics_version')!r}; expected {NUMERICS_VERSION!r}"
        )


def _read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"UID manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    uids = payload.get("uids") if isinstance(payload, dict) else payload
    if not isinstance(uids, list) or not all(isinstance(uid, str) for uid in uids):
        raise ValueError(f"UID manifest must contain a string list: {path}")
    return uids


def _completed_inference(
    out_dir: Path,
    metrics_path: Path,
    *,
    method: str,
    uid: str,
) -> bool:
    required = (
        out_dir / "asset_gaussian.ply",
        out_dir / "asset_mesh_internal.ply",
        out_dir / "predicted_ss_occ.npz",
        out_dir / "trellis_latents.pt",
    )
    if not metrics_path.is_file() or not all(
        path.is_file() and path.stat().st_size > 0 for path in required
    ):
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with np.load(out_dir / "predicted_ss_occ.npz", allow_pickle=False) as payload:
            occupancy = payload["occ"]
            occupancy_valid = occupancy.ndim == 3 and occupancy.size > 0
        return (
            metrics.get("status") == "ok"
            and metrics.get("method") == method
            and metrics.get("uid") == uid
            and occupancy_valid
        )
    except (OSError, KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return False


def _cleanup_incomplete_output(out_dir: Path) -> None:
    if not out_dir.is_dir():
        return
    generated = (
        "asset_gaussian.ply",
        "asset_mesh_internal.ply",
        "asset_mesh.glb",
        "predicted_ss_occ.npz",
        "trellis_latents.pt",
        "metrics.json",
    )
    for name in generated:
        (out_dir / name).unlink(missing_ok=True)
    for pattern in (".*.tmp*", "*.tmp", "metrics.json.*.tmp"):
        for temporary in out_dir.glob(pattern):
            if temporary.is_file():
                temporary.unlink(missing_ok=True)


def _require_free_disk(path: Path, minimum_free_gb: float) -> None:
    free_bytes = shutil.disk_usage(path).free
    required_bytes = int(minimum_free_gb * 2**30)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient free storage at {path}: {free_bytes / 2**30:.2f} GiB available, "
            f"{minimum_free_gb:.2f} GiB safety reserve required. Free space and resume with --no-overwrite."
        )


def _is_disk_exhaustion(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
            return True
        current = current.__cause__ or current.__context__
    return False


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shard one Affostruction inference method over torchrun GPUs"
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--uid-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--category")
    parser.add_argument("--conditioning-view-set", choices=("renders", "renders_cond"), default="renders")
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--conditioning-image-size", type=int, default=518)
    parser.add_argument("--occ-resolution", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--minimum-free-disk-gb",
        type=float,
        default=2.0,
        help="Stop before inference when the output filesystem falls below this safety reserve.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-image-mode", choices=("multidiffusion", "stochastic"), default="multidiffusion")
    parser.add_argument("--ss-steps", type=int, default=12)
    parser.add_argument("--ss-cfg-strength", type=float, default=7.5)
    parser.add_argument("--slat-steps", type=int, default=12)
    parser.add_argument("--slat-cfg-strength", type=float, default=3.0)
    parser.add_argument(
        "--ss-amp-dtype",
        choices=("native", "fp16", "bf16"),
        default="fp16",
        help="Autocast used only by the fine-tuned SS stage.",
    )
    parser.add_argument(
        "--slat-amp-dtype",
        choices=("native", "fp16", "bf16"),
        default="fp16",
        help="Autocast used only by the fine-tuned SLat stage; fp16 is the A6000-compatible fast path.",
    )
    parser.add_argument(
        "--slat-spconv-algo",
        choices=("auto", "native", "implicit_gemm"),
        default="native",
        help="Sparse convolution algorithm for runs containing fine-tuned SLat; native covers dynamic A6000 supports.",
    )
    parser.add_argument("--ss-checkpoint", default="outputs/affostruction_ss_official/affostruction_official_ss_last.pt")
    parser.add_argument("--slat-checkpoint", default="outputs/affostruction_slat_fp32/geovis_slat_adapter_last.pt")
    parser.add_argument("--trellis-root", default="/mnt/sda/hf/MVG/Base/TRELLIS")
    parser.add_argument("--affostruction-root", default="/mnt/sda/hf/MVG/Base/Affostruction")
    parser.add_argument("--trellis-model-path", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--vggt-root", default="/mnt/sda/hf/MVG/Base/vggt")
    parser.add_argument("--vggt-pretrained", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--hf-cache-root", default="/mnt/sda/hf/.cache/huggingface/hub")
    parser.add_argument("--torch-hub-dir", default="/mnt/sda/hf/.cache/torch/hub")
    parser.add_argument("--dinov2-repo", default="/mnt/sda/hf/.cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser


if __name__ == "__main__":
    main()
