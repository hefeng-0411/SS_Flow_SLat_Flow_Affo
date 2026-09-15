from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.datasets.objaverse_cars_rendered_dataset import ObjaverseCarsRenderedDataset
from geoss.datasets.srn_cars_dataset import SRNCarsDataset
from geoss.datasets.vehicle_multiview_dataset import VehicleMultiViewDataset, make_dry_run_batch
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper
from geoss.integration.trellis_residency import configure_trellis_training_residency
from geoss.integration.trellis_hub import configure_trellis_hub, resolve_local_hf_snapshot
from geoss.geometry.alignment import align_vggt_batch
from geoss.slat.integration.ss_slat_context import build_ss_slat_context
from geoss.slat.losses.appearance_feature_loss import appearance_feature_loss
from geoss.slat.losses.slat_flow_loss import slat_flow_matching_loss
from geoss.slat.losses.slat_prior_preservation_loss import slat_prior_preservation_loss
from geoss.slat.losses.slat_velocity_loss import slat_velocity_regularization_loss
from geoss.slat.losses.view_consistency_loss import view_consistency_loss
from geoss.slat.losses.visibility_confidence_loss import visibility_confidence_loss
from geoss.slat.losses.factorized_control_loss import factorized_control_loss
from geoss.slat.losses.decoded_asset_loss import DecodedAssetSupervisor
from geoss.slat.models.geovis_slat_adapter import GeoVisSLATAdapter
from geoss.slat.models.slat_flow_adapter import (
    AffostructionSLatFlow,
    SymmetricSLatConditioner,
    build_affostruction_image_slat_denoiser,
)
from geoss.slat.utils.normalization import SLAT_TENSOR_CONTRACT_VERSION, normalize_slat
from geoss.slat.utils.slat_visualization import save_slat_debug_npz, write_active_voxels_ply
from geoss.utils.adaptive_batch import AdaptiveBatchController, adaptive_config_defaults, add_adaptive_batch_args
from geoss.utils.checkpoint import save_checkpoint
from geoss.utils.config import add_common_args, apply_config_mappings, load_config, str2bool
from geoss.utils.run_mode import validate_real_mode
from geoss.utils.training_budget import (
    compute_training_budget,
    defer_nonfatal_early_stop_until_minimum_exposure,
    enforce_minimum_dataset_passes,
)
from geoss.utils.distributed import (
    build_dataloader,
    cleanup_distributed,
    init_distributed,
    maybe_wrap_ddp,
    next_from_loader,
    sync_should_stop,
    unwrap_model,
)
from geoss.utils.early_stopping import (
    EarlyStopper,
    apply_early_stop_action,
    distributed_early_stop_update,
    quarantine_legacy_best_checkpoint,
)


SLAT_OPTIMIZATION_NUMERICS_VERSION = "frozen_trellis_sparse_image_flow_fp32_master_v1"


class TrainableSymmetricSLatBranch(nn.Module):
    """DDP-owned sparse correction flow and pixel-aligned VGGT conditioner."""

    architecture_version = AffostructionSLatFlow.architecture_version

    def __init__(self, flow: AffostructionSLatFlow) -> None:
        super().__init__()
        self.flow = flow

    def forward(self, batch: dict) -> dict:
        sparse_state = _padded_slat_state_to_sparse(
            batch["slat_latent_tokens"],
            batch["slat_indices"],
            batch["slat_token_valid_mask"],
        )
        base_velocity = _padded_slat_state_to_sparse(
            batch["v_slat_base"],
            batch["slat_indices"],
            batch["slat_token_valid_mask"],
        )
        confidence = batch["vggt_confidence"]
        if confidence.ndim == 4:
            confidence = confidence[:, :, None]
        alignment_confidence = batch.get("alignment_confidence")
        if isinstance(alignment_confidence, torch.Tensor):
            confidence = confidence * alignment_confidence.to(confidence)
        result = self.flow(
            sparse_state,
            batch["timestep"] * 1000.0,
            base_velocity=base_velocity,
            active_indices=batch["slat_indices"],
            images=batch["images"],
            high_features=batch["vggt_features"],
            intrinsics=batch["K"],
            world_to_camera=batch["w2c"],
            foreground_masks=batch["masks"],
            aligned_depth=batch["aligned_depth"],
            vggt_confidence=confidence,
            active_valid_mask=batch["slat_token_valid_mask"],
            view_valid_mask=batch.get("view_valid_mask"),
            aligned_point_map=batch.get("aligned_pointmap"),
        )
        velocity = _sparse_slat_velocity_to_padded(
            result.velocity,
            batch["slat_indices"],
            batch["slat_token_valid_mask"],
        )
        return _pack_direct_slat_output(result, velocity, batch["v_slat_base"])


class _DryRunSparseState:
    def __init__(self, features: torch.Tensor, coords: torch.Tensor) -> None:
        self.feats = features
        self.coords = coords
        self.shape = torch.Size((1, features.shape[-1]))

    def replace(self, features: torch.Tensor):
        return _DryRunSparseState(features, self.coords)


class _DryRunSLatBackbone(nn.Module):
    def __init__(self, latent_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.in_channels = int(latent_dim)
        self.cond_channels = int(condition_dim)
        self.state_scale = nn.Parameter(torch.ones(()))
        self.condition_to_velocity = nn.Linear(condition_dim, latent_dim)

    def forward(
        self,
        state: _DryRunSparseState,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ):
        del timestep
        if cond_mask is None:
            pooled = condition.mean(dim=1)
        else:
            weights = cond_mask[..., None].to(condition.dtype)
            pooled = (condition * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        condition_velocity = self.condition_to_velocity(pooled)[0]
        return _DryRunSparseState(
            state.feats * self.state_scale + condition_velocity[None],
            state.coords,
        )


def _pack_direct_slat_output(
    result,
    velocity: torch.Tensor,
    base_velocity_padded: torch.Tensor,
) -> dict:
    conditioning = result.conditioning
    base_velocity = getattr(result.base_velocity, "feats", result.base_velocity)
    raw_correction = getattr(result.raw_correction, "feats", result.raw_correction)
    correction = getattr(result.correction, "feats", result.correction)
    return {
        "velocity": velocity,
        "v_slat_geo": velocity,
        "v_slat_base": base_velocity_padded,
        "delta_v_slat": velocity - base_velocity_padded,
        "v_slat_base_sparse": base_velocity,
        "raw_correction_sparse": raw_correction,
        "correction_sparse": correction,
        "slat_cond_tokens": conditioning.condition,
        "slat_confidence": conditioning.confidence,
        "active_xyz": conditioning.active_xyz,
        "visibility": conditioning.occlusion_mask.float(),
        "view_weights": conditioning.view_weights,
        "sampled_features": conditioning.sampled_high,
        "sampled_rgb": conditioning.sampled_low,
        "depth_residual": 1.0 - conditioning.agreement,
        "occlusion_score": 1.0 - conditioning.occlusion_mask.float(),
        "condition_valid": conditioning.condition_valid,
    }


def run_dry_run(cfg: dict, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    batch = make_synthetic_slat_batch(cfg, args, device)
    model_cfg = cfg.get("model", {})
    condition_dim = 64
    conditioner = SymmetricSLatConditioner(
        resolution=int(model_cfg.get("resolution", 64)),
        high_feature_dim=1024,
        low_feature_dim=int(model_cfg.get("low_feature_dim", 64)),
        condition_dim=condition_dim,
        depth_temperature=float(model_cfg.get("depth_temperature", 0.04)),
        occlusion_margin=float(model_cfg.get("occlusion_margin", 0.03)),
        minimum_incidence=float(model_cfg.get("minimum_incidence", 0.05)),
    )
    flow = AffostructionSLatFlow(
        _DryRunSLatBackbone(int(model_cfg.get("slat_dim", 8)), condition_dim),
        conditioner,
        classifier_free_dropout=0.0,
        residual_limit=float(model_cfg.get("residual_limit", 1.0)),
    ).to(device)
    dry_coords = torch.cat(
        [
            torch.zeros(
                batch["slat_indices"].shape[1],
                1,
                device=device,
                dtype=batch["slat_indices"].dtype,
            ),
            batch["slat_indices"][0],
        ],
        dim=-1,
    ).int()
    base_velocity = _DryRunSparseState(batch["v_slat_base"][0], dry_coords)
    result = flow(
        _DryRunSparseState(batch["slat_latent_tokens"][0], dry_coords),
        batch["timestep"] * 1000.0,
        base_velocity=base_velocity,
        active_indices=batch["slat_indices"],
        images=batch["images"],
        high_features=batch["vggt_features"],
        intrinsics=batch["K"],
        world_to_camera=batch["w2c"],
        foreground_masks=batch["masks"],
        aligned_depth=batch["aligned_depth"],
        vggt_confidence=batch["vggt_confidence"],
        active_valid_mask=batch["slat_token_valid_mask"],
        view_valid_mask=batch["view_valid_mask"],
    )
    out = _pack_direct_slat_output(
        result,
        result.velocity.feats[None],
        batch["v_slat_base"],
    )
    terms = compute_direct_slat_losses(out, batch, cfg.get("loss"))
    terms["decoded_asset"] = {"loss": out["velocity"].sum() * 0.0}
    summary = summarize_direct_slat(out, terms, "synthetic_dry_run")
    summary.update({"synthetic_spatial_flow": True, "not_for_paper_metrics": True})
    write_outputs(Path(args.output_dir), out, summary)
    return summary


def run_training(cfg: dict, args: argparse.Namespace) -> dict:
    run_modes = validate_real_mode(cfg=cfg, args=args, mode="real_train", required=("vggt", "trellis", "dataset"))
    ctx = init_distributed(args)
    device = ctx.device
    batch_controller = AdaptiveBatchController.from_args(args)
    args.batch_size = batch_controller.batch_size
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    if args.steps is None:
        raise ValueError("real_train requires an explicit --steps value or a steps entry in the config.")
    loader, sampler = build_real_loader(args, ctx)
    if loader is None:
        raise FileNotFoundError("real_train requires a non-empty real dataset loader; synthetic SLAT batches are only allowed in --dry_run.")
    training_budget = compute_training_budget(
        dataset_objects=len(loader.dataset),
        world_size=ctx.world_size,
        microbatch_per_rank=args.batch_size,
        grad_accum_steps=max(1, int(args.grad_accum_steps)),
        planned_optimizer_updates=int(args.steps),
        drop_last=bool(getattr(loader, "drop_last", False)),
    )
    enforce_minimum_dataset_passes(
        training_budget,
        minimum_dataset_passes=args.minimum_dataset_passes,
        stage="SLAT",
    )
    support_provenance = {
        "coordinate_source": "cached_trellis_slat_ground_truth_support",
        "slat_value_source": "cached_trellis_slat_teacher",
        "support_schedule": "ground_truth_ss_support_teacher_forcing_train_predicted_ss_support_inference",
        "upstream_ss_checkpoint": cfg.get("upstream_ss_checkpoint"),
        "cross_stage_gradient": "independent_sparse_image_flow_and_conditioner_only",
        "train_inference_support_match": False,
        "propagation": AffostructionSLatFlow.architecture_version,
    }
    if ctx.is_main:
        preflight_dir = Path(args.output_dir)
        preflight_dir.mkdir(parents=True, exist_ok=True)
        (preflight_dir / "training_preflight.json").write_text(
            json.dumps(
                {
                    "stage": "SLAT",
                    "training_budget": training_budget.as_dict(),
                    "minimum_dataset_passes": args.minimum_dataset_passes,
                    "support_provenance": support_provenance,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    decoded_config = cfg.get("decoded_supervision") if isinstance(cfg.get("decoded_supervision"), dict) else {}
    required_trellis_models = ["slat_flow_model", "image_cond_model"]
    if bool(decoded_config.get("enabled", False)):
        required_trellis_models.append("slat_decoder_gs")
    trellis_pipeline = _load_trellis_pipeline(args, device, ctx, required_models=required_trellis_models)
    decoded_supervisor = DecodedAssetSupervisor(trellis_pipeline, cfg.get("decoded_supervision"))
    vggt_geometry = _load_vggt_geometry(args, device, ctx)
    model_cfg = dict(cfg.get("model", {}))
    actual_slat_dim = int(trellis_pipeline.models["slat_flow_model"].in_channels)
    if int(model_cfg.get("slat_dim", actual_slat_dim)) != actual_slat_dim:
        raise ValueError(
            f"Configured slat_dim={model_cfg.get('slat_dim')} does not match TRELLIS {actual_slat_dim}"
        )
    flow_backbone = trellis_pipeline.models["slat_flow_model"]
    flow_backbone.eval().requires_grad_(False)
    condition_dim = 1024
    conditioner = SymmetricSLatConditioner(
        resolution=int(model_cfg.get("resolution", 64)),
        high_feature_dim=1024,
        low_feature_dim=int(model_cfg.get("low_feature_dim", 64)),
        condition_dim=condition_dim,
        depth_temperature=float(model_cfg.get("depth_temperature", 0.04)),
        occlusion_margin=float(model_cfg.get("occlusion_margin", 0.03)),
        minimum_incidence=float(model_cfg.get("minimum_incidence", 0.05)),
    )
    spatial_flow = build_affostruction_image_slat_denoiser(
        args.affostruction_root,
        slat_channels=actual_slat_dim,
        condition_channels=condition_dim,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model = TrainableSymmetricSLatBranch(
        AffostructionSLatFlow(
            spatial_flow,
            conditioner,
            classifier_free_dropout=float(model_cfg.get("classifier_free_dropout", 0.1)),
            residual_limit=float(model_cfg.get("residual_limit", 1.0)),
        )
    ).to(device=device, dtype=torch.float32)
    _assert_fp32_finite_trainable_parameters(model, "SLat initialization")
    start_step = 0
    resume_state = None
    initialization_runtime_overrides = {}
    if args.resume and Path(args.resume).exists():
        resume_state = torch.load(args.resume, map_location="cpu")
        _validate_checkpoint_tensor_contract(resume_state, args.resume)
        _validate_affostruction_slat_checkpoint(resume_state, args.resume)
        resume_model_state = resume_state.get("model", resume_state)
        _assert_state_dict_finite(resume_model_state, f"SLat checkpoint {args.resume}")
        model.load_state_dict(resume_model_state, strict=True)
        start_step = int(resume_state.get("step", 0))
    elif args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"SLAT initialization checkpoint not found: {init_path}")
        init_state = torch.load(init_path, map_location="cpu")
        _validate_checkpoint_tensor_contract(init_state, init_path)
        _validate_affostruction_slat_checkpoint(init_state, init_path)
        init_model_state = init_state.get("model", init_state)
        _assert_state_dict_finite(init_model_state, f"SLat checkpoint {init_path}")
        model.load_state_dict(init_model_state, strict=True)
    model = maybe_wrap_ddp(model, ctx, find_unused_parameters=args.ddp_find_unused_parameters)
    unwrapped_model = unwrap_model(model)
    opt = torch.optim.AdamW(
        [
            {
                "params": unwrapped_model.flow.spatial_flow.parameters(),
                "lr": args.lr,
                "initial_lr": args.lr,
                "group_name": "affostruction_sparse_image_flow",
            },
            {
                "params": unwrapped_model.flow.conditioner.parameters(),
                "lr": args.conditioner_lr,
                "initial_lr": args.conditioner_lr,
                "group_name": "symmetric_conditioner",
            },
        ],
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
        fused=bool(args.fused_optimizer and device.type == "cuda"),
    )
    # A800/H100-class GPUs support BF16 natively.  Its wider exponent range
    # avoids the FP16 overflow that previously poisoned the velocity head.
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = _make_grad_scaler(enabled=args.amp and amp_dtype == torch.float16 and device.type == "cuda")
    clipper = _adaptive_clipper(args.trellis_root, args.max_grad_norm)
    if resume_state is not None and "optimizer" in resume_state:
        opt.load_state_dict(resume_state["optimizer"])
        if "grad_scaler" in resume_state:
            scaler.load_state_dict(resume_state["grad_scaler"])
        if "adaptive_grad_clipper" in resume_state:
            clipper.load_state_dict(resume_state["adaptive_grad_clipper"])
    iterator = iter(loader) if loader is not None else None
    if iterator is None:
        raise FileNotFoundError("real_train requires a non-empty real dataset loader; synthetic SLAT batches are only allowed in --dry_run.")
    data_epoch = 0
    out_dir = Path(args.output_dir)
    if ctx.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_geovis_slat.jsonl"
    last = {}
    early_stopper = EarlyStopper.from_args(args, default_metric="loss")
    if resume_state is not None:
        early_stopper.load_state_dict(resume_state.get("early_stopper"))
        if ctx.is_main:
            quarantine_legacy_best_checkpoint(
                out_dir / "geovis_slat_adapter_best.pt",
                resume_state,
            )
    end_step = int(args.steps) if args.steps_are_total else start_step + int(args.steps)
    update_contract = {
        "configured_steps": int(args.steps),
        "steps_are_total": bool(args.steps_are_total),
        "resume_start_step": start_step,
        "target_step": end_step,
        "remaining_update_attempts_at_start": max(0, end_step - start_step),
    }
    if start_step >= end_step:
        return {
            "step": start_step,
            "target_step": end_step,
            "mode": "already_complete",
            "rank": ctx.rank,
            "world_size": ctx.world_size,
            "training_budget": training_budget.as_dict(),
            "update_contract": update_contract,
            "support_provenance": support_provenance,
        }
    step = start_step
    while step < end_step:
        step += 1
        step_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _set_optimizer_learning_rates(
            opt,
            update=step - 1,
            total_updates=end_step,
            warmup_updates=args.warmup_steps,
            minimum_ratio=args.min_learning_rate_ratio,
        )
        if iterator is not None:
            raw_batch, iterator, data_epoch = next_from_loader(iterator, loader, sampler, data_epoch)
        else:
            raise RuntimeError("real_train unexpectedly has no real dataloader iterator.")

        def step_fn():
            nonlocal raw_batch, iterator, data_epoch
            grad_accum_steps = max(1, int(args.grad_accum_steps))
            opt.zero_grad(set_to_none=True)
            last_batch = None
            last_out = None
            last_terms = None
            total_loss = torch.zeros((), device=device)
            for micro_step in range(grad_accum_steps):
                if micro_step > 0:
                    raw_batch, iterator, data_epoch = next_from_loader(iterator, loader, sampler, data_epoch)
                batch = prepare_batch(
                    raw_batch, cfg, args, device,
                    trellis_pipeline=trellis_pipeline,
                    vggt_geometry=vggt_geometry,
                )
                sync_context = (
                    model.no_sync()
                    if ctx.distributed and hasattr(model, "no_sync") and micro_step < grad_accum_steps - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=args.amp and device.type == "cuda",
                    ):
                        out = model(batch)
                        terms = compute_direct_slat_losses(out, batch, cfg.get("loss"))
                        terms["decoded_asset"] = decoded_supervisor(out, batch, step)
                        loss = terms["slat_flow"]["loss"] + terms["decoded_asset"]["loss"]
                    if not torch.isfinite(loss).all().item():
                        raise FloatingPointError("Stage 3 loss became NaN/Inf before backward.")
                    scaler.scale(loss / grad_accum_steps).backward()
                total_loss = total_loss + loss.detach()
                last_batch, last_out, last_terms = batch, out, terms
            scaler.unscale_(opt)
            grad_norms, nonfinite_gradients = _inspect_direct_slat_gradients(unwrap_model(model), step)
            if nonfinite_gradients:
                # AMP overflow is recoverable.  Do not write NaN/Inf into the
                # checkpoint; discard this update and let GradScaler back off.
                opt.zero_grad(set_to_none=True)
                scaler.update()
                global_grad_norm = float("nan")
            else:
                global_grad_norm = float(clipper(model.parameters()).detach().cpu())
                scaler.step(opt)
                scaler.update()
                parameters_finite = _distributed_all_true(
                    _trainable_parameters_are_finite(unwrap_model(model)), device
                )
                if not parameters_finite:
                    raise FloatingPointError(
                        "AdamW produced a non-finite SLat parameter update; the last completed checkpoint remains the recovery boundary"
                    )
            assert last_batch is not None and last_out is not None and last_terms is not None
            last_terms["grad_norms"] = grad_norms
            last_terms["optimizer_step_skipped"] = bool(nonfinite_gradients)
            last_terms["global_grad_norm"] = global_grad_norm
            return last_batch, last_out, last_terms, total_loss / grad_accum_steps

        batch, out, terms, loss = step_fn()
        completed_batch_size = int(batch["slat_latent_tokens"].shape[0])
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        )
        batch_adjustment = batch_controller.update_after_success(device)
        if batch_adjustment.changed:
            args.batch_size = batch_adjustment.new_batch_size
            loader, sampler = build_real_loader(args, ctx)
            if sampler is not None:
                sampler.set_epoch(data_epoch)
            iterator = iter(loader) if loader is not None else None
        step_seconds = time.perf_counter() - step_started
        last = summarize_direct_slat(out, terms, "real_dataset")
        last.update(run_modes)
        last["step"] = step
        last["loss"] = float(loss.detach().cpu())
        last["rank"] = ctx.rank
        last["world_size"] = ctx.world_size
        last["per_gpu_batch_size"] = completed_batch_size
        last["next_per_gpu_batch_size"] = args.batch_size
        last["global_batch_size"] = completed_batch_size * ctx.world_size
        last["effective_global_batch_size"] = completed_batch_size * ctx.world_size * max(1, int(args.grad_accum_steps))
        last["step_seconds"] = step_seconds
        last["samples_per_second"] = (
            completed_batch_size * ctx.world_size * max(1, int(args.grad_accum_steps))
        ) / max(step_seconds, 1.0e-9)
        last["max_allocated_cuda_bytes"] = peak_allocated
        last["max_reserved_cuda_bytes"] = peak_reserved
        last["grad_accum_steps"] = max(1, int(args.grad_accum_steps))
        last["adapter_grad_norms"] = terms.get("grad_norms", {})
        last["optimizer_step_skipped"] = bool(terms.get("optimizer_step_skipped", False))
        last["global_grad_norm"] = terms.get("global_grad_norm")
        last["learning_rates"] = {
            str(group.get("group_name", index)): float(group["lr"])
            for index, group in enumerate(opt.param_groups)
        }
        last["adaptive_batch"] = {**batch_controller.state_dict(), "last_adjustment": batch_adjustment.as_dict()}
        last["trellis_residency"] = getattr(trellis_pipeline, "training_residency", None)
        last["initialization_runtime_control_overrides"] = initialization_runtime_overrides
        last["training_budget"] = training_budget.as_dict()
        last["update_contract"] = update_contract
        last["support_provenance"] = support_provenance
        early_status = distributed_early_stop_update(
            early_stopper,
            last,
            rank=ctx.rank,
        )
        last["minimum_exposure_gate"] = defer_nonfatal_early_stop_until_minimum_exposure(
            early_status,
            budget=training_budget,
            step=step,
            minimum_dataset_passes=args.minimum_dataset_passes,
        )
        last["early_stop_action"] = apply_early_stop_action(opt, early_status)
        last["early_stop"] = early_status.as_dict()
        if ctx.is_main:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(last) + "\n")
        if ctx.is_main and args.save_best and early_status.is_best and early_status.handoff_ready:
            _save_slat_checkpoint(out_dir / "geovis_slat_adapter_best.pt", model, opt, scaler, clipper, step, cfg, early_stopper, early_status, training_budget=training_budget.as_dict(), support_provenance=support_provenance)
        if ctx.is_main and early_status.is_candidate:
            _save_slat_checkpoint(out_dir / "geovis_slat_adapter_candidate.pt", model, opt, scaler, clipper, step, cfg, early_stopper, early_status, training_budget=training_budget.as_dict(), support_provenance=support_provenance)
        if ctx.is_main and args.visualize_every > 0 and step % args.visualize_every == 0:
            write_outputs(out_dir, out, last)
        should_fault_save = args.fault_tolerant_save_every > 0 and step % args.fault_tolerant_save_every == 0
        periodic_save = args.save_every > 0 and step % args.save_every == 0
        if ctx.is_main and (should_fault_save or periodic_save or step == end_step):
            _save_slat_checkpoint(out_dir / "geovis_slat_adapter_last.pt", model, opt, scaler, clipper, step, cfg, early_stopper, early_status, training_budget=training_budget.as_dict(), support_provenance=support_provenance)
        archive_save = args.archive_every > 0 and step % args.archive_every == 0
        if ctx.is_main and archive_save:
            _save_slat_checkpoint(
                out_dir / f"geovis_slat_adapter_step_{step:08d}.pt",
                model, opt, scaler, clipper, step, cfg, early_stopper, early_status,
                training_budget=training_budget.as_dict(), support_provenance=support_provenance,
            )
        if sync_should_stop(early_status.should_stop, device):
            if ctx.is_main:
                _save_slat_checkpoint(out_dir / "geovis_slat_adapter_last.pt", model, opt, scaler, clipper, step, cfg, early_stopper, early_status, training_budget=training_budget.as_dict(), support_provenance=support_provenance)
            break
    if ctx.is_main:
        write_outputs(out_dir, out, last)
    return last


def compute_direct_slat_losses(
    out: dict,
    batch: dict,
    loss_config: dict | None = None,
) -> dict:
    """Continuous SLat CFM, endpoint recovery, and frozen-prior preservation."""

    loss_config = dict(loss_config or {})
    velocity = out["velocity"]
    target = batch["target_velocity"].detach()
    mask = batch["slat_supervision_mask"].float()
    if velocity.shape != target.shape or mask.shape != (*target.shape[:2], 1):
        raise ValueError(
            f"Direct SLat loss contract mismatch: velocity={tuple(velocity.shape)}, "
            f"target={tuple(target.shape)}, mask={tuple(mask.shape)}"
        )
    denominator = (mask.sum() * velocity.shape[-1]).clamp_min(1.0)
    flow_loss = ((velocity.float() - target.float()).square() * mask).sum() / denominator
    base_velocity = batch["v_slat_base"].detach().float()
    base_flow_loss = ((base_velocity - target.float()).square() * mask).sum() / denominator
    correction = velocity.float() - base_velocity
    target_residual = target.float() - base_velocity
    residual_loss = ((correction - target_residual).square() * mask).sum() / denominator
    timestep = batch["timestep"].float().reshape(velocity.shape[0], 1, 1)
    sigma_min = float(batch["flow_sigma_min"])
    sigma_t = sigma_min + (1.0 - sigma_min) * timestep
    predicted_x0 = (1.0 - sigma_min) * batch["slat_latent_tokens"].float() - sigma_t * velocity.float()
    endpoint_loss = (
        (predicted_x0 - batch["slat_clean_tokens"].detach().float()).square() * mask
    ).sum() / denominator
    low_confidence = (1.0 - out["slat_confidence"].detach().float()) * mask
    prior_denominator = (low_confidence.sum() * velocity.shape[-1]).clamp_min(1.0)
    prior_loss = (correction.square() * low_confidence).sum() / prior_denominator
    endpoint_weight = float(loss_config.get("endpoint_weight", 0.25))
    prior_weight = float(loss_config.get("prior_weight", 0.01))
    loss = flow_loss + endpoint_weight * endpoint_loss + prior_weight * prior_loss
    return {
        "slat_flow": {
            "loss": loss,
            "slat_flow_mse": flow_loss,
            "raw_residual_mse": residual_loss,
            "effective_residual_mse": residual_loss,
            "frozen_base_residual_mse": base_flow_loss,
            "endpoint_x0_mse": endpoint_loss,
        },
        "base_velocity": {"invalid_ratio": batch["v_slat_base_invalid_ratio"]},
        "prior": {"loss": prior_loss},
        "velocity": {"loss": velocity.float().square().mean()},
    }


def summarize_direct_slat(out: dict, terms: dict, mode: str) -> dict:
    decoded = terms.get("decoded_asset", {"loss": out["velocity"].new_zeros(())})
    return {
        "mode": mode,
        "architecture_version": AffostructionSLatFlow.architecture_version,
        "active_xyz": list(out["active_xyz"].shape),
        "slat_cond_tokens": list(out["slat_cond_tokens"].shape),
        "v_slat_geo": list(out["velocity"].shape),
        "slat_confidence_mean": float(out["slat_confidence"].mean().detach().cpu()),
        "slat_confidence_std": float(out["slat_confidence"].std(unbiased=False).detach().cpu()),
        "visibility_mean": float(out["visibility"].mean().detach().cpu()),
        "view_weights_std": float(out["view_weights"].std(unbiased=False).detach().cpu()),
        "loss_slat_flow": float(terms["slat_flow"]["loss"].detach().cpu()),
        "loss_cfm_mse": float(terms["slat_flow"]["slat_flow_mse"].detach().cpu()),
        "loss_frozen_base_cfm_mse": float(
            terms["slat_flow"]["frozen_base_residual_mse"].detach().cpu()
        ),
        "loss_endpoint_x0": float(
            terms["slat_flow"]["endpoint_x0_mse"].detach().cpu()
        ),
        "loss_prior_preservation": float(terms["prior"]["loss"].detach().cpu()),
        "loss_decoded_asset": float(decoded["loss"].detach().cpu()),
        "loss_decoded_render": float(decoded.get("render_loss", decoded["loss"]).detach().cpu()),
        "loss_decoded_geometry": float(decoded.get("geometry_loss", decoded["loss"]).detach().cpu()),
    }


def compute_losses(
    out: dict,
    batch: dict,
    *,
    raw_residual_weight: float = 1.0,
    effective_residual_weight: float = 1.0,
) -> dict:
    target_residual = batch.get("target_residual", batch["target_velocity"] - batch["v_slat_base"])
    velocity_debug = out.get("debug", {}).get("velocity", {})
    raw_delta = velocity_debug.get("delta_raw", out["delta_v_slat"])
    effective_delta = out["v_slat_geo"] - batch["v_slat_base"]
    _assert_slat_training_contract(batch, out, raw_delta, effective_delta, target_residual)
    supervision_mask = batch.get("slat_supervision_mask")
    raw_flow = slat_flow_matching_loss(raw_delta, target_residual.detach(), supervision_mask)
    effective_flow = slat_flow_matching_loss(effective_delta, target_residual.detach(), supervision_mask)
    frozen_base_flow = slat_flow_matching_loss(
        torch.zeros_like(effective_delta), target_residual.detach(), supervision_mask
    )
    flow_loss = raw_residual_weight * raw_flow["loss"] + effective_residual_weight * effective_flow["loss"]
    base_valid_mask = batch.get("v_slat_base_valid_mask")
    joint_confidence = out["debug"]["joint_confidence"]
    if base_valid_mask is not None:
        assert base_valid_mask.shape == joint_confidence.shape, (
            f"v_slat_base_valid_mask shape {tuple(base_valid_mask.shape)} "
            f"must match joint_confidence {tuple(joint_confidence.shape)}"
        )
        # The prior loss uses weight=(1-confidence). Invalid frozen-teacher
        # tokens must therefore be assigned confidence=1 so they contribute
        # zero preservation pressure instead of preserving the placeholder base.
        joint_confidence = torch.where(base_valid_mask.bool(), joint_confidence, torch.ones_like(joint_confidence))
    base_invalid_ratio = batch.get("v_slat_base_invalid_ratio", torch.zeros((), device=out["v_slat_geo"].device))
    token_valid_mask = batch.get("slat_token_valid_mask")
    factorized = factorized_control_loss(
        out["correction_demand"],
        out["residual_variance"],
        effective_delta,
        target_residual,
        token_mask=supervision_mask,
        correction_demand_logits=out.get("correction_demand_logits"),
    )
    return {
        "slat_flow": {
            "loss": flow_loss,
            "slat_flow_mse": flow_loss,
            "raw_residual_mse": raw_flow["loss"],
            "effective_residual_mse": effective_flow["loss"],
            "frozen_base_residual_mse": frozen_base_flow["loss"],
        },
        "base_velocity": {"invalid_ratio": base_invalid_ratio.detach()},
        "view": view_consistency_loss(out["sampled_features"], out["visibility"], token_valid_mask),
        "appearance": appearance_feature_loss(
            out["slat_cond_tokens"], out["sampled_features"], out["visibility"], out["view_weights"], token_valid_mask
        ),
        "visibility_confidence": visibility_confidence_loss(
            out["slat_confidence"],
            out["visibility"],
            out["depth_residual"],
            appearance_conflict=out["appearance_conflict"],
            occlusion_score=out["occlusion_score"],
            token_valid_mask=token_valid_mask,
        ),
        "factorized_control": factorized,
        "velocity": slat_velocity_regularization_loss(out["delta_v_slat"], batch["timestep"], token_valid_mask),
        "prior": slat_prior_preservation_loss(out["v_slat_geo"], batch["v_slat_base"], joint_confidence),
    }


def summarize(out: dict, terms: dict, mode: str) -> dict:
    return {
        "mode": mode,
        "active_xyz": list(out["active_xyz"].shape),
        "slat_cond_tokens": list(out["slat_cond_tokens"].shape),
        "v_slat_geo": list(out["v_slat_geo"].shape),
        "slat_confidence_mean": float(out["slat_confidence"].mean().detach().cpu()),
        "slat_confidence_std": float(out["slat_confidence"].std(unbiased=False).detach().cpu()),
        "visibility_mean": float(out["visibility"].mean().detach().cpu()),
        "view_weights_std": float(out["view_weights"].std(unbiased=False).detach().cpu()),
        "delta_norm": float(out["delta_v_slat"].norm(dim=-1).mean().detach().cpu()),
        "clipping_ratio": float(out["debug"]["clipping_ratio"].detach().cpu()),
        "loss_slat_flow": float(terms["slat_flow"]["loss"].detach().cpu()),
        "loss_slat_raw_residual": float(terms["slat_flow"]["raw_residual_mse"].detach().cpu()),
        "loss_slat_effective_residual": float(terms["slat_flow"]["effective_residual_mse"].detach().cpu()),
        "loss_slat_frozen_base_residual": float(terms["slat_flow"]["frozen_base_residual_mse"].detach().cpu()),
        "slat_causal_residual_gain": float(
            (
                terms["slat_flow"]["frozen_base_residual_mse"]
                - terms["slat_flow"]["effective_residual_mse"]
            ).detach().cpu()
        ),
        "slat_base_invalid_ratio": float(terms["base_velocity"]["invalid_ratio"].detach().cpu()),
        "loss_prior": float(terms["prior"]["loss"].detach().cpu()),
        "loss_velocity": float(terms["velocity"]["loss"].detach().cpu()),
        "loss_decoded_asset": float(terms["decoded_asset"]["loss"].detach().cpu()) if "decoded_asset" in terms else 0.0,
        "loss_decoded_render": float(terms["decoded_asset"].get("render_loss", terms["decoded_asset"]["loss"]).detach().cpu()) if "decoded_asset" in terms else 0.0,
        "loss_decoded_geometry": float(terms["decoded_asset"].get("geometry_loss", terms["decoded_asset"]["loss"]).detach().cpu()) if "decoded_asset" in terms else 0.0,
    }


def _assert_slat_training_contract(
    batch: dict,
    out: dict,
    raw_delta: torch.Tensor,
    effective_delta: torch.Tensor,
    target_residual: torch.Tensor,
) -> None:
    required_batch = ("slat_latent_tokens", "v_slat_base", "target_velocity", "timestep", "images", "masks", "K", "w2c")
    for key in required_batch:
        assert key in batch, f"Stage 3 batch missing '{key}'."
    x_t = batch["slat_latent_tokens"]
    v_base = batch["v_slat_base"]
    assert x_t.ndim == 3, f"slat_latent_tokens must be [B,L,C], got {tuple(x_t.shape)}"
    assert v_base.shape == x_t.shape, f"v_slat_base shape {tuple(v_base.shape)} != slat_latent_tokens {tuple(x_t.shape)}"
    assert target_residual.shape == x_t.shape, f"target_residual shape {tuple(target_residual.shape)} != slat_latent_tokens {tuple(x_t.shape)}"
    assert raw_delta.shape == x_t.shape, f"raw SLAT residual shape {tuple(raw_delta.shape)} != slat_latent_tokens {tuple(x_t.shape)}"
    assert effective_delta.shape == x_t.shape, f"effective SLAT residual shape {tuple(effective_delta.shape)} != slat_latent_tokens {tuple(x_t.shape)}"
    base_valid_mask = batch.get("v_slat_base_valid_mask")
    if base_valid_mask is not None:
        assert base_valid_mask.shape == (*x_t.shape[:2], 1), (
            f"v_slat_base_valid_mask must be [B,L,1], got {tuple(base_valid_mask.shape)}"
        )
        assert base_valid_mask.dtype == torch.float32, f"v_slat_base_valid_mask must be float32, got {base_valid_mask.dtype}"
        assert torch.isfinite(base_valid_mask).all().item(), "v_slat_base_valid_mask contains NaN or Inf."
        assert base_valid_mask.min().item() >= 0.0 and base_valid_mask.max().item() <= 1.0, "v_slat_base_valid_mask must be in [0, 1]."
    for name, tensor in {
        "slat_latent_tokens": x_t,
        "v_slat_base": v_base,
        "target_residual": target_residual,
        "raw_delta": raw_delta,
        "effective_delta": effective_delta,
        "v_slat_geo": out["v_slat_geo"],
        "delta_v_slat": out["delta_v_slat"],
        "slat_cond_tokens": out["slat_cond_tokens"],
        "slat_confidence": out["slat_confidence"],
        "visibility": out["visibility"],
    }.items():
        assert tensor.dtype in (torch.float16, torch.bfloat16, torch.float32), f"{name} must be a floating point training dtype, got {tensor.dtype}"
        assert torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf."
    assert raw_delta.requires_grad, "raw SLAT residual must depend on GeoVisSLATAdapter parameters."
    assert effective_delta.requires_grad, "effective SLAT residual must depend on GeoVisSLATAdapter parameters."
    assert not target_residual.requires_grad, "target_residual must be a frozen flow-matching target."
    assert out["slat_confidence"].shape == (*x_t.shape[:2], 1), f"slat_confidence must be [B,L,1], got {tuple(out['slat_confidence'].shape)}"
    assert out["visibility"].shape[:2] == x_t.shape[:2], f"visibility must align with SLAT tokens, got {tuple(out['visibility'].shape)}"
    assert out["slat_confidence"].min().item() >= 0.0 and out["slat_confidence"].max().item() <= 1.0, "slat_confidence must be in [0, 1]."
    assert out["visibility"].min().item() >= 0.0 and out["visibility"].max().item() <= 1.0, "visibility must be in [0, 1]."


def _inspect_geovis_slat_gradients(model: GeoVisSLATAdapter, step: int) -> tuple[dict[str, float], list[str]]:
    critical = {
        "evidence_sampler.token_mlp.0.weight": model.evidence_sampler.token_mlp[0].weight,
        "aggregator.evidence_proj.weight": model.aggregator.evidence_proj.weight,
        "aggregator.slat_proj.weight": model.aggregator.slat_proj.weight,
        "aggregator.out.2.weight": model.aggregator.out[-1].weight,
        "velocity_adapter.latent_proj.weight": model.velocity_adapter.latent_proj.weight,
        "velocity_adapter.cond_proj.weight": model.velocity_adapter.cond_proj.weight,
        "velocity_adapter.delta_head.2.weight": model.velocity_adapter.delta_head[-1].weight,
    }
    missing = [name for name, param in critical.items() if param.grad is None]
    assert not missing, f"Stage 3 DDP graph break at step={step}; missing gradients for {missing}"
    nonfinite = [name for name, param in critical.items() if param.grad is not None and not torch.isfinite(param.grad).all().item()]
    norms = {
        name: float(param.grad.detach().norm().cpu()) if name not in nonfinite else float("nan")
        for name, param in critical.items()
    }
    return norms, nonfinite


def _inspect_direct_slat_gradients(
    model: TrainableSymmetricSLatBranch,
    step: int,
) -> tuple[dict[str, float], list[str]]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    missing = [name for name, parameter in trainable if parameter.grad is None]
    if missing:
        raise RuntimeError(
            f"Direct SLat DDP graph break at step={step}; missing gradients for {missing[:16]}"
        )
    nonfinite = [
        name
        for name, parameter in trainable
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    selected = trainable[:8] + trainable[-8:]
    norms = {
        name: (
            float(parameter.grad.detach().float().norm().cpu())
            if name not in nonfinite and parameter.grad is not None
            else float("nan")
        )
        for name, parameter in selected
    }
    return norms, nonfinite


def _assert_fp32_finite_trainable_parameters(module: nn.Module, context: str) -> None:
    wrong_dtype = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.dtype != torch.float32
    ]
    if wrong_dtype:
        raise TypeError(
            f"{context} requires FP32 AdamW master parameters; non-FP32 tensors: {wrong_dtype[:16]}"
        )
    if not _trainable_parameters_are_finite(module):
        raise FloatingPointError(f"{context} contains NaN or Inf parameters")


def _trainable_parameters_are_finite(module: nn.Module) -> bool:
    checks = [
        torch.isfinite(parameter.detach()).all()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return bool(torch.stack(checks).all().item()) if checks else False


def _assert_state_dict_finite(state: dict, context: str) -> None:
    invalid = [
        name
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
        and (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item())
    ]
    if invalid:
        raise FloatingPointError(
            f"{context} is numerically corrupted; non-finite tensors: {invalid[:16]}"
        )


def _distributed_all_true(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(1 if value else 0, device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _warmup_cosine_factor(
    update: int,
    *,
    total_updates: int,
    warmup_updates: int,
    minimum_ratio: float,
) -> float:
    total = max(int(total_updates), 1)
    warmup = min(max(int(warmup_updates), 0), max(total - 1, 0))
    if warmup > 0 and update < warmup:
        return max((int(update) + 1) / warmup, 1.0 / warmup)
    progress = (int(update) - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(minimum_ratio + (1.0 - minimum_ratio) * cosine)


def _set_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    total_updates: int,
    warmup_updates: int,
    minimum_ratio: float,
) -> None:
    factor = _warmup_cosine_factor(
        update,
        total_updates=total_updates,
        warmup_updates=warmup_updates,
        minimum_ratio=minimum_ratio,
    )
    for group in optimizer.param_groups:
        group["lr"] = float(group["initial_lr"]) * factor


def make_synthetic_slat_batch(cfg: dict, args: argparse.Namespace, device: torch.device) -> dict:
    model_cfg = cfg.get("model", {})
    slat_dim = int(model_cfg.get("slat_dim", 8))
    L = int(cfg.get("dry_run_batch", {}).get("active_tokens", args.active_tokens))
    batch = make_dry_run_batch(
        batch_size=args.batch_size,
        num_views=args.num_views,
        image_size=args.image_size,
        latent_tokens=L,
        latent_dim=slat_dim,
        device=device,
    )
    if batch["images"].shape[0] != 1:
        raise ValueError("Symmetric SLat dry-run requires batch_size=1")
    B = batch["images"].shape[0]
    resolution = int(model_cfg.get("resolution", 64))
    indices = torch.randint(20, 44, (B, L, 3), device=device)
    if L >= 3:
        indices[:, 0] = torch.tensor([32, 32, 32], device=device)
        indices[:, 1] = torch.tensor([32, 32, 48], device=device)
        indices[:, 2] = torch.tensor([63, 63, 63], device=device)
    context = build_ss_slat_context(ss_active_indices=indices, resolution=resolution, target_dim=slat_dim)
    x0 = torch.randn(B, L, slat_dim, device=device) * 0.5
    noise = torch.randn_like(x0)
    t = torch.rand(B, device=device)
    sigma_min = float(cfg.get("flow", {}).get("sigma_min", 1e-5))
    x_t = (1 - t.view(B, 1, 1)) * x0 + (sigma_min + (1 - sigma_min) * t.view(B, 1, 1)) * noise
    target_v = (1 - sigma_min) * noise - x0
    batch.update(context)
    synthetic_c2w = torch.eye(4, device=device).reshape(1, 1, 4, 4).expand(
        B, args.num_views, -1, -1
    ).clone()
    synthetic_c2w[..., 2, 3] = -2.0
    batch["c2w"] = synthetic_c2w
    batch["w2c"] = torch.linalg.inv(synthetic_c2w)
    batch.update(
        {
            "slat_latent_tokens": x_t,
            "slat_clean_tokens": x0,
            "slat_raw_tokens": x0,
            "v_slat_base": torch.zeros_like(x_t),
            "target_velocity": target_v,
            "timestep": t,
            "vggt_features": torch.rand(
                B,
                args.num_views,
                1024,
                args.image_size // 4,
                args.image_size // 4,
                device=device,
            ),
            "aligned_depth": batch["depths"],
            "vggt_confidence": torch.ones(
                B, args.num_views, 1, args.image_size, args.image_size, device=device
            ),
            "view_valid_mask": torch.ones(
                B, args.num_views, device=device, dtype=torch.bool
            ),
            "slat_indices": indices,
            "slat_token_valid_mask": torch.ones(B, L, 1, device=device),
            "slat_supervision_mask": torch.ones(B, L, 1, device=device),
            "flow_sigma_min": sigma_min,
        }
    )
    batch["v_slat_base_valid_mask"] = torch.ones(B, L, 1, device=device, dtype=torch.float32)
    batch["v_slat_base_invalid_ratio"] = torch.zeros((), device=device, dtype=torch.float32)
    batch["target_residual"] = batch["target_velocity"].detach()
    return batch


def prepare_batch(
    raw: dict,
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
    *,
    trellis_pipeline=None,
    vggt_geometry: VGGTGeometryWrapper | None = None,
) -> dict:
    batch = {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in raw.items()
    }
    if vggt_geometry is None:
        raise RuntimeError("real_train requires a frozen VGGT geometry extractor; mock VGGT is not permitted.")
    geometry = vggt_geometry.extract(
        batch["images"],
        valid_view_mask=batch.get("view_valid_mask"),
        use_cache=False,
    )
    batch["vggt_features"] = geometry.visual_features
    batch["vggt_depth"] = geometry.depth
    batch["vggt_pointmap"] = geometry.point_map
    batch["vggt_confidence"] = geometry.point_confidence
    batch["vggt_camera"] = {
        "K": geometry.intrinsics,
        "w2c": geometry.extrinsics,
        "c2w": geometry.camera_to_world,
    }
    batch = align_vggt_batch(batch)
    model_cfg = cfg.get("model", {})
    slat_dim = int(model_cfg.get("slat_dim", 8))
    resolution = int(model_cfg.get("resolution", 64))
    if "trellis_slat_feats" in batch and "trellis_slat_indices" in batch:
        feats, indices, token_valid_mask = _pad_latents(
            batch["trellis_slat_feats"], batch["trellis_slat_indices"], args.active_tokens, device
        )
        x0_raw = feats[..., :slat_dim]
        if x0_raw.shape[-1] < slat_dim:
            x0_raw = torch.cat([x0_raw, x0_raw.new_zeros(*x0_raw.shape[:-1], slat_dim - x0_raw.shape[-1])], dim=-1)
        if trellis_pipeline is None:
            raise RuntimeError("Real SLAT training requires a TRELLIS pipeline with its published latent normalization.")
        x0 = normalize_slat(x0_raw, trellis_pipeline.slat_normalization)
        source = "cached_trellis_slat_teacher"
    else:
        raise KeyError("real_train requires trellis_slat_feats and trellis_slat_indices; synthetic SLAT latents are only allowed in --dry_run.")
    context = build_ss_slat_context(ss_active_indices=indices, resolution=resolution, target_dim=slat_dim)
    B = x0.shape[0]
    t = torch.rand(B, device=device)
    noise = torch.randn_like(x0)
    sigma_min = float(cfg.get("flow", {}).get("sigma_min", 1e-5))
    x_t = (1 - t.view(B, 1, 1)) * x0 + (sigma_min + (1 - sigma_min) * t.view(B, 1, 1)) * noise
    batch.update(context)
    batch["slat_latent_tokens"] = x_t
    batch["slat_clean_tokens"] = x0
    batch["slat_raw_tokens"] = x0_raw
    batch["slat_indices"] = indices
    batch["flow_sigma_min"] = sigma_min
    batch["target_velocity"] = (1 - sigma_min) * noise - x0
    batch["slat_token_valid_mask"] = token_valid_mask
    frozen_base = _compute_trellis_slat_base_velocity(
        batch,
        x_t,
        indices,
        token_valid_mask,
        t,
        trellis_pipeline,
        device,
        use_multiview=True,
    )
    frozen_base, base_valid, base_invalid_ratio = _sanitize_slat_base_velocity(
        frozen_base,
        expected_shape=x_t.shape,
        device=device,
        dtype=torch.float32,
    )
    batch["v_slat_base"] = frozen_base
    batch["v_slat_base_valid_mask"] = base_valid
    batch["v_slat_base_invalid_ratio"] = base_invalid_ratio
    batch["target_residual"] = (batch["target_velocity"] - frozen_base).detach()
    has_gt = batch.get("has_gt", batch.get("gt_available", True))
    has_gt = torch.as_tensor(has_gt, device=device, dtype=torch.float32).reshape(-1, 1, 1)
    if has_gt.shape[0] == 1 and B > 1:
        has_gt = has_gt.expand(B, -1, -1)
    batch["slat_supervision_mask"] = (
        has_gt.expand(B, x_t.shape[1], 1)
        * torch.isfinite(batch["target_residual"]).all(dim=-1, keepdim=True).float()
        * token_valid_mask
    )
    batch["timestep"] = t
    batch["slat_target_source"] = source
    return batch


def _sanitize_slat_base_velocity(
    base: torch.Tensor,
    *,
    expected_shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a finite frozen-teacher velocity and a mask of valid teacher tokens.

    Stage 3 trains a residual velocity:
        delta_theta(x_t, t, c) ~= v_target - v_base.
    When the frozen TRELLIS teacher produces NaN/Inf for a sparse token, v_base is
    undefined for that token. The correct objective is to remove the teacher prior
    for that token and train against the direct target velocity, which is exactly
    implemented by setting the invalid teacher contribution to zero and carrying
    a validity mask into the prior-preservation term.
    """
    assert isinstance(base, torch.Tensor), f"v_slat_base must be a tensor, got {type(base)!r}"
    assert len(expected_shape) == 3, f"expected_shape must describe [B,L,C], got {tuple(expected_shape)}"
    B, L, C = expected_shape
    assert base.ndim == 3, f"v_slat_base must be [B,L,C], got {tuple(base.shape)}"
    assert base.shape[0] == B, f"v_slat_base batch {base.shape[0]} != expected {B}"
    assert base.shape[1] >= L, f"v_slat_base tokens {base.shape[1]} < expected {L}"
    assert base.shape[2] >= C, f"v_slat_base channels {base.shape[2]} < expected {C}"
    sliced = base[:, :L, :C].to(device=device, dtype=dtype)
    assert sliced.dtype == torch.float32, f"SLAT base velocity must be float32 after conversion, got {sliced.dtype}"
    valid_mask_bool = torch.isfinite(sliced).all(dim=-1, keepdim=True)
    invalid_ratio = 1.0 - valid_mask_bool.float().mean()
    safe_base = torch.where(valid_mask_bool, torch.nan_to_num(sliced, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(sliced))
    valid_mask = valid_mask_bool.to(dtype=torch.float32)
    assert safe_base.shape == expected_shape, f"sanitized v_slat_base shape {tuple(safe_base.shape)} != {tuple(expected_shape)}"
    assert valid_mask.shape == (B, L, 1), f"v_slat_base_valid_mask shape {tuple(valid_mask.shape)} != {(B, L, 1)}"
    assert torch.isfinite(safe_base).all().item(), "sanitized v_slat_base still contains NaN or Inf."
    assert torch.isfinite(invalid_ratio).all().item(), "v_slat_base_invalid_ratio is non-finite."
    return safe_base, valid_mask, invalid_ratio


def build_real_loader(args: argparse.Namespace, ctx):
    datasets = []
    if args.meshfleet_root and Path(args.meshfleet_root).exists():
        datasets.append(
            MeshFleetTrellisDataset(
                args.meshfleet_root,
                split=args.meshfleet_split,
                category=args.meshfleet_category,
                num_views=args.num_views,
                image_size=args.image_size,
                slat_latent_model=args.meshfleet_slat_latent_model,
                require_slat_latents=True,
                require_voxels=True,
                uid_manifest=args.train_manifest,
            )
        )
    if args.srn_root and Path(args.srn_root).exists():
        datasets.append(SRNCarsDataset(args.srn_root, num_views=args.num_views, image_size=args.image_size))
    if args.objaverse_rendered_root and Path(args.objaverse_rendered_root).exists():
        datasets.append(ObjaverseCarsRenderedDataset(args.objaverse_rendered_root, num_views=args.num_views, image_size=args.image_size))
    if not datasets:
        return None, None
    dataset = VehicleMultiViewDataset(datasets)
    return build_dataloader(dataset, args=args, ctx=ctx, collate_fn=VehicleMultiViewDataset.collate_fn, shuffle=True)


def _padded_slat_state_to_sparse(
    features: torch.Tensor,
    indices: torch.Tensor,
    valid_mask: torch.Tensor,
):
    from affostruction.modules import sparse as sp

    if features.ndim != 3 or indices.shape != (*features.shape[:2], 3):
        raise ValueError("SLat state must be features [B,L,C] and indices [B,L,3]")
    if valid_mask.shape != (*features.shape[:2], 1):
        raise ValueError(
            f"valid_mask must be [B,L,1], got {tuple(valid_mask.shape)}"
        )
    valid = valid_mask[..., 0].bool()
    if not bool(valid.any()):
        raise RuntimeError("SLat state contains no active voxels")
    if not bool(valid.any(dim=1).all()):
        empty = torch.nonzero(~valid.any(dim=1), as_tuple=False).flatten().tolist()
        raise RuntimeError(f"SLat batch objects contain no active voxels: {empty}")
    batch_size, token_count = valid.shape
    batch_column = torch.arange(
        batch_size, device=indices.device, dtype=indices.dtype
    )[:, None, None].expand(batch_size, token_count, 1)
    coords = torch.cat([batch_column, indices], dim=-1)[valid].int().contiguous()
    selected = features[valid]
    return sp.SparseTensor(feats=selected.contiguous(), coords=coords)


def _sparse_slat_velocity_to_padded(
    sparse_velocity,
    target_indices: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    features = getattr(sparse_velocity, "feats", None)
    coords = getattr(sparse_velocity, "coords", None)
    if not isinstance(features, torch.Tensor) or not isinstance(coords, torch.Tensor):
        raise TypeError("TRELLIS SLat flow must return SparseTensor-like feats and coords")
    valid = valid_mask[..., 0].bool()
    return _repad_sparse_prediction(
        features,
        coords,
        target_indices,
        valid,
        dtype=features.dtype,
    )


def _pad_latents(feats, indices, limit: int, device: torch.device):
    if isinstance(feats, list):
        B = len(feats)
        max_length = max(f.shape[0] for f in feats)
        L = min(limit, max_length) if limit > 0 else max_length
        C = feats[0].shape[-1]
        out_feats = torch.zeros(B, L, C, device=device)
        out_idx = torch.zeros(B, L, 3, dtype=torch.long, device=device)
        valid = torch.zeros(B, L, 1, dtype=torch.float32, device=device)
        for b, (f, idx) in enumerate(zip(feats, indices)):
            take = min(L, f.shape[0])
            out_feats[b, :take] = f[:take].to(device, non_blocking=True)
            out_idx[b, :take] = idx[:take].to(device, non_blocking=True)
            valid[b, :take] = 1.0
        return out_feats, out_idx, valid
    length = min(limit, feats.shape[1]) if limit > 0 else feats.shape[1]
    valid = torch.ones(feats.shape[0], length, 1, dtype=torch.float32, device=device)
    return (
        feats[:, :length].to(device, non_blocking=True),
        indices[:, :length].to(device, non_blocking=True),
        valid,
    )


def _pad_indices(indices, limit: int, device: torch.device):
    if isinstance(indices, list):
        B = len(indices)
        max_length = max(max(1, idx.shape[0]) for idx in indices)
        L = min(limit, max_length) if limit > 0 else max_length
        out = torch.zeros(B, L, 3, dtype=torch.long, device=device)
        for b, idx in enumerate(indices):
            take = min(L, idx.shape[0])
            if take:
                out[b, :take] = idx[:take].to(device, non_blocking=True)
        return out
    length = min(limit, indices.shape[1]) if limit > 0 else indices.shape[1]
    return indices[:, :length].to(device, non_blocking=True)


def write_outputs(out_dir: Path, out: dict, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_active_voxels_ply(out_dir / "slat_confidence.ply", out["active_xyz"][0], out["slat_confidence"][0])
    write_active_voxels_ply(out_dir / "ss_active_voxels.ply", out["active_xyz"][0], out["slat_confidence"][0])
    save_slat_debug_npz(
        out_dir / "slat_visibility_debug.npz",
        visibility=out["visibility"],
        view_weights=out["view_weights"],
        slat_confidence=out["slat_confidence"],
    )
    save_slat_debug_npz(
        out_dir / "slat_velocity_debug.npz",
        velocity=out["velocity"],
        v_slat_geo=out["v_slat_geo"],
    )
    (out_dir / "train_geovis_slat_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _save_slat_checkpoint(
    path: Path,
    model,
    optimizer,
    grad_scaler,
    adaptive_grad_clipper,
    step: int,
    cfg: dict,
    early_stopper: EarlyStopper,
    early_status,
    *,
    training_budget: dict | None = None,
    support_provenance: dict | None = None,
) -> None:
    _assert_fp32_finite_trainable_parameters(
        unwrap_model(model), f"SLat checkpoint step {step}"
    )
    save_checkpoint(
        path,
        architecture_version=AffostructionSLatFlow.architecture_version,
        optimization_numerics_version=SLAT_OPTIMIZATION_NUMERICS_VERSION,
        model=unwrap_model(model).state_dict(),
        optimizer=optimizer.state_dict(),
        grad_scaler=grad_scaler.state_dict(),
        adaptive_grad_clipper=adaptive_grad_clipper.state_dict(),
        step=step,
        config=cfg,
        tensor_contract={
            "version": SLAT_TENSOR_CONTRACT_VERSION,
            "flow_state": "trellis_normalized_slat",
            "decoder_state": "trellis_raw_vae_slat",
            "control_variables": ["pixel_aligned_condition", "confidence", "occlusion", "angle_depth_agreement"],
            "teacher_velocity": "frozen_native_trellis_multiview_cfg",
            "sampler_correction": "confidence_gated_bounded_sparse_residual_flow",
            "decoded_supervision": bool(cfg.get("decoded_supervision", {}).get("enabled", False)),
        },
        early_stop=early_status.as_dict() if early_status is not None else None,
        early_stopper=early_stopper.state_dict(),
        training_budget=training_budget,
        support_provenance=support_provenance,
    )


def _make_grad_scaler(*, enabled: bool):
    """Use the current AMP API while retaining support for older torch builds."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _adaptive_clipper(trellis_root: str, max_norm: float):
    root = str(Path(trellis_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from trellis.utils.grad_clip_utils import AdaptiveGradClipper

    return AdaptiveGradClipper(
        max_norm=float(max_norm),
        clip_percentile=95.0,
        buffer_size=1000,
    )


def _validate_checkpoint_tensor_contract(state: dict, path: str | Path) -> None:
    contract = state.get("tensor_contract") if isinstance(state, dict) else None
    version = contract.get("version") if isinstance(contract, dict) else None
    if version != SLAT_TENSOR_CONTRACT_VERSION:
        raise RuntimeError(
            f"SLAT checkpoint {path} has tensor contract {version!r}; "
            f"expected {SLAT_TENSOR_CONTRACT_VERSION!r}. Retrain instead of partially loading an incompatible model."
        )


def _validate_affostruction_slat_checkpoint(state: dict, path: str | Path) -> None:
    version = state.get("architecture_version") if isinstance(state, dict) else None
    if version != AffostructionSLatFlow.architecture_version:
        raise RuntimeError(
            f"SLat checkpoint {path} belongs to {version!r}; expected "
            f"{AffostructionSLatFlow.architecture_version!r}. Full-backbone and legacy adapter checkpoints are incompatible with the decoupled sparse image-flow graph."
        )
    numerics_version = state.get("optimization_numerics_version")
    if numerics_version != SLAT_OPTIMIZATION_NUMERICS_VERSION:
        raise RuntimeError(
            f"SLat checkpoint {path} predates FP32-master optimization; got "
            f"{numerics_version!r}, expected {SLAT_OPTIMIZATION_NUMERICS_VERSION!r}. "
            "Restart from original TRELLIS weights."
        )


def _validate_checkpoint_model_config(
    state: dict,
    model_cfg: dict,
    *,
    context: str,
    allow_runtime_control_overrides: bool = False,
) -> dict:
    checkpoint_cfg = state.get("config", {}).get("model", {}) if isinstance(state, dict) else {}
    # These scalars do not alter parameter names or shapes. Stage 4 deliberately
    # changes them as part of decoded-asset fine-tuning, so a weights-only
    # initialization may override them. A true resume still requires an exact
    # match to prevent silently changing an interrupted experiment.
    runtime_control_keys = {"confidence_floor", "trust_region"}
    accepted_overrides = {}
    keys = (
        "slat_dim", "resolution", "evidence_dim", "hidden_dim", "feature_dim", "num_heads",
        "fusion_mode", "trust_region", "confidence_floor", "beta_mode", "beta_strength",
        "factorized_control", "use_geovis_slat",
    )
    for key in keys:
        if key in checkpoint_cfg and key in model_cfg and checkpoint_cfg[key] != model_cfg[key]:
            if allow_runtime_control_overrides and key in runtime_control_keys:
                accepted_overrides[key] = {
                    "checkpoint": checkpoint_cfg[key],
                    "current": model_cfg[key],
                }
                continue
            raise RuntimeError(
                f"{context} model mismatch for {key}: checkpoint={checkpoint_cfg[key]!r}, "
                f"config={model_cfg[key]!r}"
            )
    return accepted_overrides


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument("--steps", type=int, default=None, help="Required for real training; dry runs do not consume an update budget.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--active_tokens", type=int, default=0, help="Maximum SLAT tokens per object; 0 keeps every active voxel.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--conditioner_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_learning_rate_ratio", type=float, default=0.1)
    parser.add_argument("--velocity_weight", type=float, default=1e-3)
    parser.add_argument("--prior_weight", type=float, default=1e-2)
    parser.add_argument("--raw_residual_weight", type=float, default=1.0)
    parser.add_argument("--effective_residual_weight", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument(
        "--archive_every",
        type=int,
        default=0,
        help="Keep a numbered full checkpoint at this cadence; 0 keeps only rolling last/best files.",
    )
    parser.add_argument("--visualize_every", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Weights-only initialization; optimizer and step restart from zero.")
    parser.add_argument("--meshfleet_root", type=str, default=None)
    parser.add_argument("--meshfleet_split", type=str, default="train")
    parser.add_argument("--train_manifest", type=str, default=None, help="UID manifest generated by inspect_meshfleet_dataset.py.")
    parser.add_argument("--meshfleet_category", type=str, default=None)
    parser.add_argument("--meshfleet_slat_latent_model", type=str, default="dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--srn_root", type=str, default=None)
    parser.add_argument("--objaverse_rendered_root", type=str, default=None)
    parser.add_argument("--trellis_root", type=str, default=None)
    parser.add_argument(
        "--affostruction_root",
        type=str,
        default="/mnt/sda/hf/MVG/Base/Affostruction",
    )
    parser.add_argument("--trellis_model_path", type=str, default=None)
    parser.add_argument("--vggt_root", type=str, default=None)
    parser.add_argument("--vggt_checkpoint", type=str, default=None)
    parser.add_argument("--vggt_pretrained", type=str, default=None)
    parser.add_argument("--torch_hub_dir", type=str, default=None)
    parser.add_argument("--dinov2_repo", type=str, default=None)
    parser.add_argument(
        "--hf_cache_root",
        type=str,
        default="/mnt/sda/hf/.cache/huggingface/hub",
    )
    parser.add_argument("--real_train", action="store_true")
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16"), default="fp16")
    parser.add_argument("--fused_optimizer", type=str2bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    add_adaptive_batch_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    _apply_config_defaults(args, cfg, parser)
    try:
        summary = run_dry_run(cfg, args) if args.dry_run else run_training(cfg, args)
        if getattr(args, "rank", 0) == 0:
            print(json.dumps(summary, indent=2))
    finally:
        cleanup_distributed()


def _load_trellis_pipeline(
    args: argparse.Namespace,
    device: torch.device,
    ctx,
    *,
    required_models: list[str],
):
    if args.trellis_root:
        sys.path.insert(0, args.trellis_root)
    if not args.trellis_model_path:
        raise FileNotFoundError("real_train requires --trellis_model_path or trellis.pipeline in config.")
    configure_trellis_hub(args)
    args.trellis_model_path = resolve_local_hf_snapshot(
        args.trellis_model_path,
        args.hf_cache_root,
        required_file="pipeline.json",
        validate_trellis_pipeline=True,
    )

    if ctx.distributed and not ctx.is_main:
        dist.barrier()
    pipeline = _load_trellis_pipeline_impl(args, device, required_models=required_models)
    if ctx.distributed and ctx.is_main:
        dist.barrier()
    return pipeline


def _load_vggt_geometry(args: argparse.Namespace, device: torch.device, ctx) -> VGGTGeometryWrapper:
    """Load one frozen real VGGT replica per DDP rank before training starts."""
    pretrained_source = args.vggt_pretrained
    if args.vggt_checkpoint is None:
        pretrained_source = resolve_local_hf_snapshot(
            args.vggt_pretrained,
            args.hf_cache_root,
            required_file="model.safetensors",
        )
    if ctx.distributed and not ctx.is_main:
        dist.barrier()
    geometry = VGGTGeometryWrapper(
        vggt_root=args.vggt_root,
        checkpoint=args.vggt_checkpoint,
        pretrained_name=pretrained_source,
        mock=False,
    ).to(device)
    geometry.eval()
    if ctx.distributed and ctx.is_main:
        dist.barrier()
    return geometry


def _load_trellis_pipeline_impl(
    args: argparse.Namespace,
    device: torch.device,
    *,
    required_models: list[str],
):
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.trellis_model_path)
    pipeline.training_residency = configure_trellis_training_residency(
        pipeline,
        required_models=required_models,
        device=device,
    )
    return pipeline


@torch.no_grad()
def _compute_trellis_slat_base_velocity(
    batch: dict,
    x_t: torch.Tensor,
    indices: torch.Tensor,
    token_valid_mask: torch.Tensor,
    t: torch.Tensor,
    pipeline,
    device: torch.device,
    *,
    use_multiview: bool = True,
) -> torch.Tensor:
    if pipeline is None:
        raise KeyError("real_train requires trellis_slat_base_velocity or a real TRELLIS pipeline to compute frozen SLAT base velocity.")
    from trellis.modules import sparse as sp

    B, L, C = x_t.shape
    if token_valid_mask.shape != (B, L, 1):
        raise ValueError(
            f"token_valid_mask must be [B,L,1], got {tuple(token_valid_mask.shape)} for {tuple(x_t.shape)}"
        )
    valid = token_valid_mask.squeeze(-1).bool()
    if not bool(valid.any(dim=1).all()):
        empty = torch.nonzero(~valid.any(dim=1), as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Objects have no valid SLAT tokens: {empty}")
    batch_column = torch.arange(B, device=device, dtype=indices.dtype)[:, None, None].expand(B, L, 1)
    coords = torch.cat([batch_column, indices], dim=-1)[valid].int().contiguous()
    sparse = sp.SparseTensor(feats=x_t[valid].contiguous(), coords=coords)
    conditions = _real_slat_conditions(batch, device, pipeline)
    if not use_multiview:
        conditions = conditions[:1]
    predictions = [pipeline.models["slat_flow_model"](sparse, t * 1000.0, cond) for cond in conditions]
    if not predictions or any(not hasattr(item, "feats") for item in predictions):
        raise TypeError("TRELLIS slat_flow_model must return SparseTensor predictions with feats.")
    if any(not torch.equal(item.coords, predictions[0].coords) for item in predictions[1:]):
        raise RuntimeError("TRELLIS multi-view teacher returned inconsistent sparse coordinate ordering.")
    # Match the positive branch of inference-time MultiDiffusion: every view
    # predicts a velocity for the same sparse state, then velocities are averaged.
    stacked_predictions = torch.stack([item.feats for item in predictions], dim=0)
    if use_multiview:
        view_valid = batch.get("view_valid_mask")
        if view_valid is None:
            view_valid = torch.ones(B, len(predictions), device=device, dtype=torch.bool)
        view_valid = view_valid[:, : len(predictions)].to(device=device, dtype=stacked_predictions.dtype)
        row_batch = predictions[0].coords[:, 0].long()
        row_weights = view_valid.transpose(0, 1).index_select(1, row_batch).unsqueeze(-1)
        positive_feats = (stacked_predictions * row_weights).sum(dim=0) / row_weights.sum(dim=0).clamp_min(1.0)
    else:
        positive_feats = stacked_predictions[0]
    cfg_strength, cfg_interval = _trellis_slat_cfg_parameters(pipeline)
    if cfg_strength != 0.0:
        negative = pipeline.models["slat_flow_model"](
            sparse, t * 1000.0, torch.zeros_like(conditions[0])
        )
        if not hasattr(negative, "feats") or not torch.equal(negative.coords, predictions[0].coords):
            raise RuntimeError("TRELLIS negative CFG teacher changed the sparse coordinate contract.")
        cfg_active = ((t >= cfg_interval[0]) & (t <= cfg_interval[1])).to(positive_feats.dtype)
        per_row_strength = cfg_active[predictions[0].coords[:, 0].long()].unsqueeze(-1) * cfg_strength
        feats = positive_feats + per_row_strength * (positive_feats - negative.feats)
    else:
        feats = positive_feats
    if feats.ndim != 2:
        raise ValueError(f"TRELLIS slat_flow_model feats must be [B*L,C], got {tuple(feats.shape)}")
    if feats.shape[0] != int(valid.sum()):
        raise ValueError(
            f"TRELLIS slat_flow_model returned {feats.shape[0]} tokens, expected {int(valid.sum())} valid tokens."
        )
    if feats.shape[1] < C:
        raise ValueError(f"TRELLIS slat_flow_model returned {feats.shape[1]} channels, expected at least {C}.")
    return _repad_sparse_prediction(
        feats[:, :C], predictions[0].coords, indices, valid, dtype=x_t.dtype
    ).detach()


def _trellis_slat_cfg_parameters(pipeline) -> tuple[float, tuple[float, float]]:
    """Read the exact CFG defaults/overrides used by TRELLIS' configured sampler."""
    parameters = inspect.signature(pipeline.slat_sampler.sample).parameters
    if "cfg_strength" not in parameters:
        return 0.0, (0.0, 1.0)
    overrides = dict(getattr(pipeline, "slat_sampler_params", {}) or {})
    strength = float(overrides.get("cfg_strength", parameters["cfg_strength"].default))
    interval_default = parameters["cfg_interval"].default if "cfg_interval" in parameters else (0.0, 1.0)
    interval = overrides.get("cfg_interval", interval_default)
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError(f"TRELLIS SLAT cfg_interval must have two values, got {interval!r}")
    interval_pair = (float(interval[0]), float(interval[1]))
    if strength <= -1.0 or not 0.0 <= interval_pair[0] <= interval_pair[1] <= 1.0:
        raise ValueError(f"Invalid TRELLIS SLAT CFG contract: strength={strength}, interval={interval_pair}")
    return strength, interval_pair


def _repad_sparse_prediction(
    feats: torch.Tensor,
    coords: torch.Tensor,
    target_indices: torch.Tensor,
    valid: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Repad teacher output by exact coordinate identity, even if sparse ops reorder rows."""
    B, L = valid.shape
    if coords.ndim != 2 or coords.shape[1] != 4 or feats.shape[0] != coords.shape[0]:
        raise ValueError("Sparse prediction requires feats [N,C] and coords [N,4]")
    batch_column = torch.arange(
        B, device=target_indices.device, dtype=target_indices.dtype
    )[:, None, None].expand(B, L, 1)
    target_coords = torch.cat([batch_column, target_indices], dim=-1)[valid].long()
    if target_coords.shape[0] != coords.shape[0]:
        raise RuntimeError(
            "TRELLIS flow changed sparse token count: "
            f"input={target_coords.shape[0]}, output={coords.shape[0]}"
        )
    maximum_coordinate = torch.cat([coords[:, 1:].long(), target_coords[:, 1:]], dim=0).amax()
    resolution = max(1, int(maximum_coordinate.item()) + 1)
    source_keys = _batched_coordinate_keys(coords.long(), resolution)
    target_keys = _batched_coordinate_keys(target_coords, resolution)
    sorted_keys, order = source_keys.sort()
    lookup = torch.searchsorted(sorted_keys, target_keys)
    if bool((lookup >= sorted_keys.numel()).any()) or not torch.equal(
        sorted_keys[lookup], target_keys
    ):
        raise RuntimeError("TRELLIS flow changed active voxel coordinates")
    out = feats.new_zeros(B, L, feats.shape[-1], dtype=dtype)
    out[valid] = feats[order[lookup]].to(dtype=dtype)
    return out


def _batched_coordinate_keys(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    coords = coords.long()
    return (
        ((coords[:, 0] * resolution + coords[:, 1]) * resolution + coords[:, 2])
        * resolution
        + coords[:, 3]
    )


@torch.no_grad()
def _real_slat_conditions(batch: dict, device: torch.device, pipeline) -> list[torch.Tensor]:
    for key in ("trellis_cond", "trellis_cond_tokens", "image_cond", "cond"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return [value.to(device=device, dtype=torch.float32)]
    images = batch.get("images")
    if isinstance(images, torch.Tensor) and images.ndim == 5:
        return [
            pipeline.encode_image(
                F.interpolate(
                    images[:, view].to(device=device, dtype=torch.float32),
                    size=(518, 518), mode="bicubic", align_corners=False, antialias=True,
                )
            )
            for view in range(images.shape[1])
        ]
    cond_image = batch.get("trellis_cond_image")
    if isinstance(cond_image, torch.Tensor):
        return [pipeline.encode_image(cond_image.to(device=device, dtype=torch.float32))]
    images = batch.get("images")
    if isinstance(images, torch.Tensor):
        first_view = F.interpolate(images[:, 0].to(device=device, dtype=torch.float32), size=(518, 518), mode="bilinear", align_corners=False)
        return [pipeline.encode_image(first_view)]
    raise KeyError("real_train requires TRELLIS condition image/tokens to compute frozen SLAT base velocity.")


def _apply_config_defaults(args: argparse.Namespace, cfg: dict, parser: argparse.ArgumentParser) -> None:
    if not cfg:
        return
    dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
    trellis = cfg.get("trellis") if isinstance(cfg.get("trellis"), dict) else {}
    vggt = cfg.get("vggt") if isinstance(cfg.get("vggt"), dict) else {}
    affostruction = (
        cfg.get("affostruction")
        if isinstance(cfg.get("affostruction"), dict)
        else {}
    )
    mappings = {
        "meshfleet_root": cfg.get("meshfleet_root") or cfg.get("dataset_root") or dataset.get("root"),
        "meshfleet_split": cfg.get("meshfleet_split") or dataset.get("train_split") or dataset.get("split"),
        "meshfleet_category": cfg.get("meshfleet_category") or dataset.get("category"),
        "num_views": cfg.get("num_views") or dataset.get("num_views"),
        "image_size": cfg.get("image_size") or dataset.get("image_size"),
        "trellis_root": cfg.get("trellis_root") or trellis.get("root"),
        "affostruction_root": cfg.get("affostruction_root") or affostruction.get("root"),
        "trellis_model_path": cfg.get("trellis_model_path") or cfg.get("trellis_pipeline") or cfg.get("trellis_checkpoint") or trellis.get("model_path") or trellis.get("pipeline") or trellis.get("checkpoint"),
        "vggt_root": cfg.get("vggt_root") or vggt.get("root"),
        "vggt_checkpoint": cfg.get("vggt_checkpoint") or vggt.get("checkpoint"),
        "vggt_pretrained": cfg.get("vggt_pretrained") or vggt.get("pretrained"),
        "steps": cfg.get("steps"),
        "steps_are_total": cfg.get("steps_are_total"),
        "minimum_dataset_passes": cfg.get("minimum_dataset_passes"),
        "early_stop": cfg.get("early_stop"),
        "early_stop_metric": cfg.get("early_stop_metric"),
        "early_stop_mode": cfg.get("early_stop_mode"),
        "max_train_hours": cfg.get("max_train_hours"),
        "save_best": cfg.get("save_best"),
        "batch_size": cfg.get("batch_size"),
        "lr": cfg.get("lr"),
        "conditioner_lr": cfg.get("conditioner_lr"),
        "weight_decay": cfg.get("weight_decay"),
        "adam_epsilon": cfg.get("adam_epsilon"),
        "warmup_steps": cfg.get("warmup_steps"),
        "min_learning_rate_ratio": cfg.get("min_learning_rate_ratio"),
        "velocity_weight": cfg.get("velocity_weight"),
        "prior_weight": cfg.get("prior_weight"),
        "raw_residual_weight": cfg.get("raw_residual_weight"),
        "effective_residual_weight": cfg.get("effective_residual_weight"),
        "grad_accum_steps": cfg.get("grad_accum_steps"),
        "train_manifest": cfg.get("train_manifest") or dataset.get("train_manifest"),
        "save_every": cfg.get("save_every"),
        "archive_every": cfg.get("archive_every"),
        "output_dir": cfg.get("output_dir"),
        "device": cfg.get("device"),
        "amp": cfg.get("amp"),
        "amp_dtype": cfg.get("amp_dtype"),
        "fused_optimizer": cfg.get("fused_optimizer"),
        "max_grad_norm": cfg.get("max_grad_norm"),
        "gradient_checkpointing": cfg.get("gradient_checkpointing"),
        "num_workers": cfg.get("num_workers") or dataset.get("num_workers"),
        "pin_memory": cfg.get("pin_memory") if cfg.get("pin_memory") is not None else dataset.get("pin_memory"),
        "prefetch_factor": cfg.get("prefetch_factor") or dataset.get("prefetch_factor"),
        "torch_hub_dir": cfg.get("torch_hub_dir") or trellis.get("torch_hub_dir"),
        "dinov2_repo": cfg.get("dinov2_repo") or trellis.get("dinov2_repo"),
        "hf_cache_root": cfg.get("hf_cache_root") or trellis.get("hf_cache_root"),
        **adaptive_config_defaults(cfg),
    }
    apply_config_mappings(args, parser, mappings)


if __name__ == "__main__":
    main()
