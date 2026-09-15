#!/usr/bin/env python3
"""Production DDP trainer for stochastic multi-view, voxel-conditioned SS Flow."""

from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geoss.datasets.dataset_stochastic_meshfleet import (  # noqa: E402
    StochasticMeshFleetDataset,
    seed_meshfleet_worker,
    stochastic_meshfleet_collate,
)
from geoss.integration.trellis_hub import (  # noqa: E402
    configure_trellis_hub,
    resolve_local_hf_snapshot,
)
from geoss.integration.trellis_ss_hook import ss_grid_to_tokens  # noqa: E402
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryWrapper  # noqa: E402
from geoss.losses.geometric_loss import FlowMatchingLossBuilder  # noqa: E402
from geoss.models.affostruction_conditioning import extract_dinov2_spatial_features  # noqa: E402
from geoss.models.ss_flow_adapter import AffostructionSSFlow  # noqa: E402
from geoss.models.voxel_fusion_engine import ConfidenceSparseVoxelFusion  # noqa: E402
from geoss.ops.flow_matching import construct_flow_training_pair  # noqa: E402


SS_OPTIMIZATION_NUMERICS_VERSION = "fp32_master_bf16_forward_v2"


class TrainableVoxelSSBranch(nn.Module):
    """DDP branch containing voxel fusion and the complete trainable SS flow."""

    def __init__(self, fusion: ConfidenceSparseVoxelFusion, flow: AffostructionSSFlow) -> None:
        super().__init__()
        self.fusion = fusion
        self.flow = flow

    def forward(
        self,
        geometry,
        batch: Dict[str, torch.Tensor],
        x_t_grid: torch.Tensor,
        timestep_model: torch.Tensor,
        fallback_condition: torch.Tensor,
        dino_spatial_features: torch.Tensor,
        *,
        profile: bool = False,
    ):
        fusion_timer = CudaStageTimer(x_t_grid.device, profile)
        fusion_timer.start()
        fused = self.fusion(
            geometry,
            foreground_masks=batch["masks"],
            dataset_K=batch.get("K_dataset"),
            dataset_c2w=batch["c2w_dataset"],
            canonical_center=batch["canonical_center"],
            canonical_half_extent=batch["canonical_half_extent"],
            profile=profile,
            spatial_features=dino_spatial_features,
        )
        # Native TRELLIS receives the real DINO image condition when no ray
        # intersects the canonical volume.  Keep the (necessarily inactive)
        # fusion parameters in the same DDP graph so a rare fallback sample on
        # one rank cannot desynchronize gradient collectives on other ranks.
        fallback_condition = fallback_condition + parameter_graph_zero(
            self.fusion, fallback_condition
        )
        fusion_ms = fusion_timer.stop()
        flow_timer = CudaStageTimer(x_t_grid.device, profile)
        flow_timer.start()
        prediction = self.flow(
            x_t_grid,
            timestep_model,
            fused.dense_tokens,
            fused.observation_mask,
            fallback_condition=fallback_condition,
        )
        flow_ms = flow_timer.stop()
        return fused, prediction, {
            "voxel_fusion": fusion_ms,
            **fused.timings_ms,
            "ss_backbone_forward": flow_ms,
        }


def parameter_graph_zero(module: nn.Module, reference: torch.Tensor) -> torch.Tensor:
    """Return scalar zero whose autograd graph touches all trainable parameters."""
    graph_zero = reference.new_zeros(())
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.numel() > 0:
            graph_zero = graph_zero + parameter.reshape(-1)[0] * 0.0
    return graph_zero


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler with an explicit rank-local cursor for exact resume."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_index = 0

    @property
    def full_length(self) -> int:
        return super().__len__()

    def set_start_index(self, start_index: int) -> None:
        start_index = int(start_index)
        if not 0 <= start_index <= self.full_length:
            raise ValueError(
                f"Sampler cursor {start_index} lies outside [0,{self.full_length}]"
            )
        self.start_index = start_index

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.start_index, None)

    def __len__(self) -> int:
        return max(self.full_length - self.start_index, 0)


def main() -> None:
    parser = build_parser()
    apply_yaml_defaults(parser)
    args = parser.parse_args()
    context = init_distributed(args)
    try:
        run_training(args, context)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_training(args: argparse.Namespace, context: Dict[str, Any]) -> None:
    rank, world_size, local_rank, device = (
        context["rank"], context["world_size"], context["local_rank"], context["device"]
    )
    seed_everything(args.seed, rank)
    if args.batch_size != 1:
        raise ValueError(
            "VGGT has no padded-view attention mask; production stochastic 1-8 view training therefore requires --batch-size 1 per rank."
        )
    precision = resolve_precision(args.precision, device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(json.dumps({"event": "startup", "world_size": world_size, "precision": str(precision), "args": vars(args)}))

    dataset = StochasticMeshFleetDataset(
        args.dataset_root,
        split=args.train_split,
        min_views=args.min_views,
        max_views=args.max_views,
        seed=args.seed,
        rank=rank,
        use_stochastic_views=args.use_stochastic_views,
        image_size=args.image_size,
        load_gt_occupancy=args.use_occupancy_loss,
        occupancy_resolution=64,
        meshfleet_root=args.meshfleet_root,
        require_ss_latents=True,
    )
    sampler = ResumableDistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        worker_init_fn=seed_meshfleet_worker,
        collate_fn=stochastic_meshfleet_collate,
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    if len(loader) == 0:
        raise RuntimeError("Distributed MeshFleet loader is empty")
    full_batches_per_epoch = len(loader)
    validation_loader = build_validation_loader(args, rank, world_size, device)

    vggt, trellis_pipeline, base_flow, decoder = load_frozen_foundations(args, device)
    fusion = ConfidenceSparseVoxelFusion(
        grid_resolution=args.grid_resolution,
        input_feature_dim=1024,
        projected_feature_dim=1024,
        positional_frequencies=4,
        fusion_mode=args.fusion_mode,
        use_vggt_depth=args.use_vggt_depth,
        use_vggt_pointmap=args.use_vggt_pointmap,
        use_confidence_weighting=args.use_confidence_weighting,
        use_visibility_weighting=args.use_visibility_weighting,
        use_3d_positional_encoding=args.use_3d_positional_encoding,
        require_spconv=True,
        use_spconv_refinement=args.use_spconv_refinement,
        conditioning_layout="affostruction",
    ).to(device)
    flow = AffostructionSSFlow(
        base_flow,
        condition_dim=fusion.condition_dim,
        classifier_free_dropout=args.classifier_free_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    trainable: nn.Module = TrainableVoxelSSBranch(fusion, flow).to(
        device=device, dtype=torch.float32
    )
    assert_fp32_finite_trainable_parameters(trainable, "SS initialization")
    optimizer = torch.optim.AdamW(
        trainable.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update: warmup_cosine_factor(
            update,
            total_updates=args.max_steps,
            warmup_updates=args.warmup_steps,
            minimum_ratio=args.min_learning_rate / args.learning_rate,
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == torch.float16)
    start_step, epoch, batches_seen_in_epoch = maybe_resume(
        args.resume,
        trainable,
        optimizer,
        scheduler,
        scaler,
        device,
        rank,
        batches_per_epoch=full_batches_per_epoch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    if batches_seen_in_epoch >= full_batches_per_epoch:
        epoch += batches_seen_in_epoch // full_batches_per_epoch
        batches_seen_in_epoch %= full_batches_per_epoch
    if world_size > 1:
        trainable = DDP(
            trainable,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    loss_builder = FlowMatchingLossBuilder(
        lambda_cfm=args.lambda_cfm,
        lambda_depth=args.lambda_depth,
        lambda_silhouette=args.lambda_silhouette,
        lambda_occupancy=args.lambda_occupancy,
        lambda_surface=args.lambda_surface,
        lambda_prior=0.0,
        use_depth_loss=args.use_depth_loss,
        use_silhouette_loss=args.use_silhouette_loss,
        use_surface_loss=args.use_surface_loss,
        use_prior_preservation=False,
        surface_backend=args.surface_backend,
        extensions_root=args.extensions_root,
        render_resolution=args.loss_render_resolution,
        ray_samples=args.loss_ray_samples,
    ).to(device)
    report_parameter_counts(rank, trainable, (vggt, trellis_pipeline))

    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    csv_path = Path(args.output_dir) / "metrics.csv"
    sampler.set_epoch(epoch)
    sampler.set_start_index(batches_seen_in_epoch)
    dataset.set_epoch(epoch)
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.max_steps + 1):
        step_wall = time.perf_counter()
        accumulated_loss = torch.zeros((), device=device)
        last_payload: Dict[str, Any] = {}
        dataloader_seconds = 0.0
        profile_timings: Dict[str, Any] = {}
        step_healthy = True
        optimizer_skip_reason = ""
        for accumulation_index in range(args.gradient_accumulation_steps):
            fetch_start = time.perf_counter()
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                sampler.set_start_index(0)
                dataset.set_epoch(epoch)
                batches_seen_in_epoch = 0
                iterator = iter(loader)
                cpu_batch = next(iterator)
            batches_seen_in_epoch += 1
            dataloader_seconds += time.perf_counter() - fetch_start
            h2d_timer = CudaStageTimer(device, args.profile)
            h2d_timer.start()
            batch = move_batch(cpu_batch, device)
            profile_timings["h2d"] = h2d_timer.stop()

            sync_context = (
                trainable.no_sync()
                if isinstance(trainable, DDP) and accumulation_index < args.gradient_accumulation_steps - 1
                else contextlib.nullcontext()
            )
            with sync_context:
                payload = forward_training_batch(
                    args,
                    batch,
                    vggt,
                    trellis_pipeline,
                    base_flow,
                    decoder,
                    trainable,
                    loss_builder,
                    precision,
                    device,
                )
                loss = payload["losses"]["loss_total"] / args.gradient_accumulation_steps
                local_loss_finite = bool(torch.isfinite(loss).item())
                local_loss_requires_grad = bool(loss.requires_grad)
                loss_finite_on_all_ranks = distributed_all_true(
                    local_loss_finite, device
                )
                loss_graph_on_all_ranks = distributed_all_true(
                    local_loss_requires_grad, device
                )
                if not loss_finite_on_all_ranks:
                    step_healthy = False
                    optimizer_skip_reason = "nonfinite_loss_on_at_least_one_rank"
                elif not loss_graph_on_all_ranks:
                    step_healthy = False
                    optimizer_skip_reason = "graphless_loss_on_at_least_one_rank"
                payload["loss_finite"] = local_loss_finite
                payload["loss_requires_grad"] = local_loss_requires_grad
                # Persist the exact sample state before backward.  If autograd
                # itself raises, the responsible UID/rank/view set is retained.
                append_pathology_record(
                    Path(args.output_dir) / f"pathologies_rank{rank}.jsonl",
                    step=step,
                    accumulation_index=accumulation_index,
                    rank=rank,
                    batch=batch,
                    payload=payload,
                    loss_finite=local_loss_finite,
                    loss_requires_grad=local_loss_requires_grad,
                )
                if step_healthy:
                    scaler.scale(loss).backward()
            accumulated_loss += torch.nan_to_num(
                loss.detach(), nan=0.0, posinf=0.0, neginf=0.0
            )
            last_payload = payload
            profile_timings.update(payload["timings"])

        backward_timer = CudaStageTimer(device, args.profile)
        backward_timer.start()
        optimizer_step_applied = False
        grad_norm = torch.zeros((), device=device)
        if step_healthy:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable.parameters(), args.gradient_clip_norm
            )
            gradients_finite_on_all_ranks = distributed_all_true(
                bool(torch.isfinite(grad_norm).item()), device
            )
            if gradients_finite_on_all_ranks:
                scaler.step(optimizer)
                scaler.update()
                parameters_finite = distributed_all_true(
                    trainable_parameters_are_finite(trainable), device
                )
                if not parameters_finite:
                    raise FloatingPointError(
                        "AdamW produced a non-finite SS parameter update; the last saved checkpoint remains the recovery boundary"
                    )
                scheduler.step()
                optimizer_step_applied = True
            else:
                optimizer_skip_reason = "nonfinite_gradient_on_at_least_one_rank"
                if scaler.is_enabled():
                    scaler.update(new_scale=max(float(scaler.get_scale()) * 0.5, 1.0))
        optimizer.zero_grad(set_to_none=True)
        profile_timings["optimizer"] = backward_timer.stop()

        reduced_loss = distributed_mean(accumulated_loss)
        elapsed = time.perf_counter() - step_wall
        metrics = build_metrics(
            step,
            epoch,
            reduced_loss,
            last_payload,
            profile_timings,
            dataloader_seconds,
            elapsed,
            grad_norm,
            optimizer,
            batch,
            world_size,
            device,
            optimizer_step_applied,
            optimizer_skip_reason,
        )
        if rank == 0:
            append_metrics(metrics_path, csv_path, metrics)
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(metrics, default=json_default))
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(
                Path(args.output_dir) / "stochastic_voxel_ss_flow_last.pt",
                trainable,
                optimizer,
                scheduler,
                scaler,
                step,
                epoch,
                args,
                rank,
                batches_seen_in_epoch,
            )
        if validation_loader is not None and step % args.validate_every == 0:
            validation_metrics = run_validation(
                args,
                validation_loader,
                vggt,
                trellis_pipeline,
                base_flow,
                decoder,
                trainable,
                loss_builder,
                precision,
                device,
                rank,
                step,
            )
            if rank == 0:
                validation_path = Path(args.output_dir) / "validation.jsonl"
                with validation_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(validation_metrics, default=json_default) + "\n")
                print(json.dumps(validation_metrics, default=json_default))
        if world_size > 1 and step % args.barrier_every == 0:
            dist.barrier()


def forward_training_batch(
    args,
    batch,
    vggt,
    pipeline,
    base_flow,
    decoder,
    trainable,
    loss_builder,
    precision,
    device,
) -> Dict[str, Any]:
    vggt_timer = CudaStageTimer(device, args.profile)
    vggt_timer.start()
    geometry = vggt.extract(batch["images"], valid_view_mask=batch["view_valid_mask"])
    vggt_ms = vggt_timer.stop()
    raw_x0 = batch["ss_latent_grid"].float()
    latent_valid = torch.isfinite(raw_x0).flatten(1).all(dim=1)
    x0 = torch.nan_to_num(raw_x0, nan=0.0, posinf=0.0, neginf=0.0)
    noise = torch.randn_like(x0)
    timestep = sample_timestep(x0.shape[0], device, args)
    x_t, v_target_grid, flow_backend = construct_flow_training_pair(
        x0, noise, timestep, args.sigma_min, backend=args.flow_backend
    )
    trellis_timer = CudaStageTimer(device, args.profile)
    trellis_timer.start()
    with torch.inference_mode():
        fallback_condition, condition_valid = encode_real_trellis_condition(
            pipeline, batch["images"], precision, return_validity=True
        )
        dino_spatial_features = extract_dinov2_spatial_features(
            pipeline.models["image_cond_model"],
            pipeline.image_cond_model_transform,
            batch["images"],
            image_size=518,
            amp_dtype=precision,
        )
        timestep_model = timestep * 1000.0
    trellis_ms = trellis_timer.stop()
    target_tokens = ss_grid_to_tokens(v_target_grid)
    with torch.autocast(device_type=device.type, dtype=precision, enabled=device.type == "cuda"):
        try:
            fused, prediction, branch_timings = trainable(
                geometry,
                batch,
                x_t,
                timestep_model,
                fallback_condition,
                dino_spatial_features,
                profile=args.profile,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"SS voxel fusion failed for uid={batch.get('uid')}, "
                f"view_ids={batch.get('view_ids')}: {exc}"
            ) from exc
        v_final_grid = prediction.velocity
        v_final_tokens = ss_grid_to_tokens(v_final_grid)
        flow_valid = torch.isfinite(v_final_grid).flatten(1).all(dim=1)
        foundation_valid = latent_valid & condition_valid & flow_valid
        predicted_clean = velocity_to_clean_state(x_t, v_final_grid, timestep, args.sigma_min)
        target_surface_points, target_surface_weights = (
            surface_targets_from_fusion(fused)
            if args.use_surface_loss and fused.voxel_indices.numel() > 0
            else (None, None)
        )
        loss_timer = CudaStageTimer(device, args.profile)
        loss_timer.start()
        losses = loss_builder(
            v_final=v_final_tokens,
            v_target=target_tokens,
            v_base=torch.zeros_like(v_final_tokens),
            gate=torch.ones_like(fused.voxel_confidence),
            valid_mask=torch.ones_like(fused.valid_mask),
            sample_weight=foundation_valid.float(),
            predicted_clean_grid=predicted_clean,
            ss_decoder=(
                decoder
                if args.use_occupancy_loss or args.use_depth_loss or args.use_silhouette_loss or args.use_surface_loss
                else None
            ),
            gt_occ=batch.get("gt_occ") if args.use_occupancy_loss else None,
            masks=batch.get("masks"),
            view_valid_mask=batch.get("view_valid_mask"),
            K_dataset=batch.get("K_dataset"),
            c2w_dataset=batch.get("c2w_dataset"),
            canonical_center=batch.get("canonical_center"),
            canonical_half_extent=batch.get("canonical_half_extent"),
            vggt_depth=geometry.depth,
            vggt_depth_confidence=geometry.depth_confidence,
            alignment_scale=fused.alignment_scale,
            alignment_quality=fused.alignment_plausible_fraction,
            depth_target=fused.aligned_depth,
            depth_supervision_weight=fused.depth_supervision_weight,
            target_surface_points=target_surface_points,
            target_surface_weights=target_surface_weights,
        )
        losses_ms = loss_timer.stop()
    conditioning_available = (
        fused.conditioning_available
        if fused.conditioning_available is not None
        else fused.observation_mask.flatten(1).any(dim=1)
    )
    weak_evidence_only = (
        fused.weak_evidence_only
        if fused.weak_evidence_only is not None
        else torch.zeros_like(conditioning_available)
    )
    positive_weights = fused.accumulated_weight[fused.observation_mask]
    return {
        "losses": losses,
        "adapter_diagnostics": prediction.diagnostics,
        "flow_backend": flow_backend,
        "observed_fraction": fused.observation_mask.float().mean(),
        "observed_voxel_count": fused.observation_mask.sum(),
        "conditioning_available": conditioning_available.float().mean(),
        "weak_evidence_only": weak_evidence_only.float().mean(),
        "minimum_positive_observation_weight": (
            positive_weights.min() if positive_weights.numel() else fused.accumulated_weight.new_zeros(())
        ),
        "maximum_observation_weight": fused.accumulated_weight.max(),
        "alignment_plausible_fraction": fused.alignment_plausible_fraction.mean(),
        "alignment_scale": fused.alignment_scale.mean(),
        "alignment_mode": fused.alignment_mode,
        "geometry_available": (
            fused.geometry_available.float().mean()
            if fused.geometry_available is not None
            else fused.observation_mask.float().flatten(1).any(dim=1).float().mean()
        ),
        "geometry_quality": (
            fused.geometry_quality.mean()
            if fused.geometry_quality is not None
            else fused.alignment_plausible_fraction.mean()
        ),
        "geometry_status": fused.geometry_status,
        "geometry_reason": fused.geometry_reason,
        "foundation_valid": foundation_valid.float().mean(),
        "foundation_reason": _foundation_failure_reason(
            latent_valid, condition_valid, flow_valid
        ),
        "timings": {"vggt_forward": vggt_ms, "frozen_trellis_forward": trellis_ms, "losses": losses_ms, **branch_timings},
    }


def surface_targets_from_fusion(fused) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose confidence-weighted VGGT fused support to GeomLoss/FRNN."""
    batch_ids = fused.voxel_indices[:, 0].long()
    if batch_ids.numel() == 0 or int(batch_ids.max().item()) != 0:
        raise RuntimeError("Surface supervision currently requires the enforced per-rank batch_size=1 contract")
    indices = fused.voxel_indices.long()
    resolution = 16
    local_keys = indices[:, 3] + resolution * indices[:, 2] + resolution**2 * indices[:, 1]
    weights = fused.voxel_confidence[batch_ids, local_keys, 0].float().clamp_min(1e-6)
    return fused.voxel_xyz.unsqueeze(0), weights.unsqueeze(0)


def _foundation_failure_reason(
    latent_valid: torch.Tensor,
    condition_valid: torch.Tensor,
    base_flow_valid: torch.Tensor,
) -> str:
    reasons = []
    if not bool(latent_valid.all()):
        reasons.append("nonfinite_ss_latent")
    if not bool(condition_valid.all()):
        reasons.append("nonfinite_trellis_image_condition")
    if not bool(base_flow_valid.all()):
        reasons.append("nonfinite_frozen_base_velocity")
    return ",".join(reasons)


def build_validation_loader(args, rank: int, world_size: int, device: torch.device):
    if args.validate_every <= 0 or args.validation_batches <= 0:
        return None
    view_counts = parse_view_counts(args.validation_view_counts)
    validation_dataset = StochasticMeshFleetDataset(
        args.dataset_root,
        split=args.validation_split,
        min_views=max(view_counts),
        max_views=max(view_counts),
        seed=args.seed + 17,
        rank=rank,
        use_stochastic_views=False,
        image_size=args.image_size,
        load_gt_occupancy=args.use_occupancy_loss,
        occupancy_resolution=64,
        meshfleet_root=args.meshfleet_root,
        require_ss_latents=True,
    )
    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    validation_workers = max(0, int(args.validation_workers))
    return DataLoader(
        validation_dataset,
        batch_size=1,
        sampler=validation_sampler,
        num_workers=validation_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
        persistent_workers=args.persistent_workers and validation_workers > 0,
        worker_init_fn=seed_meshfleet_worker,
        collate_fn=stochastic_meshfleet_collate,
        drop_last=False,
        prefetch_factor=args.prefetch_factor if validation_workers > 0 else None,
    )


def run_validation(
    args,
    loader,
    vggt,
    pipeline,
    base_flow,
    decoder,
    trainable,
    loss_builder,
    precision,
    device,
    rank: int,
    step: int,
) -> Dict[str, Any]:
    view_counts = parse_view_counts(args.validation_view_counts)
    totals = {count: torch.zeros((), device=device) for count in view_counts}
    batches = torch.zeros((), device=device)
    was_training = trainable.training
    trainable.eval()
    cuda_devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        iterator = iter(loader)
        for batch_index in range(args.validation_batches):
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                break
            full_batch = move_batch(cpu_batch, device)
            for view_count in view_counts:
                batch = slice_validation_views(full_batch, view_count)
                validation_seed = args.seed + 1_000_003 * step + 10_007 * batch_index + 101 * view_count + rank
                torch.manual_seed(validation_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(validation_seed)
                payload = forward_training_batch(
                    args,
                    batch,
                    vggt,
                    pipeline,
                    base_flow,
                    decoder,
                    trainable,
                    loss_builder,
                    precision,
                    device,
                )
                totals[view_count] += payload["losses"]["loss_total"].detach().float()
            batches += 1
    if was_training:
        trainable.train()
    if dist.is_initialized():
        dist.all_reduce(batches, op=dist.ReduceOp.SUM)
        for value in totals.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
    if batches.item() <= 0:
        raise RuntimeError("Validation loader produced no batches")
    return {
        "event": "validation",
        "step": step,
        "view_counts": view_counts,
        "loss_by_view_count": {str(count): float((totals[count] / batches).cpu()) for count in view_counts},
        "global_batches": int(batches.item()),
    }


def slice_validation_views(batch: Dict[str, Any], requested_views: int) -> Dict[str, Any]:
    available = int(batch["view_valid_mask"][0].sum().item())
    count = min(int(requested_views), available)
    if count < 1:
        raise RuntimeError("Validation object has no usable views")
    view_keys = {
        "images",
        "masks",
        "K_dataset",
        "c2w_dataset",
        "w2c_dataset",
        "view_valid_mask",
        "view_ids",
        "view_metadata_indices",
    }
    result = {
        key: value[:, :count] if key in view_keys and isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    result["num_views"] = torch.full_like(batch["num_views"], count)
    return result


def load_frozen_foundations(args, device):
    if not Path(args.trellis_root).is_dir():
        raise FileNotFoundError(f"TRELLIS root not found: {args.trellis_root}")
    if not Path(args.vggt_root).is_dir():
        raise FileNotFoundError(f"VGGT root not found: {args.vggt_root}")
    sys.path.insert(0, args.trellis_root)
    configure_trellis_hub(args)
    try:
        from trellis.pipelines import TrellisImageTo3DPipeline
    except Exception as exc:
        raise RuntimeError("Could not import the real local TRELLIS image-to-3D pipeline") from exc
    trellis_source = resolve_local_hf_snapshot(
        args.trellis_model,
        args.hf_cache_root,
        required_file="pipeline.json",
        validate_trellis_pipeline=True,
    )
    try:
        pipeline = TrellisImageTo3DPipeline.from_pretrained(trellis_source)
    except Exception as exc:
        raise RuntimeError(f"Could not load real local TRELLIS pipeline {trellis_source!r}") from exc
    pipeline.to(device)
    required = ("sparse_structure_flow_model", "sparse_structure_decoder", "image_cond_model")
    missing = [name for name in required if name not in pipeline.models]
    if missing:
        raise RuntimeError(f"TRELLIS pipeline lacks production components: {missing}")
    for model in pipeline.models.values():
        if isinstance(model, nn.Module):
            model.eval().requires_grad_(False)
    base_flow = pipeline.models["sparse_structure_flow_model"]
    decoder = pipeline.models["sparse_structure_decoder"]
    contract = (base_flow.in_channels, base_flow.out_channels, base_flow.resolution, base_flow.cond_channels)
    if contract != (8, 8, 16, 1024):
        raise RuntimeError(f"Incompatible TRELLIS SS contract {contract}; expected (8,8,16,1024)")
    vggt_source = None
    if args.vggt_checkpoint is None:
        vggt_source = resolve_local_hf_snapshot(
            args.vggt_pretrained,
            args.hf_cache_root,
            required_file="model.safetensors",
        )
    vggt = VGGTGeometryWrapper(
        vggt_root=args.vggt_root,
        checkpoint=args.vggt_checkpoint,
        pretrained_name=vggt_source,
        mock=False,
        require_real=True,
        vggt_image_size=518,
    ).to(device)
    return vggt, pipeline, base_flow, decoder


@torch.inference_mode()
def encode_real_trellis_condition(
    pipeline,
    images: torch.Tensor,
    precision: torch.dtype,
    *,
    return_validity: bool = False,
):
    if images.ndim != 5 or images.shape[0] < 1:
        raise ValueError(f"Images must be real [B,V,3,H,W], got {tuple(images.shape)}")
    batch_size, views = images.shape[:2]
    all_views = torch.nn.functional.interpolate(
        images.reshape(batch_size * views, 3, *images.shape[-2:]).float(),
        (518, 518),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    with torch.autocast(device_type=images.device.type, dtype=precision, enabled=images.device.type == "cuda"):
        condition = pipeline.encode_image(all_views)
    if condition.ndim != 3 or condition.shape[-1] != 1024:
        raise RuntimeError(f"Real TRELLIS DINO condition contract failed: {tuple(condition.shape)}")
    tokens = condition.shape[1]
    valid = torch.isfinite(condition).flatten(1).all(dim=1).reshape(
        batch_size, views
    ).any(dim=1)
    condition = torch.nan_to_num(
        condition.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).to(dtype=condition.dtype).reshape(batch_size, views * tokens, 1024)
    return (condition, valid) if return_validity else condition


def velocity_to_clean_state(x_t: torch.Tensor, velocity: torch.Tensor, t: torch.Tensor, sigma_min: float) -> torch.Tensor:
    t_view = t.view(-1, *([1] * (x_t.ndim - 1))).float()
    # Exact inverse used by local TRELLIS FlowEulerSampler._v_to_xstart_eps.
    return (1.0 - sigma_min) * x_t - (sigma_min + (1.0 - sigma_min) * t_view) * velocity


def sample_timestep(batch_size: int, device: torch.device, args) -> torch.Tensor:
    if args.timestep_sampling == "uniform":
        return torch.rand(batch_size, device=device)
    return torch.sigmoid(
        torch.randn(batch_size, device=device) * args.timestep_logit_std + args.timestep_logit_mean
    )


def init_distributed(args) -> Dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds visible CUDA devices={torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1:
        dist.init_process_group(backend=backend, init_method="env://")
    args.rank, args.world_size, args.local_rank = rank, world_size, local_rank
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank, "device": device}


def seed_everything(seed: int, rank: int) -> None:
    rank_seed = int(seed) + 100_003 * int(rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed % (2**32))
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def resolve_precision(name: str, device: torch.device) -> torch.dtype:
    if name == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if name in {"bf16", "fp16"} and device.type == "cuda":
        return torch.float16
    return torch.float32


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def distributed_mean(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= dist.get_world_size()
    return result


def distributed_all_true(local_value: bool, device: torch.device) -> bool:
    """Rank-synchronous health decision used before backward/optimizer collectives."""
    flag = torch.tensor(1 if local_value else 0, device=device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def maybe_resume(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    rank: int,
    *,
    batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int, int]:
    if path is None:
        return 0, 0, 0
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    architecture_version = payload.get("architecture_version")
    if architecture_version != AffostructionSSFlow.architecture_version:
        raise RuntimeError(
            "Resume checkpoint belongs to a different SS propagation graph: "
            f"checkpoint={architecture_version!r}, required={AffostructionSSFlow.architecture_version!r}. "
            "Use the legacy checkpoint for inference or start this full-backbone migration from TRELLIS weights."
        )
    numerics_version = payload.get("optimization_numerics_version")
    if numerics_version != SS_OPTIMIZATION_NUMERICS_VERSION:
        raise RuntimeError(
            "Resume checkpoint was produced before FP32-master optimization was enforced: "
            f"checkpoint={numerics_version!r}, required={SS_OPTIMIZATION_NUMERICS_VERSION!r}. "
            "Restart from the original TRELLIS weights."
        )
    assert_state_dict_finite(payload["model"], f"SS checkpoint {checkpoint_path}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    rng_by_rank = payload.get("rng_by_rank")
    if rng_by_rank is not None:
        if rank >= len(rng_by_rank):
            raise RuntimeError(f"Checkpoint contains {len(rng_by_rank)} RNG streams but resume requested rank={rank}")
        rng = rng_by_rank[rank]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state(rng["cuda"].cpu(), device)
    else:  # backward compatibility with earlier rank-0-only checkpoints
        torch.set_rng_state(payload["torch_rng"].cpu())
        if device.type == "cuda" and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(payload["cuda_rng"].cpu(), device)
        if rank != 0:
            seed_everything(int(payload.get("args", {}).get("seed", 42)) + 1_000_003 * int(payload["step"]), rank)
    step = int(payload["step"])
    batches_seen = payload.get("batches_seen_in_epoch")
    if batches_seen is None:
        # Legacy checkpoints omitted the sampler cursor. Global optimizer-step
        # count is the only faithful reconstruction when no epoch boundary was
        # recorded between saves.
        batches_seen = (
            step * int(gradient_accumulation_steps)
        ) % max(int(batches_per_epoch), 1)
    return step, int(payload["epoch"]), int(batches_seen)


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    epoch,
    args,
    rank: int,
    batches_seen_in_epoch: int,
) -> None:
    local_rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    if dist.is_initialized():
        rng_by_rank = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(local_rng, rng_by_rank, dst=0)
    else:
        rng_by_rank = [local_rng]
    if rank != 0:
        return
    unwrapped = model.module if isinstance(model, DDP) else model
    assert_fp32_finite_trainable_parameters(
        unwrapped, f"SS checkpoint step {step}"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "architecture_version": AffostructionSSFlow.architecture_version,
            "optimization_numerics_version": SS_OPTIMIZATION_NUMERICS_VERSION,
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "epoch": epoch,
            "batches_seen_in_epoch": int(batches_seen_in_epoch),
            "args": vars(args),
            "rng_by_rank": rng_by_rank,
            # Retain legacy fields for older readers.
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    os.replace(temporary, path)


def report_parameter_counts(rank: int, trainable: nn.Module, frozen_modules: Iterable[Any]) -> None:
    if rank != 0:
        return
    train_count = sum(parameter.numel() for parameter in trainable.parameters() if parameter.requires_grad)
    frozen_count = 0
    seen = {
        id(parameter)
        for parameter in trainable.parameters()
        if parameter.requires_grad
    }
    for module in frozen_modules:
        modules = module.models.values() if hasattr(module, "models") else (module,)
        for item in modules:
            if not isinstance(item, nn.Module):
                continue
            for parameter in item.parameters():
                if not parameter.requires_grad and id(parameter) not in seen:
                    frozen_count += parameter.numel()
                    seen.add(id(parameter))
    if train_count <= 0:
        raise RuntimeError("Affostruction SS full-backbone graph has no trainable parameters")
    print(
        json.dumps(
            {
                "architecture_version": AffostructionSSFlow.architecture_version,
                "trainable_parameters": train_count,
                "frozen_foundation_parameters": frozen_count,
            }
        )
    )


def assert_fp32_finite_trainable_parameters(module: nn.Module, context: str) -> None:
    wrong_dtype = [
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and parameter.dtype != torch.float32
    ]
    if wrong_dtype:
        raise TypeError(
            f"{context} requires FP32 AdamW master parameters; non-FP32 tensors: {wrong_dtype[:16]}"
        )
    if not trainable_parameters_are_finite(module):
        raise FloatingPointError(f"{context} contains NaN or Inf parameters")


def trainable_parameters_are_finite(module: nn.Module) -> bool:
    checks = [
        torch.isfinite(parameter.detach()).all()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    return bool(torch.stack(checks).all().item()) if checks else False


def assert_state_dict_finite(state: Dict[str, Any], context: str) -> None:
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


def warmup_cosine_factor(
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


def build_metrics(
    step,
    epoch,
    loss,
    payload,
    timings,
    dataloader,
    elapsed,
    grad_norm,
    optimizer,
    batch,
    world_size,
    device,
    optimizer_step_applied=True,
    optimizer_skip_reason="",
):
    diagnostics = payload["adapter_diagnostics"]
    losses = payload["losses"]
    loss_diagnostics = losses["diagnostics"]
    views = int(batch["view_valid_mask"].sum().item())
    uid = batch.get("uid", [""])
    uid = uid[0] if isinstance(uid, (list, tuple)) else uid
    selected_view_ids = batch["view_ids"][batch["view_valid_mask"]].detach().cpu().tolist()
    metrics = {
        "step": step,
        "epoch": epoch,
        "uid": uid,
        "view_ids": selected_view_ids,
        "loss_total": float(loss.cpu()),
        "loss_cfm": float(distributed_mean(losses["loss_cfm"]).cpu()),
        "loss_depth": float(distributed_mean(losses["loss_depth"]).cpu()),
        "loss_silhouette": float(distributed_mean(losses["loss_silhouette"]).cpu()),
        "loss_occupancy": float(distributed_mean(losses["loss_occupancy"]).cpu()),
        "loss_surface": float(distributed_mean(losses["loss_surface"]).cpu()),
        "loss_prior": float(distributed_mean(losses["loss_prior"]).cpu()),
        "architecture_version": AffostructionSSFlow.architecture_version,
        "backbone_velocity_norm": float(diagnostics["mean_adapter_norm"].detach().float().cpu()),
        "conditioning_voxel_count": float(diagnostics["conditioning_voxel_count"].detach().float().cpu()),
        "geometry_condition_fraction": float(diagnostics["geometry_condition_fraction"].detach().float().cpu()),
        "observed_percent": float(diagnostics["percentage_observed"].detach().float().cpu()),
        "alignment_plausible_percent": float(
            100.0 * payload["alignment_plausible_fraction"].detach().float().cpu()
        ),
        "alignment_scale": float(payload["alignment_scale"].detach().float().cpu()),
        "alignment_mode": payload["alignment_mode"],
        "geometry_available": bool(payload["geometry_available"].detach().float().cpu() > 0.5),
        "conditioning_available": bool(
            payload["conditioning_available"].detach().float().cpu() > 0.5
        ),
        "weak_evidence_only": bool(
            payload["weak_evidence_only"].detach().float().cpu() > 0.5
        ),
        "observed_voxel_count": int(
            payload["observed_voxel_count"].detach().cpu()
        ),
        "minimum_positive_observation_weight": float(
            payload["minimum_positive_observation_weight"].detach().float().cpu()
        ),
        "maximum_observation_weight": float(
            payload["maximum_observation_weight"].detach().float().cpu()
        ),
        "geometry_quality_percent": float(
            100.0 * payload["geometry_quality"].detach().float().cpu()
        ),
        "geometry_status": payload["geometry_status"],
        "geometry_reason": payload["geometry_reason"],
        "foundation_valid": bool(payload["foundation_valid"].detach().float().cpu() > 0.5),
        "foundation_reason": payload["foundation_reason"],
        "decoder_numerically_valid": bool(
            loss_diagnostics.get("decoder_numerically_valid", True)
        ),
        "decoder_empty_active": bool(loss_diagnostics.get("decoder_empty_active", False)),
        "decoder_active_percent": float(
            100.0 * _diagnostic_scalar(loss_diagnostics, "decoder_active_fraction", 0.0)
        ),
        "decoder_mean_occupancy_probability": float(
            _diagnostic_scalar(
                loss_diagnostics, "decoder_mean_occupancy_probability", 0.0
            )
        ),
        "depth_supervision_available": bool(
            loss_diagnostics.get("depth_available", False)
        ),
        "adapter_numerical_fallback": bool(
            _diagnostic_scalar(diagnostics, "adapter_numerical_fallback", 0.0) > 0.5
        ),
        "grad_norm": float(grad_norm.detach().float().cpu()),
        "optimizer_step_applied": bool(optimizer_step_applied),
        "optimizer_skip_reason": optimizer_skip_reason,
        "loss_requires_grad": bool(payload.get("loss_requires_grad", True)),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "flow_backend": payload["flow_backend"],
        "dataloader_wait": dataloader,
        "step_seconds": elapsed,
        "samples_per_second": world_size / max(elapsed, 1e-8),
        "views_per_second": views * world_size / max(elapsed, 1e-8),
        "max_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "max_reserved_cuda_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "timings_ms": timings,
    }
    return metrics


def append_metrics(jsonl_path: Path, csv_path: Path, metrics: Dict[str, Any]) -> None:
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, default=json_default) + "\n")
    flat = {key: value for key, value in metrics.items() if isinstance(value, (str, int, float, bool))}
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    fieldnames = list(flat)
    if not new_file:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header:
            # Resumed runs may add JSONL diagnostics without corrupting the
            # fixed schema of an already-created CSV metrics file.
            fieldnames = existing_header
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(flat)


def append_pathology_record(
    path: Path,
    *,
    step: int,
    accumulation_index: int,
    rank: int,
    batch: Dict[str, Any],
    payload: Dict[str, Any],
    loss_finite: bool,
    loss_requires_grad: bool,
) -> None:
    """Persist non-normal sample outcomes independently on every DDP rank."""
    diagnostics = payload["losses"]["diagnostics"]
    adapter = payload["adapter_diagnostics"]
    decoder_empty = bool(diagnostics.get("decoder_empty_active", False))
    decoder_valid = bool(diagnostics.get("decoder_numerically_valid", True))
    adapter_fallback = _diagnostic_scalar(
        adapter, "adapter_numerical_fallback", 0.0
    ) > 0.5
    geometry_status = str(payload.get("geometry_status", "usable"))
    conditioning_available = _diagnostic_scalar(
        payload, "conditioning_available", 1.0
    ) > 0.5
    weak_evidence_only = _diagnostic_scalar(
        payload, "weak_evidence_only", 0.0
    ) > 0.5
    foundation_valid = bool(
        float(payload.get("foundation_valid", torch.tensor(1.0)).detach().float().cpu())
        > 0.5
    )
    if (
        geometry_status == "usable"
        and conditioning_available
        and not weak_evidence_only
        and foundation_valid
        and not decoder_empty
        and decoder_valid
        and not adapter_fallback
        and loss_finite
        and loss_requires_grad
    ):
        return
    uid = batch.get("uid", [""])
    uid = uid[0] if isinstance(uid, (list, tuple)) else uid
    view_mask = batch["view_valid_mask"]
    record = {
        "event": "sample_pathology",
        "step": step,
        "accumulation_index": accumulation_index,
        "rank": rank,
        "uid": uid,
        "view_ids": batch["view_ids"][view_mask].detach().cpu().tolist(),
        "geometry_status": geometry_status,
        "geometry_reason": payload.get("geometry_reason", ""),
        "conditioning_available": conditioning_available,
        "weak_evidence_only": weak_evidence_only,
        "observed_voxel_count": int(
            _diagnostic_scalar(payload, "observed_voxel_count", 0.0)
        ),
        "minimum_positive_observation_weight": _diagnostic_scalar(
            payload, "minimum_positive_observation_weight", 0.0
        ),
        "maximum_observation_weight": _diagnostic_scalar(
            payload, "maximum_observation_weight", 0.0
        ),
        "foundation_valid": foundation_valid,
        "foundation_reason": payload.get("foundation_reason", ""),
        "alignment_mode": payload.get("alignment_mode", ""),
        "geometry_quality_percent": 100.0
        * float(payload["geometry_quality"].detach().float().cpu()),
        "decoder_numerically_valid": decoder_valid,
        "decoder_empty_active": decoder_empty,
        "decoder_active_percent": 100.0
        * _diagnostic_scalar(diagnostics, "decoder_active_fraction", 0.0),
        "decoder_mean_occupancy_probability": _diagnostic_scalar(
            diagnostics, "decoder_mean_occupancy_probability", 0.0
        ),
        "depth_supervision_available": bool(diagnostics.get("depth_available", False)),
        "adapter_numerical_fallback": adapter_fallback,
        "loss_finite": bool(loss_finite),
        "loss_requires_grad": bool(loss_requires_grad),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=json_default) + "\n")


def _diagnostic_scalar(mapping: Dict[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().mean().cpu())
    return float(value)


def json_default(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().item() if value.numel() == 1 else value.detach().float().cpu().tolist()
    return str(value)


class CudaStageTimer:
    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.enabled = bool(enabled and device.type == "cuda")
        self.start_event = torch.cuda.Event(enable_timing=True) if self.enabled else None
        self.end_event = torch.cuda.Event(enable_timing=True) if self.enabled else None

    def start(self) -> None:
        if self.start_event is not None:
            self.start_event.record()

    def stop(self) -> Optional[float]:
        if self.start_event is None or self.end_event is None:
            return None
        self.end_event.record()
        self.end_event.synchronize()  # profiler boundary only
        return float(self.start_event.elapsed_time(self.end_event))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stochastic_voxel_ss_flow.yaml")
    parser.add_argument("--trellis-root", required=False, default="/mnt/sda/hf/MVG/Base/TRELLIS")
    parser.add_argument("--trellis-model", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--vggt-root", default="/mnt/sda/hf/MVG/Base/vggt")
    parser.add_argument("--vggt-pretrained", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--hf-cache-root", default="/mnt/sda/hf/.cache/huggingface/hub")
    parser.add_argument("--meshfleet-root", default="/mnt/sda/hf/MVG/Base/MeshFleet")
    parser.add_argument("--dataset-root", default="/mnt/sda/hf/MVG/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97")
    parser.add_argument("--extensions-root", default="/mnt/sda/hf/MVG/Base/extensions")
    parser.add_argument("--torch-hub-dir", default="/mnt/sda/hf/.cache/torch/hub")
    parser.add_argument("--dinov2-repo", default="/mnt/sda/hf/.cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="test")
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=1)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--validation-view-counts", type=parse_view_counts, default=(1, 2, 4, 6, 8))
    parser.add_argument("--output-dir", default="outputs/stochastic_voxel_ss_flow")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--grid-resolution", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--classifier-free-dropout", type=float, default=0.1)
    parser.add_argument("--sigma-min", type=float, default=1e-5)
    parser.add_argument("--timestep-sampling", choices=("uniform", "logit_normal"), default="logit_normal")
    parser.add_argument("--timestep-logit-mean", type=float, default=0.0)
    parser.add_argument("--timestep-logit-std", type=float, default=1.0)
    parser.add_argument("--flow-backend", choices=("auto", "torch", "triton"), default="auto")
    parser.add_argument("--fusion-mode", choices=("confidence", "average_ablation"), default="confidence")
    for name, default in (
        ("use-stochastic-views", True),
        ("use-vggt-depth", True),
        ("use-vggt-pointmap", True),
        ("use-confidence-weighting", True),
        ("use-visibility-weighting", True),
        ("use-3d-positional-encoding", True),
        ("use-occupancy-loss", True),
        ("use-depth-loss", True),
        ("use-silhouette-loss", True),
        ("use-surface-loss", False),
        ("use-spconv-refinement", False),
    ):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=default)
    parser.add_argument("--surface-backend", choices=("geomloss", "frnn"), default="geomloss")
    parser.add_argument("--lambda-cfm", type=float, default=1.0)
    parser.add_argument("--lambda-depth", type=float, default=0.1)
    parser.add_argument("--lambda-silhouette", type=float, default=0.1)
    parser.add_argument("--lambda-occupancy", type=float, default=0.5)
    parser.add_argument("--lambda-surface", type=float, default=0.05)
    parser.add_argument("--loss-render-resolution", type=int, default=64)
    parser.add_argument("--loss-ray-samples", type=int, default=48)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--barrier-every", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    return parser


def apply_yaml_defaults(parser: argparse.ArgumentParser) -> None:
    preliminary, _ = argparse.ArgumentParser(add_help=False).parse_known_args()
    config_arg = next((sys.argv[index + 1] for index, value in enumerate(sys.argv[:-1]) if value == "--config"), None)
    config_path = Path(config_arg or "configs/stochastic_voxel_ss_flow.yaml")
    if not config_path.is_file():
        return
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    flattened = {}
    for section in (
        config,
        config.get("dataset", {}),
        config.get("model", {}),
        config.get("loss", {}),
        config.get("training", {}),
        config.get("evaluation", {}),
    ):
        for key, value in section.items():
            if not isinstance(value, dict):
                flattened[key.replace("-", "_")] = value
    valid_destinations = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in flattened.items() if key in valid_destinations})


def parse_view_counts(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        counts = tuple(int(item) for item in value)
    else:
        counts = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not counts or any(count not in {1, 2, 4, 6, 8} for count in counts):
        raise argparse.ArgumentTypeError("validation view counts must be a comma-separated subset of 1,2,4,6,8")
    return tuple(dict.fromkeys(counts))


if __name__ == "__main__":
    main()
