#!/usr/bin/env python3
"""DDP training for the released Affostruction Stage-1 formulation."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geoss.datasets.dataset_stochastic_meshfleet import (
    StochasticMeshFleetDataset,
    seed_meshfleet_worker,
    stochastic_meshfleet_collate,
)
from geoss.models.official_affostruction_ss import (
    OFFICIAL_SS_ARCHITECTURE,
    OFFICIAL_SS_CONFIG,
    OfficialAffostructionRGBDConditioner,
    build_official_ss_denoiser,
)
from geoss.losses.affostruction_flow_matching import (
    AffostructionConditionalFlowMatching,
)
from geoss.utils.adaptive_batch import AdaptiveBatchController


NUMERICS_VERSION = "official_fp32_master_fp16_amp_v1"


def main() -> None:
    parser = _parser()
    config_path, remaining = _config_path(sys.argv[1:])
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    parser.set_defaults(**_flatten_config(config))
    args = parser.parse_args(remaining)
    args.config = config_path
    rank, world_size, local_rank, device = _init_distributed()
    try:
        _train(args, rank, world_size, local_rank, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(args, rank, world_size, local_rank, device) -> None:
    _seed(args.seed, rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    if args.precision != "fp16":
        raise ValueError("The released Affostruction Stage-1 recipe requires FP16 mixed precision")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    precision = _precision(args.precision, device)
    flow_matching = AffostructionConditionalFlowMatching(
        sigma_min=args.sigma_min,
        logit_mean=args.timestep_logit_mean,
        logit_std=args.timestep_logit_std,
    ).to(device)
    torch.hub.set_dir(args.torch_hub_dir)
    image_model = torch.hub.load(
        args.dinov2_repo,
        "dinov2_vitl14_reg",
        source="local",
        pretrained=True,
    ).to(device).eval().requires_grad_(False)
    image_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    conditioner = OfficialAffostructionRGBDConditioner(
        image_model,
        image_transform,
        image_size=224,
        voxel_resolution=16,
        feature_dim=1024,
    ).to(device).eval()

    model = build_official_ss_denoiser(
        args.affostruction_root,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device=device, dtype=torch.float32)
    initialization = _load_initial_weights(model, args.initial_checkpoint)
    model.train().requires_grad_(True)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    state_elements = sum(value.numel() for value in model.state_dict().values())
    if trainable_parameters != 161_445_128 or state_elements != 164_590_856:
        raise RuntimeError(
            "Official Stage-1 parameter contract failed: "
            f"trainable={trainable_parameters}, state_elements={state_elements}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        eps=args.adam_epsilon,
        fused=bool(args.fused_optimizer and device.type == "cuda"),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == torch.float16
    )
    clipper = _adaptive_clipper(args.trellis_root, args.gradient_clip_norm)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if rank == 0 else None

    batch_controller = AdaptiveBatchController.from_args(args)
    args.batch_size = batch_controller.batch_size
    train_dataset = _dataset(args, args.train_split, rank, stochastic=True)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    train_loader = _loader(args, train_dataset, train_sampler, drop_last=True)
    validation_loader = _validation_loader(args, rank, world_size)

    start_step = 0
    epoch = 0
    if args.resume:
        start_step, epoch = _resume(
            args.resume, model, optimizer, scaler, clipper, ema, device
        )
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "startup",
                    "architecture_version": OFFICIAL_SS_ARCHITECTURE,
                    "official_config": OFFICIAL_SS_CONFIG,
                    "initialization": initialization,
                    "world_size": world_size,
                    "batch_size_per_gpu": args.batch_size,
                    "global_batch_size": args.batch_size
                    * world_size
                    * args.gradient_accumulation_steps,
                    "precision": args.precision,
                    "trainable_parameters": trainable_parameters,
                    "state_elements": state_elements,
                    "loss": "conditional_flow_matching_mse_only",
                    "args": vars(args),
                }
            ),
            flush=True,
        )

    metrics_path = output_dir / "metrics.jsonl"
    train_sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.max_steps + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        loss_sum = torch.zeros((), device=device)
        voxel_count_sum = torch.zeros((), device=device)
        dropout_sum = torch.zeros((), device=device)
        last_uid = []
        for accumulation_index in range(args.gradient_accumulation_steps):
            try:
                cpu_batch = next(iterator)
            except StopIteration:
                epoch += 1
                train_sampler.set_epoch(epoch)
                train_dataset.set_epoch(epoch)
                iterator = iter(train_loader)
                cpu_batch = next(iterator)
            batch = _to_device(cpu_batch, device)
            last_uid = batch["uid"]
            with torch.no_grad():
                condition = conditioner(
                    batch["images"],
                    batch["depths"],
                    batch["masks"],
                    batch["K_dataset"],
                    batch["c2w_dataset"],
                    batch["view_valid_mask"],
                    amp_dtype=precision,
                )
                clean = batch["ss_latent_grid"].float()
                path = flow_matching.sample_path(clean)
                dropped = (
                    torch.rand(clean.shape[0], device=device)
                    < args.classifier_free_dropout
                )
                cond_tokens = torch.where(
                    dropped[:, None, None],
                    torch.zeros_like(condition.tokens),
                    condition.tokens,
                )

            sync = (
                model.no_sync()
                if isinstance(model, DDP)
                and accumulation_index < args.gradient_accumulation_steps - 1
                else contextlib.nullcontext()
            )
            with sync:
                with torch.autocast(
                    device_type=device.type,
                    dtype=precision,
                    enabled=device.type == "cuda" and precision != torch.float32,
                ):
                    prediction = model(
                        path.noisy_latent,
                        path.timestep * 1000.0,
                        cond_tokens,
                        cond_mask=condition.valid_mask,
                    )
                    loss = flow_matching(prediction, path.target_velocity)
                    scaled_loss = loss / args.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
            loss_sum += loss.detach()
            voxel_count_sum += condition.voxel_counts.float().mean()
            dropout_sum += dropped.float().mean()

        scaler.unscale_(optimizer)
        grad_norm = clipper(model.parameters())
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step={step}: {grad_norm}")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        bare_model = model.module if isinstance(model, DDP) else model
        if rank == 0:
            _update_ema(ema, bare_model, args.ema_rate)

        completed_batch_size = int(batch["images"].shape[0])
        peak_allocated = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        )
        batch_adjustment = batch_controller.update_after_success(device)
        if batch_adjustment.changed:
            args.batch_size = batch_adjustment.new_batch_size
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
            train_sampler.set_epoch(epoch)
            train_loader = _loader(args, train_dataset, train_sampler, drop_last=True)
            iterator = iter(train_loader)

        denominator = float(args.gradient_accumulation_steps)
        values = torch.stack(
            (
                loss_sum / denominator,
                voxel_count_sum / denominator,
                dropout_sum / denominator,
            )
        )
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= world_size
        elapsed = time.perf_counter() - started
        record = {
            "step": step,
            "epoch": epoch,
            "loss_cfm_mse": float(values[0]),
            "conditioning_voxels": float(values[1]),
            "unconditional_fraction": float(values[2]),
            "grad_norm": float(grad_norm),
            "adaptive_clip_max_norm": float(clipper._max_norm)
            if clipper._max_norm is not None
            else None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            "samples_per_second": (
                completed_batch_size * world_size * args.gradient_accumulation_steps
            ) / max(elapsed, 1.0e-9),
            "batch_size_per_gpu": completed_batch_size,
            "next_batch_size_per_gpu": args.batch_size,
            "global_batch_size": completed_batch_size
            * world_size
            * args.gradient_accumulation_steps,
            "adaptive_batch": {
                **batch_controller.state_dict(),
                "last_adjustment": batch_adjustment.as_dict(),
            },
            "uids": last_uid,
        }
        if device.type == "cuda":
            record["max_allocated_cuda_bytes"] = peak_allocated
            record["max_reserved_cuda_bytes"] = peak_reserved
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(record), flush=True)

        if validation_loader is not None and step % args.validate_every == 0:
            validation_mse = _validate(
                model,
                conditioner,
                validation_loader,
                flow_matching,
                precision,
                args,
                device,
            )
            if rank == 0:
                with (output_dir / "validation.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"step": step, "cfm_mse": validation_mse}) + "\n"
                    )
        if step % args.save_every == 0 or step == args.max_steps:
            _save(
                output_dir / "affostruction_official_ss_last.pt",
                bare_model,
                optimizer,
                scaler,
                clipper,
                ema,
                step,
                epoch,
                args,
                rank,
                initialization,
            )


@torch.no_grad()
def _validate(model, conditioner, loader, flow_matching, precision, args, device):
    was_training = model.training
    model.eval()
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        batch = _to_device(cpu_batch, device)
        condition = conditioner(
            batch["images"],
            batch["depths"],
            batch["masks"],
            batch["K_dataset"],
            batch["c2w_dataset"],
            batch["view_valid_mask"],
            amp_dtype=precision,
        )
        clean = batch["ss_latent_grid"].float()
        generator = torch.Generator(device=device).manual_seed(
            args.seed + 10_000_019 + batch_index
        )
        path = flow_matching.sample_path(clean, generator=generator)
        with torch.autocast(
            device_type=device.type,
            dtype=precision,
            enabled=device.type == "cuda" and precision != torch.float32,
        ):
            prediction = model(
                path.noisy_latent,
                path.timestep * 1000.0,
                condition.tokens,
                cond_mask=condition.valid_mask,
            )
        total += flow_matching(prediction, path.target_velocity)
        count += 1
    if dist.is_initialized():
        dist.all_reduce(total)
        dist.all_reduce(count)
    if was_training:
        model.train()
    return float(total / count.clamp_min(1.0))


def _dataset(args, split, rank, *, stochastic):
    return StochasticMeshFleetDataset(
        args.dataset_root,
        split=split,
        min_views=args.min_views if stochastic else args.max_views,
        max_views=args.max_views,
        seed=args.seed,
        rank=rank,
        use_stochastic_views=stochastic,
        image_size=224,
        load_gt_occupancy=False,
        meshfleet_root=args.meshfleet_root,
        depth_root=args.depth_root,
        require_depth=True,
        require_ss_latents=True,
    )


def _loader(args, dataset, sampler, *, drop_last):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=seed_meshfleet_worker,
        collate_fn=stochastic_meshfleet_collate,
        drop_last=drop_last,
    )


def _validation_loader(args, rank, world_size):
    if args.validate_every <= 0 or args.validation_batches <= 0:
        return None
    dataset = _dataset(args, args.validation_split, rank, stochastic=False)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    return _loader(args, dataset, sampler, drop_last=False)


def _load_initial_weights(model, path):
    if not path:
        return "official_architecture_random_initialization"
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Initial Affostruction checkpoint not found: {checkpoint}")
    if checkpoint.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint), device="cpu")
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("ema_model", payload.get("model", payload))
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return str(checkpoint.resolve())


def _save(path, model, optimizer, scaler, clipper, ema, step, epoch, args, rank, initialization):
    if dist.is_initialized():
        dist.barrier()
    if rank != 0:
        return
    payload = {
        "architecture_version": OFFICIAL_SS_ARCHITECTURE,
        "optimization_numerics_version": NUMERICS_VERSION,
        "official_model_config": OFFICIAL_SS_CONFIG,
        "algorithm": "Affostruction Eq.1/Eq.2, RGBD average voxel fusion, full Stage-1 flow",
        "initialization": initialization,
        "step": step,
        "epoch": epoch,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "ema_model": {key: value.detach().cpu() for key, value in ema.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "adaptive_grad_clipper": clipper.state_dict(),
        "args": vars(args),
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _resume(path, model, optimizer, scaler, clipper, ema, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("architecture_version") != OFFICIAL_SS_ARCHITECTURE:
        raise RuntimeError(
            f"Cannot resume incompatible architecture {payload.get('architecture_version')!r}"
        )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    clipper.load_state_dict(payload["adaptive_grad_clipper"])
    if ema is not None:
        ema.load_state_dict(payload["ema_model"], strict=True)
    return int(payload["step"]), int(payload["epoch"])


@torch.no_grad()
def _update_ema(ema, model, rate):
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.mul_(rate).add_(parameter, alpha=1.0 - rate)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _adaptive_clipper(trellis_root, max_norm):
    root = str(Path(trellis_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from trellis.utils.grad_clip_utils import AdaptiveGradClipper

    return AdaptiveGradClipper(
        max_norm=max_norm, clip_percentile=95.0, buffer_size=1000
    )


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _init_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Official Affostruction training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://")
    return rank, world_size, local_rank, device


def _seed(seed, rank):
    value = int(seed) + 100_003 * int(rank)
    random.seed(value)
    np.random.seed(value % 2**32)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _precision(name, device):
    if name == "fp16":
        return torch.float16
    if name == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _config_path(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/affostruction_ss_official.yaml")
    known, remaining = pre.parse_known_args(argv)
    return known.config, remaining


def _flatten_config(config):
    flattened = {}
    for section in config.values():
        if isinstance(section, dict):
            flattened.update(section)
    return flattened


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/affostruction_ss_official.yaml")
    parser.add_argument("--affostruction-root")
    parser.add_argument("--trellis-root")
    parser.add_argument("--dinov2-repo")
    parser.add_argument("--torch-hub-dir")
    parser.add_argument("--meshfleet-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--depth-root")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="test")
    parser.add_argument("--output-dir")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--adaptive-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-min-batch-size", type=int, default=1)
    parser.add_argument("--adaptive-max-batch-size", type=int, default=16)
    parser.add_argument("--adaptive-target-utilization", type=float, default=0.94)
    parser.add_argument("--adaptive-low-utilization", type=float, default=0.84)
    parser.add_argument("--adaptive-hard-utilization", type=float, default=0.98)
    parser.add_argument("--adaptive-grow-patience-steps", type=int, default=4)
    parser.add_argument("--adaptive-cooldown-steps", type=int, default=5)
    parser.add_argument("--adaptive-grow-factor", type=float, default=1.25)
    parser.add_argument("--adaptive-reduce-factor", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=450000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--ema-rate", type=float, default=0.9999)
    parser.add_argument("--classifier-free-dropout", type=float, default=0.1)
    parser.add_argument("--sigma-min", type=float, default=1e-5)
    parser.add_argument("--timestep-logit-mean", type=float, default=1.0)
    parser.add_argument("--timestep-logit-std", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    return parser


if __name__ == "__main__":
    main()
