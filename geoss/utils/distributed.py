from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(args) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = getattr(args, "dist_backend", None) or ("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if local_rank < 0 or local_rank >= device_count:
                visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
                raise RuntimeError(
                    "Invalid distributed CUDA topology: "
                    f"LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, "
                    f"torch.cuda.device_count()={device_count}, CUDA_VISIBLE_DEVICES={visible}. "
                    "Relaunch with --nproc_per_node <= number of visible GPUs; the launcher now clamps this automatically."
                )
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method=getattr(args, "dist_url", "env://"))
    else:
        device = torch.device(getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is false: {device}")
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Invalid single-process CUDA device index {index}; "
                    f"torch.cuda.device_count()={torch.cuda.device_count()}."
                )
            torch.cuda.set_device(index)
            # Tensor.device is always indexed (for example cuda:0). Returning
            # an indexed context prevents false contract failures from
            # comparing torch.device('cuda') with torch.device('cuda:0').
            device = torch.device("cuda", index)

    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    args.distributed = distributed
    args.device = str(device)
    return DistributedContext(distributed, rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def maybe_wrap_ddp(model: torch.nn.Module, ctx: DistributedContext, find_unused_parameters: bool = False) -> torch.nn.Module:
    if not ctx.distributed:
        return model
    if ctx.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[ctx.local_rank],
            output_device=ctx.local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    return DistributedDataParallel(model, find_unused_parameters=find_unused_parameters)


def build_dataloader(
    dataset,
    *,
    args,
    ctx: DistributedContext,
    collate_fn: Optional[Callable] = None,
    shuffle: bool = True,
    drop_last: bool = False,
) -> tuple[DataLoader, Optional[DistributedSampler]]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=shuffle,
        drop_last=drop_last,
    ) if ctx.distributed else None
    worker_count = int(getattr(args, "num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=worker_count,
        pin_memory=bool(getattr(args, "pin_memory", False)) and ctx.device.type == "cuda",
        persistent_workers=worker_count > 0,
        prefetch_factor=max(1, int(getattr(args, "prefetch_factor", 2))) if worker_count > 0 else None,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    return loader, sampler


def next_from_loader(iterator, loader: DataLoader, sampler: Optional[DistributedSampler], epoch: int):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= dist.get_world_size()
    return out


def sync_should_stop(should_stop: bool, device: torch.device) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return should_stop
    flag = torch.tensor(1 if should_stop else 0, device=device, dtype=torch.int)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())
