from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist


@dataclass
class BatchAdjustment:
    changed: bool
    old_batch_size: int
    new_batch_size: int
    reason: str
    vram_utilization: float | None
    peak_allocated_gb: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "old_batch_size": self.old_batch_size,
            "new_batch_size": self.new_batch_size,
            "reason": self.reason,
            "vram_utilization": self.vram_utilization,
            "peak_allocated_gb": self.peak_allocated_gb,
        }


class AdaptiveBatchController:
    """Per-rank batch-size controller for high VRAM utilization.

    The dataloader still owns actual batching; this controller decides when
    training scripts should rebuild that dataloader with a new `args.batch_size`.
    Growth decisions use the maximum CUDA utilization across DDP ranks so all
    ranks keep the same per-GPU batch size.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: int = 16,
        target_utilization: float = 0.92,
        low_utilization: float = 0.82,
        hard_utilization: float = 0.97,
        grow_patience_steps: int = 8,
        cooldown_steps: int = 3,
        grow_factor: float = 1.25,
        reduce_factor: float = 0.5,
    ) -> None:
        self.enabled = bool(enabled)
        self.batch_size = max(1, int(batch_size))
        self.min_batch_size = max(1, int(min_batch_size))
        self.max_batch_size = max(self.min_batch_size, int(max_batch_size))
        self.target_utilization = _clamp(float(target_utilization), 0.1, 0.99)
        self.low_utilization = _clamp(float(low_utilization), 0.1, self.target_utilization)
        self.hard_utilization = _clamp(float(hard_utilization), self.target_utilization, 0.995)
        self.grow_patience_steps = max(1, int(grow_patience_steps))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.grow_factor = max(1.01, float(grow_factor))
        self.reduce_factor = _clamp(float(reduce_factor), 0.1, 0.95)
        self._low_vram_steps = 0
        self._cooldown_remaining = 0
        self._safe_batch_size = self.batch_size
        self._unsafe_batch_size: int | None = None
        self._last_adjustment = BatchAdjustment(False, self.batch_size, self.batch_size, "init", None, None)

    @classmethod
    def from_args(cls, args: Any) -> "AdaptiveBatchController":
        return cls(
            enabled=bool(getattr(args, "adaptive_batch", False)),
            batch_size=int(getattr(args, "batch_size", 1)),
            min_batch_size=int(getattr(args, "adaptive_min_batch_size", 1)),
            max_batch_size=int(getattr(args, "adaptive_max_batch_size", max(1, getattr(args, "batch_size", 1)))),
            target_utilization=float(getattr(args, "adaptive_target_utilization", 0.92)),
            low_utilization=float(getattr(args, "adaptive_low_utilization", 0.82)),
            hard_utilization=float(getattr(args, "adaptive_hard_utilization", 0.97)),
            grow_patience_steps=int(getattr(args, "adaptive_grow_patience_steps", 8)),
            cooldown_steps=int(getattr(args, "adaptive_cooldown_steps", 3)),
            grow_factor=float(getattr(args, "adaptive_grow_factor", 1.25)),
            reduce_factor=float(getattr(args, "adaptive_reduce_factor", 0.5)),
        )

    @staticmethod
    def is_cuda_oom(exc: BaseException) -> bool:
        text = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in text or "outofmemoryerror" in text

    def update_after_success(self, device: torch.device) -> BatchAdjustment:
        if not self.enabled or device.type != "cuda" or not torch.cuda.is_available():
            self._last_adjustment = BatchAdjustment(False, self.batch_size, self.batch_size, "disabled", None, None)
            return self._last_adjustment
        util, peak_gb = cuda_vram_stats(device)
        util = _sync_max_float(util, device)
        peak_gb = _sync_max_float(peak_gb or 0.0, device)
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._low_vram_steps = 0
            self._last_adjustment = BatchAdjustment(False, self.batch_size, self.batch_size, "cooldown", util, peak_gb)
            _reset_peak_stats(device)
            return self._last_adjustment

        if util >= self.hard_utilization and self.batch_size > self.min_batch_size:
            old = self.batch_size
            self._unsafe_batch_size = old if self._unsafe_batch_size is None else min(self._unsafe_batch_size, old)
            self._safe_batch_size = min(self._safe_batch_size, old - 1)
            self.batch_size = max(self.min_batch_size, old - 1)
            self._low_vram_steps = 0
            self._cooldown_remaining = self.cooldown_steps
            self._last_adjustment = BatchAdjustment(
                True, old, self.batch_size, "shrink_hard_vram", util, peak_gb
            )
            _reset_peak_stats(device)
            return self._last_adjustment

        if util < self.low_utilization and self.batch_size < self.max_batch_size:
            self._low_vram_steps += 1
        else:
            self._low_vram_steps = 0

        if self._low_vram_steps >= self.grow_patience_steps:
            old = self.batch_size
            self._safe_batch_size = max(self._safe_batch_size, old)
            high = self.unsafe_or_max()
            if self._unsafe_batch_size is not None and high > old + 1:
                proposed = (old + high) // 2
                reason = "binary_probe_low_vram"
            else:
                proposed = old + 1
                reason = "sequential_probe_low_vram"
            self.batch_size = min(self.max_batch_size, proposed)
            self._low_vram_steps = 0
            self._cooldown_remaining = self.cooldown_steps
            self._last_adjustment = BatchAdjustment(True, old, self.batch_size, reason, util, peak_gb)
        else:
            self._last_adjustment = BatchAdjustment(False, self.batch_size, self.batch_size, "steady", util, peak_gb)
        _reset_peak_stats(device)
        return self._last_adjustment

    def update_after_oom(self, device: torch.device) -> BatchAdjustment:
        if not self.enabled:
            raise RuntimeError("OOM occurred and adaptive batch is disabled.")
        old = self.batch_size
        self._unsafe_batch_size = old if self._unsafe_batch_size is None else min(self._unsafe_batch_size, old)
        if old <= self.min_batch_size:
            raise RuntimeError(f"OOM at adaptive_min_batch_size={self.min_batch_size}; reduce model/image settings.")
        lower = max(self.min_batch_size, min(self._safe_batch_size, old - 1))
        if lower < old - 1:
            self.batch_size = max(self.min_batch_size, (lower + old - 1) // 2)
        else:
            self.batch_size = max(self.min_batch_size, int(math.floor(old * self.reduce_factor)))
        if self.batch_size == old:
            self.batch_size = old - 1
        self._cooldown_remaining = self.cooldown_steps
        self._low_vram_steps = 0
        if device.type == "cuda":
            torch.cuda.empty_cache()
        util, peak_gb = cuda_vram_stats(device) if device.type == "cuda" else (None, None)
        self._last_adjustment = BatchAdjustment(True, old, self.batch_size, "oom_reduce", util, peak_gb)
        return self._last_adjustment

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "batch_size": self.batch_size,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "target_utilization": self.target_utilization,
            "low_utilization": self.low_utilization,
            "hard_utilization": self.hard_utilization,
            "low_vram_steps": self._low_vram_steps,
            "cooldown_remaining": self._cooldown_remaining,
            "safe_batch_size": self._safe_batch_size,
            "unsafe_batch_size": self._unsafe_batch_size,
            "last_adjustment": self._last_adjustment.as_dict(),
        }

    def unsafe_or_max(self) -> int:
        return min(self.max_batch_size, self._unsafe_batch_size - 1 if self._unsafe_batch_size is not None else self.max_batch_size)


def add_adaptive_batch_args(parser):
    parser.add_argument("--adaptive_batch", type=_str2bool, default=False)
    parser.add_argument("--adaptive_oom_retries", type=int, default=8)
    parser.add_argument("--adaptive_min_batch_size", type=int, default=1)
    parser.add_argument("--adaptive_max_batch_size", type=int, default=16)
    parser.add_argument("--adaptive_target_utilization", type=float, default=0.92)
    parser.add_argument("--adaptive_low_utilization", type=float, default=0.82)
    parser.add_argument("--adaptive_hard_utilization", type=float, default=0.97)
    parser.add_argument("--adaptive_grow_patience_steps", type=int, default=8)
    parser.add_argument("--adaptive_cooldown_steps", type=int, default=3)
    parser.add_argument("--adaptive_grow_factor", type=float, default=1.25)
    parser.add_argument("--adaptive_reduce_factor", type=float, default=0.5)
    return parser


def adaptive_config_defaults(cfg: Mapping[str, Any]) -> dict[str, Any]:
    section = cfg.get("adaptive_batch", {})
    if not isinstance(section, Mapping):
        return {}
    return {
        "adaptive_batch": section.get("enabled"),
        "adaptive_min_batch_size": section.get("min_batch_size"),
        "adaptive_max_batch_size": section.get("max_batch_size"),
        "adaptive_target_utilization": section.get("target_utilization"),
        "adaptive_low_utilization": section.get("low_utilization"),
        "adaptive_hard_utilization": section.get("hard_utilization"),
        "adaptive_grow_patience_steps": section.get("grow_patience_steps"),
        "adaptive_cooldown_steps": section.get("cooldown_steps"),
        "adaptive_grow_factor": section.get("grow_factor"),
        "adaptive_reduce_factor": section.get("reduce_factor"),
    }


def cuda_vram_stats(device: torch.device) -> tuple[float, float]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    used_by_driver = max(0, total - free)
    reserved = torch.cuda.memory_reserved(index)
    peak_allocated = torch.cuda.max_memory_allocated(index)
    externally_used = max(0, used_by_driver - reserved)
    effective_peak = min(total, externally_used + peak_allocated)
    return float(effective_peak / max(1, total)), float(peak_allocated / (1024 ** 3))


def _sync_max_float(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _reset_peak_stats(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(index)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y", "on"}
