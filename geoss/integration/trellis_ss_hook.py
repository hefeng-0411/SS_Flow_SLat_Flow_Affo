from __future__ import annotations

from types import MethodType
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from geoss.models.ss_velocity_adapter import SSVelocityAdapter
from geoss.models.ss_flow_adapter import SSFlowAdapter
from geoss.models.affostruction_conditioning import compact_condition_tokens


def ss_grid_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """Return a zero-copy ``[B,L,C]`` view of a TRELLIS SS grid."""
    if x.ndim != 5:
        raise ValueError(f"SS grid must be [B,C,D,H,W], got {tuple(x.shape)}")
    # A channel-first grid cannot also be token-contiguous without a copy.
    # Linear/attention operators accept this strided view, so keep one storage
    # owner and let the consuming kernel honor the explicit strides.
    return x.flatten(2).transpose(1, 2)


def tokens_to_ss_grid(tokens: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Convert tokens [B,L,C] back to TRELLIS SS grid [B,C,D,H,W]."""
    B, L, C = tokens.shape
    D, H, W = spatial_shape
    if L != D * H * W:
        raise ValueError(f"Token length {L} does not match spatial shape {spatial_shape}")
    return tokens.transpose(1, 2).reshape(B, C, D, H, W).contiguous()


class GeoSSTrellisSSWrapper(nn.Module):
    """Minimal wrapper that injects GeoSS velocity residual after TRELLIS SS velocity output."""

    def __init__(
        self,
        base_model: nn.Module,
        velocity_adapter: Optional[SSVelocityAdapter] = None,
        use_geoss_adapter: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        latent_dim = getattr(base_model, "out_channels", getattr(base_model, "in_channels", 8))
        self.velocity_adapter = velocity_adapter or SSVelocityAdapter(latent_dim=latent_dim)
        self.use_geoss_adapter = use_geoss_adapter
        self.last_debug: Dict[str, Any] = {}
        self.debug_trajectory: list[Dict[str, Any]] = []

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        *,
        geoss_context: Optional[Dict[str, torch.Tensor]] = None,
        use_geoss_adapter: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        v_base = self.base_model(x, t, cond, **kwargs)
        enabled = self.use_geoss_adapter if use_geoss_adapter is None else use_geoss_adapter
        if not enabled or geoss_context is None:
            self.last_debug = {"v_base": v_base.detach(), "enabled": False}
            return v_base
        if "geo_tokens" not in geoss_context or "geo_confidence" not in geoss_context:
            raise ValueError("geoss_context must contain geo_tokens and geo_confidence")
        ss_tokens = ss_grid_to_tokens(x)
        v_base_tokens = ss_grid_to_tokens(v_base)
        vel = self.velocity_adapter(
            ss_latent_tokens=ss_tokens,
            geo_tokens=geoss_context["geo_tokens"],
            geo_confidence=geoss_context["geo_confidence"],
            timestep=t,
            v_base=v_base_tokens,
            voxel_xyz=geoss_context.get("ss_voxel_xyz", _ss_grid_xyz(x, ss_tokens.dtype)),
            anchor_xyz=geoss_context.get("anchor_xyz"),
            anchor_metadata=geoss_context.get("anchor_metadata"),
        )
        v_geo = tokens_to_ss_grid(vel["v_geo"], tuple(x.shape[-3:]))
        self.last_debug = {
            "enabled": True,
            "v_base": v_base.detach(),
            # Stage 2 trains from these debug tensors. Do not detach them here:
            # detaching makes the adapter projections invisible to DDP, which
            # leaves reducer buckets unfinished on the next iteration.
            "delta_v_geo": tokens_to_ss_grid(vel["delta_v_geo"], tuple(x.shape[-3:])),
            "v_geo": v_geo,
            "alpha_t": vel["alpha_t"],
            "token_confidence": vel["token_confidence"],
            "velocity_norm": vel["debug"]["velocity_norm"],
            "velocity_base_norm": vel["debug"]["velocity_base_norm"],
            "velocity_delta_norm": vel["debug"]["velocity_delta_norm"],
            "clipping_ratio": vel["debug"]["clipping_ratio"],
            "confidence_mean": vel["debug"]["confidence_mean"],
        }
        self.debug_trajectory.append(
            {
                "timestep": t.detach().float().mean().cpu(),
                "alpha_t": vel["alpha_t"].detach().float().mean().cpu(),
                "delta_norm": vel["debug"]["velocity_delta_norm"].detach().float().cpu(),
                "clipping_ratio": vel["debug"]["clipping_ratio"].detach().float().cpu(),
                "confidence_mean": vel["debug"]["confidence_mean"].detach().float().cpu(),
            }
        )
        return v_geo

    def reset_debug_trajectory(self) -> None:
        self.debug_trajectory.clear()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def install_trellis_ss_hook(base_model: nn.Module, velocity_adapter: Optional[SSVelocityAdapter] = None) -> GeoSSTrellisSSWrapper:
    return GeoSSTrellisSSWrapper(base_model=base_model, velocity_adapter=velocity_adapter, use_geoss_adapter=True)


class GeometryConditionedTrellisSSWrapper(nn.Module):
    """Production removable wrapper around the frozen real TRELLIS SS DiT."""

    def __init__(self, base_model: nn.Module, adapter: SSFlowAdapter) -> None:
        super().__init__()
        self.base_model = base_model.eval().requires_grad_(False)
        self.adapter = adapter
        self.last_debug: Dict[str, Any] = {}

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        *,
        voxel_condition: Optional[torch.Tensor] = None,
        observation_mask: Optional[torch.Tensor] = None,
        voxel_confidence: Optional[torch.Tensor] = None,
        adapter_enabled: bool = True,
    ) -> torch.Tensor:
        if cond is None:
            raise ValueError("Production TRELLIS SS forward requires real DINO image condition tokens")
        with torch.no_grad():
            v_base = self.base_model(x, t, cond)
        if not adapter_enabled:
            self.last_debug = {"enabled": False, "v_base": v_base}
            return v_base
        if voxel_condition is None or observation_mask is None or voxel_confidence is None:
            raise ValueError("Enabled geometry adapter requires voxel_condition, observation_mask, and voxel_confidence")
        output = self.adapter(
            ss_grid_to_tokens(x),
            voxel_condition,
            t,
            observation_mask,
            voxel_confidence,
            v_base=ss_grid_to_tokens(v_base),
        )
        result = tokens_to_ss_grid(output.v_final, tuple(x.shape[-3:]))
        self.last_debug = {"enabled": True, "v_base": v_base, **output.diagnostics}
        return result


def install_geometry_conditioned_ss_hook(
    base_model: nn.Module,
    adapter: SSFlowAdapter,
) -> GeometryConditionedTrellisSSWrapper:
    return GeometryConditionedTrellisSSWrapper(base_model, adapter)


class DirectConditionedTrellisSSWrapper(nn.Module):
    """Inference wrapper for a full-backbone Affostruction SS checkpoint."""

    architecture_version = "affostruction_direct_ss_cross_attention_v1"

    def __init__(self, flow_model: nn.Module) -> None:
        super().__init__()
        self.flow_model = flow_model
        self.last_debug: Dict[str, Any] = {}

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as original_error:
            wrapped = self.__dict__.get("_modules", {}).get("flow_model")
            if wrapped is not None and hasattr(wrapped, name):
                return getattr(wrapped, name)
            raise original_error

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        *,
        geoss_context: Optional[Dict[str, torch.Tensor]] = None,
        voxel_condition: Optional[torch.Tensor] = None,
        observation_mask: Optional[torch.Tensor] = None,
        adapter_enabled: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if geoss_context is not None:
            if voxel_condition is not None or observation_mask is not None:
                raise ValueError(
                    "Pass direct SS conditioning either as geoss_context or as explicit tensors, not both"
                )
            voxel_condition = geoss_context.get(
                "voxel_condition", geoss_context.get("dense_tokens")
            )
            observation_mask = geoss_context.get("observation_mask")
        if not adapter_enabled or voxel_condition is None:
            return self.flow_model(x, t, cond, **kwargs)
        if observation_mask is None:
            raise ValueError("Direct SS conditioning requires observation_mask")
        compact = compact_condition_tokens(voxel_condition, observation_mask)
        if compact.tokens.shape[0] != 1:
            raise RuntimeError("Direct SS inference requires one object per sampler call")
        if int(compact.counts[0].item()) < 1:
            output = self.flow_model(x, t, cond, **kwargs)
            self.last_debug = {
                "direct_voxel_condition": False,
                "fallback": "native_image_condition",
            }
            return output
        condition = compact.tokens[:, : int(compact.counts[0].item())]
        output = self.flow_model(x, t, condition, **kwargs)
        self.last_debug = {
            "direct_voxel_condition": True,
            "condition_tokens": int(condition.shape[1]),
        }
        return output


def install_direct_conditioned_ss_hook(flow_model: nn.Module) -> DirectConditionedTrellisSSWrapper:
    return DirectConditionedTrellisSSWrapper(flow_model)


def _ss_grid_xyz(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    B, _, D, H, W = x.shape
    # Dense SS tokens correspond to voxel centers, not cube endpoints.
    z = (torch.arange(D, device=x.device, dtype=dtype) + 0.5) / D * 2.0 - 1.0
    y = (torch.arange(H, device=x.device, dtype=dtype) + 0.5) / H * 2.0 - 1.0
    x_coord = (torch.arange(W, device=x.device, dtype=dtype) + 0.5) / W * 2.0 - 1.0
    zz, yy, xx = torch.meshgrid(
        z,
        y,
        x_coord,
        indexing="ij",
    )
    xyz = torch.stack([xx, yy, zz], dim=-1).reshape(1, D * H * W, 3)
    return xyz.expand(B, -1, -1).contiguous()


def zero_geoss_context(geoss_context: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
    if geoss_context is None:
        return None
    zeroed: Dict[str, torch.Tensor] = {}
    for key, value in geoss_context.items():
        zeroed[key] = torch.zeros_like(value) if isinstance(value, torch.Tensor) else value
    return zeroed


def split_geoss_context_for_cfg(
    geoss_context: Optional[Dict[str, torch.Tensor]],
    *,
    geoss_apply_to_uncond: bool = False,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    """Return `(cond_context, uncond_context)` for CFG sampling."""
    if geoss_context is None:
        return None, None
    return geoss_context, geoss_context if geoss_apply_to_uncond else None


class GeoSSSamplerWrapper:
    """Patch a TRELLIS FlowEuler sampler so CFG applies GeoSS only to the conditional branch by default."""

    def __init__(self, sampler: Any, geoss_apply_to_uncond: bool = False) -> None:
        self.sampler = sampler
        self.geoss_apply_to_uncond = geoss_apply_to_uncond
        self._old_inference_model = None
        self.last_cfg_debug: Dict[str, Any] = {}

    def __enter__(self) -> Any:
        self._old_inference_model = self.sampler._inference_model

        def _patched_inference_model(sampler_self, model, x_t, t, cond=None, neg_cond=None, cfg_strength=None, **kwargs):
            geoss_context = kwargs.pop("geoss_context", None)
            apply_to_uncond = kwargs.pop("geoss_apply_to_uncond", self.geoss_apply_to_uncond)
            if neg_cond is None or cfg_strength is None:
                self.last_cfg_debug = {
                    "cfg": False,
                    "cond_geoss": geoss_context is not None,
                    "uncond_geoss": False,
                    "geoss_apply_to_uncond": False,
                }
                sampler_self.last_geoss_cfg_debug = self.last_cfg_debug
                return _call_flow_model(model, x_t, t, cond, geoss_context=geoss_context, **kwargs)
            cond_context, uncond_context = split_geoss_context_for_cfg(
                geoss_context,
                geoss_apply_to_uncond=apply_to_uncond,
            )
            self.last_cfg_debug = {
                "cfg": True,
                "cond_geoss": cond_context is not None,
                "uncond_geoss": uncond_context is not None,
                "geoss_apply_to_uncond": bool(apply_to_uncond),
            }
            sampler_self.last_geoss_cfg_debug = self.last_cfg_debug
            pred = _call_flow_model(model, x_t, t, cond, geoss_context=cond_context, **kwargs)
            neg_pred = _call_flow_model(model, x_t, t, neg_cond, geoss_context=uncond_context, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred

        self.sampler._inference_model = MethodType(_patched_inference_model, self.sampler)
        return self.sampler

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old_inference_model is not None:
            self.sampler._inference_model = self._old_inference_model
        self._old_inference_model = None


def _call_flow_model(
    model: nn.Module,
    x_t: torch.Tensor,
    t: float,
    cond: Optional[torch.Tensor],
    *,
    geoss_context: Optional[Dict[str, torch.Tensor]],
    **kwargs,
) -> torch.Tensor:
    t_tensor = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
    if cond is not None and hasattr(cond, "shape") and cond.shape[0] == 1 and x_t.shape[0] > 1:
        cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
    return model(x_t, t_tensor, cond, geoss_context=geoss_context, **kwargs)
