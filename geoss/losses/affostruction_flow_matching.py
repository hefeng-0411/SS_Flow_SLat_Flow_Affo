from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ConditionalFlowPath:
    noisy_latent: torch.Tensor
    target_velocity: torch.Tensor
    timestep: torch.Tensor
    noise: torch.Tensor


class AffostructionConditionalFlowMatching(nn.Module):
    """Conditional flow-matching path used by Affostruction reconstruction."""

    def __init__(
        self,
        *,
        sigma_min: float = 1.0e-5,
        logit_mean: float = 1.0,
        logit_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)

    @torch.no_grad()
    def sample_path(
        self,
        clean_latent: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> ConditionalFlowPath:
        noise = torch.randn(
            clean_latent.shape,
            device=clean_latent.device,
            dtype=clean_latent.dtype,
            generator=generator,
        )
        logits = torch.randn(
            clean_latent.shape[0],
            device=clean_latent.device,
            dtype=torch.float32,
            generator=generator,
        )
        timestep = torch.sigmoid(logits * self.logit_std + self.logit_mean)
        broadcast_t = timestep.reshape(
            clean_latent.shape[0],
            *((1,) * (clean_latent.ndim - 1)),
        ).to(clean_latent.dtype)
        noisy_latent = (1.0 - broadcast_t) * clean_latent + (
            self.sigma_min + (1.0 - self.sigma_min) * broadcast_t
        ) * noise
        target_velocity = (1.0 - self.sigma_min) * noise - clean_latent
        return ConditionalFlowPath(
            noisy_latent=noisy_latent,
            target_velocity=target_velocity,
            timestep=timestep,
            noise=noise,
        )

    def forward(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(predicted_velocity.float(), target_velocity.float())


__all__ = [
    "AffostructionConditionalFlowMatching",
    "ConditionalFlowPath",
]
