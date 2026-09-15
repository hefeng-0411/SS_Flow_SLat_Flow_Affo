from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from geoss.losses.affostruction_flow_matching import (
    AffostructionConditionalFlowMatching,
)
from geoss.models.official_affostruction_ss import (
    OFFICIAL_SS_CONFIG,
    OfficialAffostructionRGBDConditioner,
)


class _DummyDINO(nn.Module):
    num_register_tokens = 4

    def forward(self, images, is_training=True):
        batch = images.shape[0]
        base = images.mean(dim=(1, 2, 3), keepdim=False)
        patch = base[:, None, None].expand(batch, 256, 1024)
        prefix = torch.zeros(batch, 5, 1024, device=images.device)
        return {"x_prenorm": torch.cat((prefix, patch), dim=1)}


def _identity(value):
    return value


def test_official_stage1_architecture_contract_has_no_adapter_dimensions():
    assert OFFICIAL_SS_CONFIG == {
        "resolution": 16,
        "in_channels": 8,
        "out_channels": 8,
        "model_channels": 768,
        "cond_channels": 1024,
        "num_blocks": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "patch_size": 1,
        "pe_mode": "ape",
        "qk_rms_norm": True,
    }
    assert not any("adapter" in key.lower() or "lora" in key.lower() for key in OFFICIAL_SS_CONFIG)


def test_rgbd_conditioner_averages_views_and_masks_padding():
    conditioner = OfficialAffostructionRGBDConditioner(_DummyDINO(), _identity)
    images = torch.full((2, 2, 3, 224, 224), 0.5)
    depths = torch.full((2, 2, 224, 224), 0.25)
    masks = torch.ones(2, 2, 1, 224, 224)
    K = torch.tensor(
        [[224.0, 0.0, 112.0], [0.0, 224.0, 112.0], [0.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3).repeat(2, 2, 1, 1)
    c2w = torch.eye(4).view(1, 1, 4, 4).repeat(2, 2, 1, 1)
    valid = torch.tensor([[True, False], [True, True]])
    result = conditioner(images, depths, masks, K, c2w, valid, amp_dtype=torch.float32)
    assert result.voxel_counts[0] == result.voxel_counts[1]
    count = int(result.voxel_counts[0])
    torch.testing.assert_close(result.tokens[0, :count], result.tokens[1, :count])
    assert result.valid_mask.all()
    assert torch.isfinite(result.tokens).all()


def test_rgbd_conditioner_ignores_invalid_view_content():
    conditioner = OfficialAffostructionRGBDConditioner(_DummyDINO(), _identity)
    images = torch.full((1, 2, 3, 224, 224), 0.5)
    images[:, 1] = 100.0
    depths = torch.full((1, 2, 224, 224), 0.25)
    masks = torch.ones(1, 2, 1, 224, 224)
    K = torch.tensor(
        [[224.0, 0.0, 112.0], [0.0, 224.0, 112.0], [0.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    c2w = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    one = conditioner(
        images[:, :1], depths[:, :1], masks[:, :1], K[:, :1], c2w[:, :1],
        torch.ones(1, 1, dtype=torch.bool), amp_dtype=torch.float32,
    )
    padded = conditioner(
        images, depths, masks, K, c2w, torch.tensor([[True, False]]),
        amp_dtype=torch.float32,
    )
    torch.testing.assert_close(one.tokens, padded.tokens)
    torch.testing.assert_close(one.voxel_counts, padded.voxel_counts)


def test_official_flow_path_and_velocity_are_exact():
    clean = torch.randn(3, 8, 4, 4, 4)
    matcher = AffostructionConditionalFlowMatching(
        sigma_min=1.0e-5,
        logit_mean=1.0,
        logit_std=1.0,
    )
    path = matcher.sample_path(
        clean,
        generator=torch.Generator().manual_seed(9182),
    )
    timestep = path.timestep.reshape(3, 1, 1, 1, 1)
    expected_noisy = (1.0 - timestep) * clean + (
        1.0e-5 + (1.0 - 1.0e-5) * timestep
    ) * path.noise
    expected_velocity = (1.0 - 1.0e-5) * path.noise - clean
    torch.testing.assert_close(path.noisy_latent, expected_noisy)
    torch.testing.assert_close(path.target_velocity, expected_velocity)
    assert torch.equal(
        matcher(path.target_velocity, path.target_velocity), torch.zeros(())
    )
    assert bool(((path.timestep > 0.0) & (path.timestep < 1.0)).all())


def test_official_trainer_cannot_reenter_adapter_or_auxiliary_loss_path():
    root = Path(__file__).resolve().parents[1]
    trainer = (root / "scripts/train_affostruction_ss_official.py").read_text()
    launcher = (root / "train_ddp_affostruction.sh").read_text()
    for forbidden in (
        "AffostructionSSFlow",
        "ConfidenceSparseVoxelFusion",
        "FlowMatchingLossBuilder",
        "LoRA",
        "loss_depth",
        "loss_silhouette",
        "loss_occupancy",
        "loss_surface",
    ):
        assert forbidden not in trainer
    assert "scripts/train_affostruction_ss_official.py" in launcher
    assert "run_slat()" in launcher
    assert "vggt_affostruction_sparse_residual_slat_flow" not in trainer
    assert '--batch-size "${SS_BATCH_SIZE:-8}"' in launcher
    assert "--precision fp16" in launcher
