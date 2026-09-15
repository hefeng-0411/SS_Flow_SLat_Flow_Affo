from __future__ import annotations

import torch
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoss.slat.models.slat_velocity_adapter import SLATVelocityAdapter
from geoss.slat.utils.normalization import denormalize_slat, normalize_slat
from geoss.slat.utils.active_voxel_utils import active_xyz_to_indices, indices_to_active_xyz
from geoss.slat.losses.factorized_control_loss import factorized_control_loss
from geoss.slat.losses.decoded_asset_loss import flow_x0_from_velocity
from geoss.slat.integration.trellis_slat_hook import GeoVisTrellisSLATWrapper
from scripts.train_geovis_slat import _repad_sparse_prediction


def test_slat_velocity_adapter_identity_and_clipping():
    B, L, C = 2, 32, 8
    x = torch.randn(B, L, C)
    cond = torch.randn(B, L, C)
    slat_conf = torch.ones(B, L, 1)
    ss_conf = torch.ones(B, L, 1)
    v_base = torch.randn(B, L, C)
    adapter = SLATVelocityAdapter(slat_dim=C, cond_dim=C, hidden_dim=32, trust_region=0.05, enabled=True)
    disabled = adapter(x, cond, slat_conf, ss_conf, torch.tensor([0.5, 0.5]), v_base, use_geovis_slat=False)
    assert torch.equal(disabled["v_slat_geo"], v_base)
    out = adapter(x, cond, slat_conf, ss_conf, torch.tensor([1000.0, 1000.0]), v_base)
    assert out["delta_v_slat"].abs().max() <= 0.050001
    assert out["v_slat_geo"].shape == v_base.shape


def test_aligned_fusion_is_token_local_and_padding_safe():
    torch.manual_seed(4)
    B, L, C = 1, 6, 8
    x = torch.randn(B, L, C)
    cond = torch.randn(B, L, C)
    confidence = torch.ones(B, L, 1)
    valid = torch.ones(B, L, 1)
    valid[:, -2:] = 0
    v_base = torch.randn(B, L, C)
    adapter = SLATVelocityAdapter(
        slat_dim=C, cond_dim=C, hidden_dim=32, fusion_mode="aligned", confidence_floor=0.1
    ).eval()
    first = adapter(x, cond, confidence, confidence, 500.0, v_base, token_valid_mask=valid)
    changed = cond.clone()
    changed[:, 0] += 100.0
    second = adapter(x, changed, confidence, confidence, 500.0, v_base, token_valid_mask=valid)
    assert torch.allclose(first["v_slat_geo"][:, 1:], second["v_slat_geo"][:, 1:], atol=1e-6)
    assert torch.equal(first["v_slat_geo"][:, -2:], v_base[:, -2:])


def test_slat_indices_match_trellis_decoder_world_frame():
    indices = torch.tensor([[0, 0, 0], [63, 63, 63]])
    xyz = indices_to_active_xyz(indices, resolution=64)
    assert torch.allclose(xyz[0], torch.full((3,), -0.5 + 0.5 / 64))
    assert torch.allclose(xyz[1], torch.full((3,), 0.5 - 0.5 / 64))
    assert torch.equal(active_xyz_to_indices(xyz, 64), indices)


if __name__ == "__main__":
    test_slat_velocity_adapter_identity_and_clipping()
def test_slat_normalization_round_trip_matches_trellis_contract():
    raw = torch.tensor([[[1.0, 4.0], [3.0, 8.0]]])
    stats = {"mean": [1.0, 2.0], "std": [2.0, 3.0]}
    normalized = normalize_slat(raw, stats)
    assert torch.allclose(normalized, torch.tensor([[[0.0, 2.0 / 3.0], [1.0, 2.0]]]))
    assert torch.allclose(denormalize_slat(normalized, stats), raw)


def test_reliability_and_correction_demand_are_distinct_gates():
    torch.manual_seed(7)
    x = torch.randn(1, 4, 8)
    cond = torch.randn_like(x)
    v_base = torch.zeros_like(x)
    adapter = SLATVelocityAdapter(slat_dim=8, cond_dim=8, hidden_dim=16, confidence_floor=0.0).eval()
    reliable = torch.ones(1, 4, 1)
    no_demand = torch.zeros(1, 4, 1)
    output = adapter(
        x, cond, reliable, reliable, torch.tensor([500.0]), v_base,
        correction_demand=no_demand,
    )
    assert torch.equal(output["v_slat_geo"], v_base)
    assert output["joint_confidence"].min() == 1
    assert output["correction_gate"].max() == 0


def test_factorized_control_loss_trains_demand_and_uncertainty():
    demand = torch.full((1, 3, 1), 0.5, requires_grad=True)
    variance = torch.full((1, 3, 1), 1.0, requires_grad=True)
    prediction = torch.zeros(1, 3, 2, requires_grad=True)
    target = torch.ones_like(prediction)
    terms = factorized_control_loss(demand, variance, prediction, target)
    terms["loss"].backward()
    assert demand.grad is not None and variance.grad is not None and prediction.grad is not None


def test_factorized_control_logits_preserve_bce_objective_and_gradient():
    logits = torch.tensor([[[-0.75], [0.25], [1.25]]], requires_grad=True)
    demand = logits.sigmoid()
    variance = torch.full_like(demand, 0.75, requires_grad=True)
    prediction = torch.zeros(1, 3, 2, requires_grad=True)
    target = torch.tensor([[[0.2, -0.1], [0.5, 0.3], [-0.4, 0.7]]])
    probability_terms = factorized_control_loss(demand, variance, prediction, target)
    logits_terms = factorized_control_loss(
        demand,
        variance,
        prediction,
        target,
        correction_demand_logits=logits,
    )
    assert torch.allclose(
        probability_terms["correction_demand_bce"],
        logits_terms["correction_demand_bce"],
        atol=1e-6,
    )
    logits_terms["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression requires a CUDA device")
def test_factorized_control_loss_is_cuda_autocast_safe():
    device = torch.device("cuda")
    logits = torch.zeros(1, 4, 1, device=device, requires_grad=True)
    variance = torch.ones(1, 4, 1, device=device, requires_grad=True)
    prediction = torch.zeros(1, 4, 8, device=device, requires_grad=True)
    target = torch.ones_like(prediction)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        terms = factorized_control_loss(
            logits.sigmoid(),
            variance,
            prediction,
            target,
            correction_demand_logits=logits,
        )
    assert terms["loss"].dtype == torch.float32
    terms["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_flow_velocity_inversion_recovers_clean_slat():
    torch.manual_seed(9)
    clean = torch.randn(2, 5, 3)
    noise = torch.randn_like(clean)
    timestep = torch.tensor([0.2, 0.8])
    sigma_min = 1e-5
    sigma = sigma_min + (1 - sigma_min) * timestep[:, None, None]
    x_t = (1 - timestep[:, None, None]) * clean + sigma * noise
    velocity = (1 - sigma_min) * noise - clean
    recovered = flow_x0_from_velocity(x_t, velocity, timestep, sigma_min)
    assert torch.allclose(recovered, clean, atol=1e-5)


def test_sparse_teacher_prediction_is_repadded_by_coordinate_not_row_order():
    target_indices = torch.tensor([[[1, 2, 3], [4, 5, 6], [0, 0, 0]]])
    valid = torch.tensor([[True, True, False]])
    coords = torch.tensor([[0, 4, 5, 6], [0, 1, 2, 3]])
    feats = torch.tensor([[40.0], [10.0]])
    padded = _repad_sparse_prediction(feats, coords, target_indices, valid, dtype=torch.float32)
    assert torch.equal(padded, torch.tensor([[[10.0], [40.0], [0.0]]]))


def test_sparse_teacher_prediction_repads_multiple_objects_without_cross_talk():
    target_indices = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6], [0, 0, 0]],
            [[4, 5, 6], [1, 2, 3], [7, 8, 9]],
        ]
    )
    valid = torch.tensor([[True, True, False], [True, False, True]])
    coords = torch.tensor(
        [[1, 7, 8, 9], [0, 4, 5, 6], [1, 4, 5, 6], [0, 1, 2, 3]]
    )
    feats = torch.tensor([[79.0], [46.0], [146.0], [13.0]])
    padded = _repad_sparse_prediction(
        feats, coords, target_indices, valid, dtype=torch.float32
    )
    expected = torch.tensor(
        [[[13.0], [46.0], [0.0]], [[146.0], [0.0], [79.0]]]
    )
    assert torch.equal(padded, expected)


def test_slat_wrapper_residual_scale_cancels_cfg_amplification():
    class ZeroFlow(torch.nn.Module):
        in_channels = 2

        def forward(self, x, t, cond=None, **kwargs):
            return torch.zeros_like(x)

    torch.manual_seed(3)
    adapter = SLATVelocityAdapter(
        slat_dim=2, cond_dim=2, hidden_dim=8, num_heads=1,
        confidence_floor=0.0, trust_region=1.0,
    ).eval()
    wrapper = GeoVisTrellisSLATWrapper(ZeroFlow(), adapter).eval()
    x = torch.randn(1, 3, 2)
    context = {
        "slat_cond_tokens": torch.randn(1, 3, 2),
        "slat_confidence": torch.ones(1, 3, 1),
        "ss_confidence": torch.ones(1, 3, 1),
        "correction_demand": torch.ones(1, 3, 1),
        "residual_variance": torch.zeros(1, 3, 1),
    }
    full = wrapper(x, torch.tensor([500.0]), geovis_slat_context=context)
    quarter = wrapper(
        x, torch.tensor([500.0]), geovis_slat_context=context, geovis_residual_scale=0.25
    )
    assert torch.allclose(quarter, full * 0.25, atol=1e-6)
