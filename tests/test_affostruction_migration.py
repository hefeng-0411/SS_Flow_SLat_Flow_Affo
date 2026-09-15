from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from geoss.geometry.kaolin_camera import project_opencv_points_kaolin
from geoss.losses.geometric_loss import decoded_surface_cross_view_consistency
from geoss.models.affostruction_conditioning import (
    VoxelPositionalEncoding3D,
    compact_condition_tokens,
    extract_dinov2_spatial_features,
)
from geoss.models.ss_flow_adapter import AffostructionSSFlow
from geoss.integration.trellis_ss_hook import DirectConditionedTrellisSSWrapper
from geoss.slat.integration.trellis_slat_hook import DirectConditionedTrellisSLATWrapper
from geoss.integration.trellis_hub import resolve_local_hf_snapshot
from geoss.integration.real_trellis_pipeline import _atomic_asset_write
from geoss.utils.visualization import save_npz
from geoss.slat.models.slat_flow_adapter import (
    AFFOSTRUCTION_IMAGE_SLAT_CONFIG,
    AffostructionSLatFlow,
    SymmetricSLatConditioner,
)
from scripts.train_geovis_slat import (
    _assert_fp32_finite_trainable_parameters as assert_slat_fp32,
    _warmup_cosine_factor as slat_lr_factor,
)
from scripts.train_stochastic_voxel_ss_flow import (
    assert_fp32_finite_trainable_parameters as assert_ss_fp32,
    warmup_cosine_factor as ss_lr_factor,
)
from scripts.evaluate_affostruction_batch import _canonical_occupancy
from scripts.infer_affostruction_batch import (
    _cleanup_incomplete_output,
    _completed_inference,
)


def test_evaluation_canonicalizes_legacy_singleton_occupancy_axes() -> None:
    legacy = torch.ones(1, 1, 8, 8, 8)
    canonical = _canonical_occupancy(legacy, "prediction")
    assert canonical.shape == (8, 8, 8)
    assert canonical.dtype == torch.bool


def test_atomic_asset_write_never_publishes_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "asset.ply"
    _atomic_asset_write(target, lambda temporary: temporary.write_bytes(b"complete"))
    assert target.read_bytes() == b"complete"

    failed_target = tmp_path / "failed.ply"

    def fail_after_partial_write(temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise OSError("simulated write failure")

    try:
        _atomic_asset_write(failed_target, fail_after_partial_write)
    except OSError:
        pass
    else:
        raise AssertionError("Simulated asset failure was swallowed")
    assert not failed_target.exists()
    assert not list(tmp_path.glob(".*.tmp.ply"))


def test_resume_validation_and_incomplete_cleanup(tmp_path: Path) -> None:
    uid = "unit_uid"
    method = "ss_slat"
    directory = tmp_path / uid
    directory.mkdir()
    for name in (
        "asset_gaussian.ply",
        "asset_mesh_internal.ply",
        "trellis_latents.pt",
    ):
        (directory / name).write_bytes(b"valid")
    np.savez_compressed(directory / "predicted_ss_occ.npz", occ=np.ones((8, 8, 8), dtype=np.uint8))
    metrics = directory / "metrics.json"
    metrics.write_text(
        json.dumps({"status": "ok", "method": method, "uid": uid}),
        encoding="utf-8",
    )
    assert _completed_inference(directory, metrics, method=method, uid=uid)
    metrics.write_text(json.dumps({"status": "failed", "method": method, "uid": uid}))
    assert not _completed_inference(directory, metrics, method=method, uid=uid)
    (directory / "user_note.txt").write_text("preserve")
    _cleanup_incomplete_output(directory)
    assert (directory / "user_note.txt").read_text() == "preserve"
    assert not (directory / "asset_gaussian.ply").exists()


def test_kaolin_projection_matches_opencv_pixels() -> None:
    points = torch.tensor([[[0.0, 0.0, 2.0], [1.0, 1.0, 2.0]]])
    K = torch.tensor([[[[50.0, 0.0, 60.0], [0.0, 40.0, 30.0], [0.0, 0.0, 1.0]]]])
    w2c = torch.eye(4).reshape(1, 1, 4, 4)
    result = project_opencv_points_kaolin(points, K, w2c, 80, 100)
    expected = torch.tensor([[[[60.0, 30.0]], [[85.0, 50.0]]]])
    torch.testing.assert_close(result.pixel, expected, atol=1.0e-4, rtol=1.0e-4)
    assert result.valid.all()


def test_condition_compaction_is_stable_and_never_reindexes_values() -> None:
    tokens = torch.arange(30.0).reshape(2, 5, 3)
    valid = torch.tensor([[False, True, False, True, False], [True, True, True, False, False]])
    result = compact_condition_tokens(tokens, valid)
    torch.testing.assert_close(result.tokens[0, :2], tokens[0, [1, 3]])
    torch.testing.assert_close(result.tokens[1, :3], tokens[1, :3])
    assert result.counts.tolist() == [2, 3]


def test_affostruction_position_encoding_is_additive_width() -> None:
    encoding = VoxelPositionalEncoding3D(1024, 16)
    values = encoding(torch.tensor([[[0, 0, 0], [15, 15, 15]]]))
    assert values.shape == (1, 2, 1024)
    assert torch.isfinite(values).all()
    assert not torch.equal(values[:, 0], values[:, 1])


class _DinoMock(nn.Module):
    num_register_tokens = 4
    patch_size = 2

    def forward(self, images: torch.Tensor, *, is_training: bool) -> dict:
        assert is_training
        batch = images.shape[0]
        values = torch.arange(batch * 9 * 6, device=images.device, dtype=images.dtype)
        return {"x_prenorm": values.reshape(batch, 9, 6)}


def test_dino_extractor_retains_only_spatial_patch_tokens() -> None:
    features = extract_dinov2_spatial_features(
        _DinoMock(),
        lambda value: value,
        torch.rand(1, 2, 3, 8, 8),
        image_size=4,
        amp_dtype=torch.bfloat16,
    )
    assert features.shape == (1, 2, 6, 2, 2)


class _DenseFlowMock(nn.Module):
    cond_channels = 4

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.seen_condition = None

    def forward(self, state: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        self.seen_condition = condition
        return state * self.scale + condition.mean() * 0.0


def test_ss_full_backbone_receives_compact_voxel_condition_and_gradients() -> None:
    backbone = _DenseFlowMock()
    flow = AffostructionSSFlow(
        backbone,
        condition_dim=4,
        classifier_free_dropout=0.0,
        gradient_checkpointing=True,
    )
    state = torch.randn(1, 2, 2, 2, 2)
    condition = torch.randn(1, 8, 4)
    observed = torch.tensor([[True, False, True, False, False, False, False, False]])
    result = flow(state, torch.ones(1), condition, observed)
    assert backbone.seen_condition.shape == (1, 2, 4)
    result.velocity.square().mean().backward()
    assert backbone.scale.grad is not None and torch.isfinite(backbone.scale.grad)


def test_direct_ss_sampler_wrapper_unpacks_context_and_falls_back() -> None:
    backbone = _DenseFlowMock()
    wrapper = DirectConditionedTrellisSSWrapper(backbone)
    state = torch.randn(1, 2, 2, 2, 2)
    image_condition = torch.randn(1, 3, 4)
    voxel_condition = torch.randn(1, 8, 4)
    observed = torch.tensor([[True, False, True, False, False, False, False, False]])
    wrapper(
        state,
        torch.ones(1),
        image_condition,
        geoss_context={
            "voxel_condition": voxel_condition,
            "observation_mask": observed,
        },
    )
    torch.testing.assert_close(backbone.seen_condition, voxel_condition[:, [0, 2]])
    wrapper(
        state,
        torch.ones(1),
        image_condition,
        geoss_context={
            "voxel_condition": voxel_condition,
            "observation_mask": torch.zeros_like(observed),
        },
    )
    torch.testing.assert_close(backbone.seen_condition, image_condition)


def _symmetric_inputs(high_channels: int = 8) -> dict:
    batch, views, height, width = 1, 2, 16, 16
    indices = torch.tensor([[[32, 32, 40], [30, 30, 40], [34, 34, 40]]])
    K = torch.tensor([[[[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]]]])
    K = K.expand(batch, views, -1, -1).clone()
    w2c = torch.eye(4).reshape(1, 1, 4, 4).expand(batch, views, -1, -1).clone()
    w2c[..., 2, 3] = 1.0
    return {
        "active_indices": indices,
        "images": torch.rand(batch, views, 3, height, width),
        "high_features": torch.rand(batch, views, high_channels, 4, 4),
        "intrinsics": K,
        "world_to_camera": w2c,
        "foreground_masks": torch.ones(batch, views, 1, height, width),
        "aligned_depth": torch.full((batch, views, 1, height, width), 1.1328125),
        "vggt_confidence": torch.ones(batch, views, height, width),
    }


def test_symmetric_slat_conditioner_is_pixel_aligned_and_confidence_normalized() -> None:
    conditioner = SymmetricSLatConditioner(
        high_feature_dim=8,
        low_feature_dim=8,
        condition_dim=16,
    )
    result = conditioner(**_symmetric_inputs())
    assert result.condition.shape == (1, 3, 16)
    assert result.condition_valid.all()
    torch.testing.assert_close(
        result.view_weights.sum(dim=2),
        torch.ones_like(result.view_weights.sum(dim=2)),
    )


def test_sparse_image_flow_uses_affostruction_capacity_without_text_conditioning() -> None:
    assert AFFOSTRUCTION_IMAGE_SLAT_CONFIG == {
        "resolution": 64,
        "model_channels": 768,
        "num_blocks": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
        "patch_size": 2,
        "num_io_res_blocks": 2,
        "io_block_channels": [128],
        "pe_mode": "ape",
        "qk_rms_norm": True,
    }
    assert all(
        "text" not in key.lower() and "clip" not in key.lower()
        for key in AFFOSTRUCTION_IMAGE_SLAT_CONFIG
    )


class _SparseState:
    def __init__(self, features: torch.Tensor, coords: torch.Tensor) -> None:
        self.feats = features
        self.coords = coords
        batch_size = int(coords[:, 0].max().item()) + 1 if coords.numel() else 0
        self.shape = torch.Size((batch_size, features.shape[-1]))

    def replace(self, features: torch.Tensor) -> "_SparseState":
        return _SparseState(features, self.coords)


class _SparseFlowMock(nn.Module):
    cond_channels = 16

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.seen_condition = None
        self.seen_condition_mask = None

    def forward(
        self,
        state: _SparseState,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> _SparseState:
        self.seen_condition = condition
        self.seen_condition_mask = cond_mask
        return _SparseState(state.feats * self.scale + condition.mean(), state.coords)


class _ZeroSparseFlowMock(nn.Module):
    cond_channels = 16
    resolution = 64

    def forward(
        self,
        state: _SparseState,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        cond_mask: torch.Tensor | None = None,
    ) -> _SparseState:
        return _SparseState(state.feats * 0.0 + condition.mean() * 0.0, state.coords)


def test_zero_initialized_sparse_flow_is_exact_native_trellis_identity() -> None:
    native = _SparseFlowMock()
    correction = _ZeroSparseFlowMock()
    wrapper = DirectConditionedTrellisSLATWrapper(native, correction)
    coords = torch.tensor([[0, 32, 32, 40], [0, 30, 30, 40], [0, 34, 34, 40]])
    state = _SparseState(torch.randn(3, 8), coords)
    image_condition = torch.randn(1, 5, 16)
    direct = torch.randn(1, 3, 16)
    native_result = native(state, torch.ones(1), image_condition)
    wrapped_result = wrapper(
        state,
        torch.ones(1),
        image_condition,
        geovis_slat_context={
            "condition": direct,
            "condition_valid": torch.ones(1, 3, dtype=torch.bool),
            "confidence": torch.ones(1, 3, 1),
        },
    )
    torch.testing.assert_close(wrapped_result.feats, native_result.feats)


def test_slat_sparse_image_flow_corrects_frozen_base_with_symmetric_condition() -> None:
    conditioner = SymmetricSLatConditioner(
        high_feature_dim=8,
        low_feature_dim=8,
        condition_dim=16,
    )
    spatial_flow = _SparseFlowMock()
    model = AffostructionSLatFlow(
        spatial_flow,
        conditioner,
        classifier_free_dropout=0.0,
    )
    inputs = _symmetric_inputs()
    coords = torch.cat([torch.zeros(3, 1, dtype=torch.long), inputs["active_indices"][0]], dim=-1)
    state = _SparseState(torch.randn(3, 8), coords)
    base = _SparseState(torch.randn(3, 8), coords)
    result = model(state, torch.ones(1), base_velocity=base, **inputs)
    result.velocity.feats.square().mean().backward()
    assert spatial_flow.scale.grad is not None
    assert conditioner.feature_projection[1].weight.grad is not None
    assert result.velocity is not result.base_velocity


def test_slat_sparse_image_flow_supports_true_multi_object_microbatches() -> None:
    conditioner = SymmetricSLatConditioner(
        high_feature_dim=8,
        low_feature_dim=8,
        condition_dim=16,
    )
    spatial_flow = _SparseFlowMock()
    model = AffostructionSLatFlow(
        spatial_flow,
        conditioner,
        classifier_free_dropout=0.0,
    )
    single = _symmetric_inputs()
    inputs = {
        key: value.repeat(2, *([1] * (value.ndim - 1)))
        if isinstance(value, torch.Tensor) and value.shape[0] == 1
        else value
        for key, value in single.items()
    }
    xyz = inputs["active_indices"]
    batch_column = torch.arange(2)[:, None, None].expand(2, xyz.shape[1], 1)
    coords = torch.cat([batch_column, xyz], dim=-1).reshape(-1, 4)
    state = _SparseState(torch.randn(coords.shape[0], 8), coords)
    base = _SparseState(torch.randn(coords.shape[0], 8), coords)
    result = model(state, torch.ones(2), base_velocity=base, **inputs)
    assert result.velocity.feats.shape == (coords.shape[0], 8)
    assert spatial_flow.seen_condition.shape[0] == 2
    assert spatial_flow.seen_condition_mask.shape == spatial_flow.seen_condition.shape[:2]


def test_direct_slat_sampler_wrapper_preserves_native_and_adds_sparse_correction() -> None:
    backbone = _SparseFlowMock()
    correction = _SparseFlowMock()
    wrapper = DirectConditionedTrellisSLATWrapper(backbone, correction)
    coords = torch.tensor([[0, 32, 32, 40], [0, 30, 30, 40], [0, 34, 34, 40]])
    state = _SparseState(torch.randn(3, 8), coords)
    image_condition = torch.randn(1, 5, 16)
    direct = torch.randn(1, 3, 16)
    valid = torch.tensor([[True, False, True]])
    wrapper(
        state,
        torch.ones(1),
        image_condition,
        geovis_slat_context={
            "condition": direct,
            "condition_valid": valid,
            "confidence": torch.ones(1, 3, 1),
        },
    )
    torch.testing.assert_close(backbone.seen_condition, image_condition)
    torch.testing.assert_close(correction.seen_condition, direct[:, [0, 2]])
    identity = wrapper(
        state,
        torch.ones(1),
        image_condition,
        geovis_slat_context={
            "condition": direct,
            "condition_valid": torch.zeros_like(valid),
        },
    )
    torch.testing.assert_close(backbone.seen_condition, image_condition)
    expected_native = state.feats * backbone.scale + image_condition.mean()
    torch.testing.assert_close(identity.feats, expected_native)


def test_slat_zero_visibility_is_rejected_instead_of_training_on_global_pooling() -> None:
    conditioner = SymmetricSLatConditioner(
        high_feature_dim=8,
        low_feature_dim=8,
        condition_dim=16,
    )
    spatial_flow = _SparseFlowMock()
    model = AffostructionSLatFlow(
        spatial_flow,
        conditioner,
        classifier_free_dropout=0.0,
    )
    inputs = _symmetric_inputs()
    inputs["foreground_masks"] = torch.zeros_like(inputs["foreground_masks"])
    coords = torch.cat([torch.zeros(3, 1, dtype=torch.long), inputs["active_indices"][0]], dim=-1)
    state = _SparseState(torch.randn(3, 8), coords)
    try:
        model(state, torch.ones(1), base_velocity=state, **inputs)
    except RuntimeError as error:
        assert "valid multi-view VGGT evidence" in str(error)
    else:
        raise AssertionError("Zero-visibility sample entered sparse image-flow training")


def test_full_backbone_optimizers_require_fp32_master_parameters() -> None:
    fp32 = nn.Linear(4, 4)
    assert_ss_fp32(fp32, "test")
    assert_slat_fp32(fp32, "test")
    fp16 = nn.Linear(4, 4).half()
    for assertion in (assert_ss_fp32, assert_slat_fp32):
        try:
            assertion(fp16, "test")
        except TypeError as error:
            assert "FP32 AdamW master" in str(error)
        else:
            raise AssertionError("FP16 trainable parameters were accepted")


def test_ss_and_slat_warmup_cosine_schedules_share_numerical_contract() -> None:
    for update in (0, 49, 99, 100, 500, 999):
        ss_value = ss_lr_factor(
            update,
            total_updates=1000,
            warmup_updates=100,
            minimum_ratio=0.1,
        )
        slat_value = slat_lr_factor(
            update,
            total_updates=1000,
            warmup_updates=100,
            minimum_ratio=0.1,
        )
        assert ss_value == slat_value
        assert 0.0 < ss_value <= 1.0


def test_trellis_pipeline_resolves_complete_local_snapshot(tmp_path: Path) -> None:
    repository = tmp_path / "models--microsoft--TRELLIS-image-large"
    revision = "complete-revision"
    snapshot = repository / "snapshots" / revision
    (repository / "refs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (repository / "refs" / "main").write_text(revision, encoding="utf-8")
    pipeline = {
        "args": {
            "models": {
                "slat_flow_model": "ckpts/slat_flow",
                "slat_decoder_gs": "ckpts/slat_decoder",
            }
        }
    }
    (snapshot / "pipeline.json").write_text(json.dumps(pipeline), encoding="utf-8")
    (snapshot / "ckpts").mkdir()
    for name in ("slat_flow", "slat_decoder"):
        (snapshot / "ckpts" / f"{name}.json").write_text("{}", encoding="utf-8")
        (snapshot / "ckpts" / f"{name}.safetensors").write_bytes(b"weights")

    resolved = resolve_local_hf_snapshot(
        "microsoft/TRELLIS-image-large",
        tmp_path,
        required_file="pipeline.json",
        validate_trellis_pipeline=True,
    )
    assert resolved == str(snapshot.resolve())


def test_trellis_pipeline_rejects_incomplete_local_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "local_pipeline"
    snapshot.mkdir()
    payload = {"args": {"models": {"slat_flow_model": "ckpts/missing"}}}
    (snapshot / "pipeline.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        resolve_local_hf_snapshot(
            str(snapshot),
            tmp_path,
            required_file="pipeline.json",
            validate_trellis_pipeline=True,
        )
    except FileNotFoundError as error:
        assert "incomplete model" in str(error)
    else:
        raise AssertionError("Incomplete TRELLIS pipeline was accepted")


def test_debug_npz_converts_bfloat16_for_numpy(tmp_path: Path) -> None:
    output = tmp_path / "debug.npz"
    save_npz(output, values=torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    import numpy as np

    with np.load(output) as payload:
        assert payload["values"].dtype.name == "float32"


def test_decoded_cross_view_consistency_backpropagates_to_surface() -> None:
    height = width = 16
    points = torch.tensor(
        [[[0.0, 0.0, 2.0], [0.1, -0.1, 2.2]]], requires_grad=True
    )
    K = torch.tensor([[[[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]]]])
    K = K.expand(1, 2, -1, -1).clone()
    w2c = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 2, -1, -1).clone()
    w2c[:, 1, 0, 3] = 0.2
    ramp = torch.linspace(0.0, 1.0, width).reshape(1, 1, 1, 1, width)
    images = ramp.expand(1, 2, 3, height, width).clone()
    features = torch.cat((images, 1.0 - images), dim=2)
    result = decoded_surface_cross_view_consistency(
        images=images,
        high_features=features,
        confidence=torch.ones(1, 2, 1, height, width),
        masks=torch.ones(1, 2, 1, height, width),
        intrinsics=K,
        world_to_camera=w2c,
        decoded_surface_points=points,
        decoded_surface_weights=torch.ones(1, 2),
        view_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        max_points=2,
    )
    (result["feature"] + result["rgb"]).backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert points.grad.abs().sum() > 0
