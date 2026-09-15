from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from geoss.datasets.dataset_stochastic_meshfleet import (
    StochasticMeshFleetDataset,
    stochastic_meshfleet_collate,
)
from geoss.integration.vggt_geometry_wrapper import VGGTGeometryBatch, VGGTGeometryWrapper
from geoss.losses.geometric_loss import FlowMatchingLossBuilder
from geoss.models.ss_flow_adapter import AffostructionSSFlow, SSFlowAdapter
from geoss.models.voxel_fusion_engine import (
    ConfidenceSparseVoxelFusion,
    VoxelFusionOutput,
    _maximum_interval_consensus_scale,
    _validate_alignment_retention,
    align_vggt_reference_to_dataset,
    unproject_depth_batched,
)
from geoss.ops.flow_matching import construct_flow_training_pair
from geoss.samplers.fast_ss_sampler import FastGeometryConditionedSSSampler
from geoss.utils.projection import project_points
from scripts.train_stochastic_voxel_ss_flow import (
    ResumableDistributedSampler,
    TrainableVoxelSSBranch,
    append_metrics,
)
from scripts.cache_affostruction_depth import _canonical_support_pixels


def _make_meshfleet_object(root: Path, views: int = 10, missing: tuple[int, ...] = ()) -> None:
    uid = "unit_object"
    render = root / "train" / "renders" / uid
    latent = root / "train" / "ss_latents"
    render.mkdir(parents=True)
    latent.mkdir(parents=True)
    frames = []
    for index in range(views):
        if index not in missing:
            rgba = np.zeros((16, 16, 4), dtype=np.uint8)
            rgba[..., :3] = index * 10
            rgba[..., 3] = 255
            Image.fromarray(rgba, "RGBA").save(render / f"{index:03d}.png")
        transform = np.eye(4)
        transform[0, 3] = index / 100.0
        frames.append(
            {
                "file_path": f"{index:03d}.png",
                "camera_angle_x": 0.7,
                "transform_matrix": transform.tolist(),
            }
        )
    (render / "transforms.json").write_text(
        json.dumps({"aabb": [[-0.5] * 3, [0.5] * 3], "scale": 1.0, "offset": [0, 0, 0], "frames": frames})
    )
    np.savez(latent / f"{uid}.npz", mean=np.zeros((8, 16, 16, 16), dtype=np.float32))


def test_dataset_stochastic_views_are_bounded_distinct_and_reproducible(tmp_path: Path):
    _make_meshfleet_object(tmp_path)
    first = StochasticMeshFleetDataset(str(tmp_path), min_views=1, max_views=8, seed=9, rank=2, image_size=16, load_gt_occupancy=False)
    second = StochasticMeshFleetDataset(str(tmp_path), min_views=1, max_views=8, seed=9, rank=2, image_size=16, load_gt_occupancy=False)
    first.set_epoch(3)
    second.set_epoch(3)
    a, b = first[0], second[0]
    assert 1 <= a["num_views"] <= 8
    assert torch.equal(a["view_ids"], b["view_ids"])
    assert a["view_ids"].unique().numel() == a["num_views"]
    epoch_selections = []
    for epoch in range(6):
        first.set_epoch(epoch)
        epoch_selections.append(tuple(first[0]["view_ids"].tolist()))
    assert len(set(epoch_selections)) > 1


def test_distributed_sampler_resume_cursor_preserves_remaining_order():
    dataset = list(range(17))
    sampler = ResumableDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=13, drop_last=True
    )
    sampler.set_epoch(4)
    complete = list(iter(sampler))
    sampler.set_start_index(3)

    assert list(iter(sampler)) == complete[3:]
    assert len(sampler) == len(complete) - 3


def test_dataset_gapped_render_ids_keep_image_camera_and_metadata_aligned(tmp_path: Path):
    _make_meshfleet_object(tmp_path, views=10, missing=(1, 4))
    dataset = StochasticMeshFleetDataset(
        str(tmp_path),
        min_views=8,
        max_views=8,
        seed=5,
        use_stochastic_views=False,
        image_size=16,
        load_gt_occupancy=False,
    )
    sample = dataset[0]

    assert sample["num_views"] == 8
    assert set(sample["view_ids"].tolist()) == {0, 2, 3, 5, 6, 7, 8, 9}
    assert torch.equal(sample["view_ids"], sample["view_metadata_indices"])
    assert sample["metadata"]["missing_frame_ids"] == ["001", "004"]
    assert sample["metadata"]["num_frames_total"] == 10
    assert sample["metadata"]["num_frames_available"] == 8
    for position, view_id in enumerate(sample["view_ids"].tolist()):
        expected_rgb = view_id * 10 / 255.0
        assert torch.allclose(
            sample["images"][position].mean(), torch.tensor(expected_rgb), atol=1e-6
        )
        assert sample["c2w_dataset"][position, 0, 3].item() == pytest.approx(
            view_id / 100.0
        )
        assert Path(sample["metadata"]["selected_frame_paths"][position]).stem == f"{view_id:03d}"
    batch = stochastic_meshfleet_collate([sample])
    assert torch.equal(batch["view_ids"][0], sample["view_ids"])
    assert torch.equal(batch["view_metadata_indices"][0], sample["view_metadata_indices"])


def test_depth_quality_manifest_preserves_exact_gapped_frame_alignment(tmp_path: Path):
    _make_meshfleet_object(tmp_path, views=10, missing=(1, 4))
    depth_dir = tmp_path / "depth" / "train" / "unit_object"
    depth_dir.mkdir(parents=True)
    valid_ids = ["002", "007", "009"]
    for frame_id in valid_ids:
        value = np.float32(int(frame_id) + 1)
        np.savez_compressed(
            depth_dir / f"{frame_id}.npz",
            depth_u16=np.ones((16, 16), dtype=np.uint16),
            depth_min=value,
            depth_max=value,
        )
    (depth_dir / "manifest.json").write_text(
        json.dumps(
            {
                "quality_protocol": "per_frame_rgb_depth_camera_intersection_v2",
                "status": "complete",
                "declared_frames": 8,
                "valid_frames": 3,
                "rejected_frames": 5,
                "valid_frame_ids": valid_ids,
            }
        ),
        encoding="utf-8",
    )
    dataset = StochasticMeshFleetDataset(
        str(tmp_path),
        min_views=3,
        max_views=3,
        seed=17,
        use_stochastic_views=False,
        image_size=16,
        load_gt_occupancy=False,
        depth_root=str(tmp_path / "depth"),
        require_depth=True,
    )
    sample = dataset[0]

    assert set(sample["view_ids"].tolist()) == {2, 7, 9}
    assert sample["metadata"]["depth_frame_ids"] == sample["metadata"]["selected_frame_ids"]
    assert sample["metadata"]["missing_frame_ids"] == ["001", "004"]
    assert sample["metadata"]["num_frames_available"] == 8
    assert sample["metadata"]["num_frames_verified"] == 3
    assert sample["metadata"]["quality_rejected_frames"] == 5
    for position, view_id in enumerate(sample["view_ids"].tolist()):
        assert torch.allclose(
            sample["depths"][position],
            torch.full((16, 16), float(view_id + 1)),
        )
        assert Path(sample["metadata"]["selected_frame_paths"][position]).stem == f"{view_id:03d}"


def test_depth_quality_requires_support_in_trellis_canonical_volume():
    depth = torch.full((1, 2, 2), 0.25)
    intrinsic = torch.tensor(
        [[[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]]
    )
    canonical_camera = torch.eye(4).unsqueeze(0)
    displaced_camera = canonical_camera.clone()
    displaced_camera[:, 0, 3] = 2.0

    assert _canonical_support_pixels(depth, intrinsic, canonical_camera).tolist() == [4]
    assert _canonical_support_pixels(depth, intrinsic, displaced_camera).tolist() == [0]


def test_alignment_reports_rays_that_no_scale_can_place_in_canonical_box():
    points = torch.tensor(
        [[[[[0.0, 2.0]], [[0.0, 0.0]], [[2.0, 0.0]]]]], dtype=torch.float32
    )
    vggt_w2c = torch.eye(4).view(1, 1, 4, 4)
    dataset_c2w = torch.eye(4).view(1, 1, 4, 4)
    dataset_c2w[..., 2, 3] = -2.0
    aligned, scale, alignment_inlier = align_vggt_reference_to_dataset(
        points,
        vggt_w2c,
        dataset_c2w,
        torch.zeros(1, 3),
        torch.full((1, 3), 0.5),
        torch.ones(1, 1, 1, 2, dtype=torch.bool),
        torch.ones(1, 1, 1, 2),
    )

    assert torch.isfinite(aligned).all()
    assert torch.isfinite(scale).all()
    assert alignment_inlier.flatten().tolist() == [True, False]


def test_partial_alignment_is_retained_but_zero_consensus_fails():
    _validate_alignment_retention(torch.tensor([0.9845559597015381]))
    _validate_alignment_retention(torch.tensor([0.8705096244812012]))
    _validate_alignment_retention(torch.tensor([0.01]))
    with pytest.raises(RuntimeError, match="retained no reliable geometry"):
        _validate_alignment_retention(torch.tensor([0.0]))


def test_scale_uses_maximum_interval_consensus_instead_of_all_ray_intersection():
    entry = torch.tensor([[1.0] * 99 + [5.0]])
    exit = torch.tensor([[2.0] * 99 + [6.0]])
    scale, fraction = _maximum_interval_consensus_scale(
        torch.tensor([1.5]), entry, exit, torch.ones_like(entry, dtype=torch.bool)
    )

    assert scale.item() == pytest.approx(1.5)
    assert fraction.item() == pytest.approx(0.99)


def _geometry_with_nonintersecting_pointmap() -> VGGTGeometryBatch:
    depth = torch.full((1, 1, 1, 2, 2), 2.0)
    point_map = torch.zeros(1, 1, 3, 2, 2)
    point_map[:, :, 0] = 10.0
    identity = torch.eye(4).view(1, 1, 4, 4)
    intrinsics = torch.tensor(
        [[[[10.0, 0.0, 0.5], [0.0, 10.0, 0.5], [0.0, 0.0, 1.0]]]]
    )
    return VGGTGeometryBatch(
        depth=depth,
        depth_confidence=torch.ones(1, 1, 2, 2),
        point_map=point_map,
        point_confidence=torch.ones(1, 1, 2, 2),
        intrinsics=intrinsics,
        extrinsics=identity,
        camera_to_world=identity,
        visual_features=torch.ones(1, 1, 2, 2, 2),
        valid_view_mask=torch.ones(1, 1, dtype=torch.bool),
        image_resolution=(2, 2),
        feature_resolution=(2, 2),
        patch_size=1,
    )


def _small_fusion() -> ConfidenceSparseVoxelFusion:
    return ConfidenceSparseVoxelFusion(
        input_feature_dim=2,
        projected_feature_dim=4,
        positional_frequencies=1,
        require_spconv=False,
        use_spconv_refinement=False,
    )


def test_nonintersecting_pointmap_falls_back_to_dataset_camera_depth():
    camera = torch.eye(4).view(1, 1, 4, 4)
    camera[..., 2, 3] = -2.0
    fused = _small_fusion()(
        _geometry_with_nonintersecting_pointmap(),
        foreground_masks=torch.ones(1, 1, 1, 2, 2),
        dataset_c2w=camera,
        canonical_center=torch.zeros(1, 3),
        canonical_half_extent=torch.full((1, 3), 0.5),
    )

    assert fused.alignment_mode == "dataset_camera_depth"
    assert fused.alignment_plausible_fraction.item() > 0
    assert fused.observation_mask.any()


class _DirectSSBackbone(torch.nn.Module):
    def __init__(self, condition_dim: int) -> None:
        super().__init__()
        self.cond_channels = condition_dim
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.seen_condition = None

    def forward(self, state, timestep, condition):
        del timestep
        self.seen_condition = condition
        return state * self.scale + condition.mean() * 0.0


def test_no_intersecting_geometry_uses_real_dino_fallback_and_ddp_safe_graph():
    camera = torch.eye(4).view(1, 1, 4, 4)
    camera[..., 0, 3] = 10.0
    camera[..., 2, 3] = -2.0
    fusion = _small_fusion()
    backbone = _DirectSSBackbone(fusion.condition_dim)
    flow = AffostructionSSFlow(
        backbone,
        condition_dim=fusion.condition_dim,
        classifier_free_dropout=0.0,
    )
    branch = TrainableVoxelSSBranch(fusion, flow)
    state = torch.randn(1, 8, 2, 2, 2)
    fallback = torch.randn(1, 3, fusion.condition_dim)
    fused, prediction, _ = branch(
        _geometry_with_nonintersecting_pointmap(),
        {
            "masks": torch.ones(1, 1, 1, 2, 2),
            "c2w_dataset": camera,
            "canonical_center": torch.zeros(1, 3),
            "canonical_half_extent": torch.full((1, 3), 0.5),
        },
        state,
        torch.zeros(1),
        fallback,
        torch.ones(1, 1, 2, 2, 2),
    )
    prediction.velocity.square().mean().backward()

    assert fused.alignment_mode == "base_only_no_canonical_rays"
    assert not fused.observation_mask.any()
    assert fused.alignment_plausible_fraction.item() == 0.0
    torch.testing.assert_close(backbone.seen_condition, fallback)
    assert all(parameter.grad is not None for parameter in fusion.parameters())
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in fusion.parameters())
    assert backbone.scale.grad is not None
    assert torch.count_nonzero(backbone.scale.grad) > 0


class _AlignedGeometryWithoutCondition(torch.nn.Module):
    """Reproduce the former geometry-available/zero-observation contradiction."""

    def __init__(self, condition_dim: int) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.projection_weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, geometry, **kwargs):
        del geometry, kwargs
        token_count = 4
        scalar = torch.zeros(1, token_count, 1)
        observation = torch.zeros(1, token_count, 1, dtype=torch.bool)
        return VoxelFusionOutput(
            sparse_tensor=None,
            dense_tokens=torch.zeros(1, token_count, self.condition_dim),
            voxel_indices=torch.empty(0, 4, dtype=torch.int32),
            voxel_xyz=torch.empty(0, 3),
            voxel_confidence=scalar,
            observation_mask=observation,
            accumulated_weight=scalar.clone(),
            observation_count=scalar.clone(),
            valid_mask=observation[..., 0],
            alignment_scale=torch.ones(1),
            condition_dim=self.condition_dim,
            alignment_plausible_fraction=torch.ones(1),
            geometry_available=torch.ones(1, dtype=torch.bool),
            conditioning_available=torch.zeros(1, dtype=torch.bool),
        )


def test_aligned_but_zero_condition_uses_real_fallback_and_keeps_ddp_graph():
    fusion = _AlignedGeometryWithoutCondition(condition_dim=5)
    backbone = _DirectSSBackbone(condition_dim=5)
    flow = AffostructionSSFlow(
        backbone,
        condition_dim=5,
        classifier_free_dropout=0.0,
    )
    branch = TrainableVoxelSSBranch(fusion, flow)
    state = torch.randn(1, 3, 2, 2, 2)
    fallback = torch.randn(1, 4, 5)
    fused, prediction, _ = branch(
        object(),
        {
            "masks": torch.ones(1, 1, 1, 1, 1),
            "c2w_dataset": torch.eye(4).view(1, 1, 4, 4),
            "canonical_center": torch.zeros(1, 3),
            "canonical_half_extent": torch.ones(1, 3),
        },
        state,
        torch.zeros(1),
        fallback,
        torch.randn(1, 1, 2, 1, 1),
    )
    prediction.velocity.square().mean().backward()

    assert fused.geometry_available.item()
    assert not fused.conditioning_available.item()
    torch.testing.assert_close(backbone.seen_condition, fallback)
    assert all(parameter.grad is not None for parameter in branch.parameters())
    assert torch.count_nonzero(fusion.projection_weight.grad) == 0
    assert torch.count_nonzero(backbone.scale.grad) > 0


def test_all_invalid_foundation_geometry_is_an_explicit_base_only_state():
    geometry = _geometry_with_nonintersecting_pointmap()
    geometry = VGGTGeometryBatch(
        **{
            **geometry.__dict__,
            "depth_valid_mask": torch.zeros(1, 1, 2, 2, dtype=torch.bool),
            "point_valid_mask": torch.zeros(1, 1, 2, 2, dtype=torch.bool),
            "camera_valid_mask": torch.zeros(1, 1, dtype=torch.bool),
        }
    )
    fused = _small_fusion()(
        geometry,
        foreground_masks=torch.ones(1, 1, 1, 2, 2),
        dataset_K=torch.eye(3).view(1, 1, 3, 3),
        dataset_c2w=torch.eye(4).view(1, 1, 4, 4),
        canonical_center=torch.zeros(1, 3),
        canonical_half_extent=torch.full((1, 3), 0.5),
    )

    assert fused.geometry_status == "base_only"
    assert not fused.geometry_available.any()
    assert fused.geometry_quality.item() == 0.0
    assert fused.depth_supervision_weight.sum().item() == 0.0


class _NegativeOccupancyDecoder(torch.nn.Module):
    def forward(self, clean_grid):
        return clean_grid[:, :1] - 10.0


class _NonfiniteOccupancyDecoder(torch.nn.Module):
    def forward(self, clean_grid):
        return clean_grid[:, :1] * torch.tensor(float("nan"), device=clean_grid.device)


def _minimal_loss_inputs():
    v_final = torch.zeros(1, 8, 8, requires_grad=True)
    v_target = torch.ones_like(v_final)
    v_base = torch.zeros_like(v_final)
    predicted_clean = torch.zeros(1, 8, 2, 2, 2, requires_grad=True)
    return {
        "v_final": v_final,
        "v_target": v_target,
        "v_base": v_base,
        "gate": torch.ones(1, 8, 1),
        "predicted_clean_grid": predicted_clean,
        "gt_occ": torch.ones(1, 1, 2, 2, 2),
    }


def test_empty_active_decoder_support_is_a_trainable_prediction_not_an_error():
    builder = FlowMatchingLossBuilder(
        use_depth_loss=False,
        use_silhouette_loss=False,
        use_prior_preservation=False,
    )
    inputs = _minimal_loss_inputs()
    losses = builder(ss_decoder=_NegativeOccupancyDecoder(), **inputs)
    losses["loss_total"].backward()

    assert torch.isfinite(losses["loss_total"])
    assert losses["diagnostics"]["decoder_empty_active"] is True
    assert losses["loss_occupancy"] > 0
    assert torch.isfinite(inputs["predicted_clean_grid"].grad).all()
    assert torch.count_nonzero(inputs["predicted_clean_grid"].grad) > 0


def test_nonfinite_decoder_output_disables_only_decoder_losses_without_nan_gradients():
    builder = FlowMatchingLossBuilder(
        use_depth_loss=False,
        use_silhouette_loss=False,
        use_prior_preservation=False,
    )
    inputs = _minimal_loss_inputs()
    losses = builder(ss_decoder=_NonfiniteOccupancyDecoder(), **inputs)
    losses["loss_total"].backward()

    assert torch.isfinite(losses["loss_total"])
    assert losses["diagnostics"]["decoder_numerically_valid"] is False
    assert losses["loss_occupancy"].item() == 0.0
    assert torch.isfinite(inputs["v_final"].grad).all()
    assert inputs["predicted_clean_grid"].grad is None


def test_resumed_metrics_csv_keeps_existing_schema(tmp_path: Path):
    jsonl_path = tmp_path / "metrics.jsonl"
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("step,loss_total\n1,2.0\n", encoding="utf-8")

    append_metrics(
        jsonl_path,
        csv_path,
        {"step": 2, "loss_total": 1.5, "alignment_plausible_percent": 98.5},
    )

    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "step,loss_total",
        "1,2.0",
        "2,1.5",
    ]
    assert "alignment_plausible_percent" in jsonl_path.read_text(encoding="utf-8")


def test_project_unproject_roundtrip():
    depth = torch.full((1, 1, 1, 3, 4), 2.0)
    K = torch.tensor([[[[4.0, 0.0, 1.5], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]]]])
    c2w = torch.eye(4).view(1, 1, 4, 4)
    points = unproject_depth_batched(depth, K, c2w)[0, 0].permute(1, 2, 0).reshape(-1, 3)
    projection = project_points(points, K[0, 0], torch.eye(4))
    y, x = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    expected = torch.stack([x, y], dim=-1).reshape(-1, 2).float()
    assert torch.allclose(projection["uv"], expected, atol=1e-5)
    assert torch.allclose(projection["depth"], torch.full((12, 1), 2.0), atol=1e-5)


def test_weighted_duplicate_voxel_reduction_and_observation_mask():
    fusion = ConfidenceSparseVoxelFusion(
        input_feature_dim=2,
        projected_feature_dim=2,
        positional_frequencies=1,
        require_spconv=False,
        use_spconv_refinement=False,
    )
    fusion.feature_projection = torch.nn.Identity()
    fusion.condition_dim = 2 + 3 + 6
    points = torch.tensor([[[[[-0.9, -0.9]], [[-0.9, -0.9]], [[-0.9, -0.9]]]]])
    # Reshape as [B,V,3,H,W], two view observations in one voxel.
    points = torch.tensor([[[[[-0.9]], [[-0.9]], [[-0.9]]], [[[ -0.9]], [[-0.9]], [[-0.9]]]]])
    features = torch.tensor([[[[[2.0]], [[4.0]]], [[[10.0]], [[14.0]]]]])
    confidence = torch.tensor([[[0.25], [0.75]]])
    weights = confidence.clone()
    valid = torch.ones(1, 2, 1, 1, dtype=torch.bool)
    output = fusion._reduce_to_voxels(points, features, confidence, weights, valid, torch.ones(1, 2, dtype=torch.bool), torch.ones(1))
    expected = torch.tensor([(0.25 * 2 + 0.75 * 10), (0.25 * 4 + 0.75 * 14)])
    assert torch.allclose(output.dense_tokens[0, 0, :2], expected, atol=1e-6)
    assert bool(output.observation_mask[0, 0, 0])
    assert not bool(output.observation_mask[0, 1, 0])
    assert output.observation_count[0, 0, 0] == 2


def test_positive_weak_evidence_remains_observed_below_legacy_absolute_cutoff():
    fusion = ConfidenceSparseVoxelFusion(
        input_feature_dim=2,
        projected_feature_dim=2,
        positional_frequencies=1,
        observation_threshold=1e-4,
        require_spconv=False,
        use_spconv_refinement=False,
    )
    fusion.feature_projection = torch.nn.Identity()
    fusion.condition_dim = 2 + 3 + 6
    points = torch.full((1, 1, 3, 1, 1), -0.9)
    features = torch.tensor([[[[[2.0]], [[4.0]]]]])
    confidence = torch.full((1, 1, 1, 1), 1e-6)
    weights = confidence.clone()
    valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)

    output = fusion._reduce_to_voxels(
        points,
        features,
        confidence,
        weights,
        valid,
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1),
    )

    assert output.accumulated_weight.max().item() < fusion.observation_threshold
    assert output.observation_mask.any()
    assert output.conditioning_available.item()
    assert output.weak_evidence_only.item()


def test_zero_init_and_spatial_gate_invariants():
    adapter = SSFlowAdapter(latent_dim=3, condition_dim=5, hidden_dim=16, num_heads=4, num_blocks=1, trust_region=10.0)
    x = torch.randn(1, 4, 3)
    cond = torch.randn(1, 4, 5)
    base = torch.randn_like(x)
    observed = torch.ones(1, 4, 1, dtype=torch.bool)
    confidence = torch.ones(1, 4, 1)
    initialized = adapter(x, cond, torch.zeros(1), observed, confidence, v_base=base)
    assert torch.equal(initialized.v_final, base)
    torch.nn.init.zeros_(adapter.residual_head.weight)
    torch.nn.init.constant_(adapter.residual_head.bias, 0.2)
    unobserved = adapter(x, cond, torch.zeros(1), torch.zeros_like(observed), confidence, v_base=base, enabled=False)
    assert torch.equal(unobserved.v_final, base)
    full = adapter(x, cond, torch.zeros(1), observed, confidence, v_base=base)
    assert torch.allclose(full.v_final, base + full.delta_v_geo, atol=1e-6)


def test_enabled_adapter_without_observations_is_exact_base_and_backward_safe():
    adapter = SSFlowAdapter(
        latent_dim=3,
        condition_dim=5,
        hidden_dim=16,
        num_heads=4,
        num_blocks=1,
    )
    base = torch.randn(1, 4, 3)
    output = adapter(
        torch.randn_like(base),
        torch.randn(1, 4, 5),
        torch.zeros(1),
        torch.zeros(1, 4, 1, dtype=torch.bool),
        torch.zeros(1, 4, 1),
        v_base=base,
    )

    assert torch.equal(output.v_final, base)
    assert output.v_final.requires_grad
    output.v_final.sum().backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in adapter.parameters())


def test_trellis_flow_pair_matches_sigma_min_equation():
    x0 = torch.randn(2, 8, 2, 2, 2)
    noise = torch.randn_like(x0)
    t = torch.tensor([0.2, 0.8])
    xt, target, backend = construct_flow_training_pair(x0, noise, t, 1e-5, backend="torch")
    view = t.view(2, 1, 1, 1, 1)
    assert torch.allclose(xt, (1 - view) * x0 + (1e-5 + (1 - 1e-5) * view) * noise)
    assert torch.allclose(target, (1 - 1e-5) * noise - x0)
    assert backend == "torch"


class _ConstantBase(torch.nn.Module):
    def forward(self, x, t, cond):
        return torch.full_like(x, 0.125)


def test_five_step_sampler_is_deterministic():
    adapter = SSFlowAdapter(latent_dim=8, condition_dim=5, hidden_dim=16, num_heads=4, num_blocks=1)
    sampler = FastGeometryConditionedSSSampler(_ConstantBase(), adapter, num_steps=5, rescale_t=3.0)
    noise = torch.randn(1, 8, 16, 16, 16, generator=torch.Generator().manual_seed(4))
    fusion = VoxelFusionOutput(
        sparse_tensor=None,
        dense_tokens=torch.zeros(1, 4096, 5),
        voxel_indices=torch.empty(0, 4, dtype=torch.int32),
        voxel_xyz=torch.empty(0, 3),
        voxel_confidence=torch.ones(1, 4096, 1),
        observation_mask=torch.ones(1, 4096, 1, dtype=torch.bool),
        accumulated_weight=torch.ones(1, 4096, 1),
        observation_count=torch.ones(1, 4096, 1),
        valid_mask=torch.ones(1, 4096, dtype=torch.bool),
        alignment_scale=torch.ones(1),
        condition_dim=5,
    )
    condition = torch.ones(1, 3, 1024)
    first = sampler.sample(noise.clone(), base_condition=condition, voxel_fusion=fusion, adapter_enabled=False)
    second = sampler.sample(noise.clone(), base_condition=condition, voxel_fusion=fusion, adapter_enabled=False)
    assert torch.equal(first.samples, second.samples)


def test_production_vggt_rejects_mock():
    with pytest.raises(RuntimeError, match="Production VGGT"):
        VGGTGeometryWrapper(mock=True, require_real=True)


def test_production_trainer_has_no_mock_or_random_condition_tokens():
    source = (Path(__file__).parents[1] / "scripts" / "train_stochastic_voxel_ss_flow.py").read_text()
    assert "MockSSFlow" not in source
    assert "random condition" not in source.lower()
    assert "pipeline.encode_image(all_views)" in source
    assert "views * tokens" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP smoke requires a GPU")
def test_bf16_adapter_forward_backward_is_finite():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is unavailable")
    device = torch.device("cuda")
    adapter = SSFlowAdapter(latent_dim=8, condition_dim=9, hidden_dim=32, num_heads=4, num_blocks=1).to(device)
    torch.nn.init.normal_(adapter.residual_head.weight, std=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = adapter(
            torch.randn(1, 16, 8, device=device),
            torch.randn(1, 16, 9, device=device),
            torch.tensor([500.0], device=device),
            torch.ones(1, 16, 1, device=device, dtype=torch.bool),
            torch.ones(1, 16, 1, device=device),
            v_base=torch.randn(1, 16, 8, device=device),
        )
        loss = output.v_final.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in adapter.parameters())
