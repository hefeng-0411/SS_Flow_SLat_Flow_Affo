"""Confidence-aware CUDA voxel fusion aligned to TRELLIS sparse structure."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoss.integration.vggt_geometry_wrapper import VGGTGeometryBatch
from geoss.models.affostruction_conditioning import VoxelPositionalEncoding3D


class NoCanonicalRayIntersectionError(RuntimeError):
    """No reliable predicted ray supports the known canonical object box."""


@dataclass(frozen=True)
class VoxelFusionOutput:
    sparse_tensor: Any
    dense_tokens: torch.Tensor
    voxel_indices: torch.Tensor
    voxel_xyz: torch.Tensor
    voxel_confidence: torch.Tensor
    observation_mask: torch.Tensor
    accumulated_weight: torch.Tensor
    observation_count: torch.Tensor
    valid_mask: torch.Tensor
    alignment_scale: torch.Tensor
    condition_dim: int
    alignment_plausible_fraction: Optional[torch.Tensor] = None
    alignment_mode: str = "pointmap_reference"
    geometry_available: Optional[torch.Tensor] = None
    conditioning_available: Optional[torch.Tensor] = None
    weak_evidence_only: Optional[torch.Tensor] = None
    geometry_quality: Optional[torch.Tensor] = None
    geometry_status: str = "usable"
    geometry_reason: str = ""
    aligned_depth: Optional[torch.Tensor] = None
    depth_supervision_weight: Optional[torch.Tensor] = None
    timings_ms: Dict[str, Optional[float]] = field(default_factory=dict)


class ConfidenceSparseVoxelFusion(nn.Module):
    """Fuse aligned VGGT samples with weighted CUDA sort/scatter reduction.

    Duplicate reduction is performed explicitly before construction of the
    ``spconv.SparseConvTensor`` because spconv does not define weighted
    duplicate-coordinate semantics.
    """

    def __init__(
        self,
        *,
        grid_resolution: int = 16,
        input_feature_dim: int = 2048,
        projected_feature_dim: int = 256,
        positional_frequencies: int = 4,
        observation_threshold: float = 1e-4,
        fusion_mode: str = "confidence",
        use_vggt_depth: bool = True,
        use_vggt_pointmap: bool = True,
        use_confidence_weighting: bool = True,
        use_visibility_weighting: bool = True,
        use_3d_positional_encoding: bool = True,
        require_spconv: bool = True,
        use_spconv_refinement: bool = True,
        conditioning_layout: str = "legacy",
    ) -> None:
        super().__init__()
        if grid_resolution != 16:
            raise ValueError("Production TRELLIS SS fusion requires grid_resolution=16")
        if fusion_mode not in {"confidence", "average_ablation"}:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}")
        if not use_vggt_pointmap and not use_vggt_depth:
            raise ValueError("At least one real VGGT geometry source must be enabled")
        if conditioning_layout not in {"legacy", "affostruction"}:
            raise ValueError(f"Unsupported conditioning_layout={conditioning_layout!r}")
        self.grid_resolution = int(grid_resolution)
        self.input_feature_dim = int(input_feature_dim)
        self.projected_feature_dim = int(projected_feature_dim)
        self.positional_frequencies = int(positional_frequencies)
        self.observation_threshold = float(observation_threshold)
        self.fusion_mode = fusion_mode
        self.use_vggt_depth = bool(use_vggt_depth)
        self.use_vggt_pointmap = bool(use_vggt_pointmap)
        self.use_confidence_weighting = bool(use_confidence_weighting)
        self.use_visibility_weighting = bool(use_visibility_weighting)
        self.use_3d_positional_encoding = bool(use_3d_positional_encoding)
        self.require_spconv = bool(require_spconv)
        self.conditioning_layout = conditioning_layout
        if conditioning_layout == "affostruction":
            projection_layers: list[nn.Module] = [nn.LayerNorm(input_feature_dim)]
            if input_feature_dim != projected_feature_dim:
                projection_layers.extend(
                    [nn.Linear(input_feature_dim, projected_feature_dim), nn.LayerNorm(projected_feature_dim)]
                )
            self.feature_projection = nn.Sequential(*projection_layers)
            self.positional_dim = projected_feature_dim if use_3d_positional_encoding else 0
            self.condition_dim = projected_feature_dim
            self.affostruction_position = (
                VoxelPositionalEncoding3D(projected_feature_dim, grid_resolution)
                if use_3d_positional_encoding
                else None
            )
        else:
            self.feature_projection = nn.Sequential(
                nn.LayerNorm(input_feature_dim),
                nn.Linear(input_feature_dim, projected_feature_dim),
                nn.SiLU(),
            )
            self.positional_dim = 6 * positional_frequencies if use_3d_positional_encoding else 0
            self.condition_dim = projected_feature_dim + 3 + self.positional_dim
            self.affostruction_position = None
        self.sparse_refiner = None
        if use_spconv_refinement:
            spconv = _import_spconv(required=require_spconv)
            if spconv is not None:
                self.sparse_refiner = spconv.SubMConv3d(
                    self.condition_dim,
                    self.condition_dim,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    indice_key="ss_voxel_condition",
                )

    def _reduce_to_voxels(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        confidence: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
        view_valid_mask: torch.Tensor,
        alignment_scale: torch.Tensor,
        profile: bool = False,
    ) -> VoxelFusionOutput:
        B, V, C, H, W = features.shape
        R = self.grid_resolution
        # [B,V,H,W,3], where channels are explicitly xyz.
        point_samples = points.permute(0, 1, 3, 4, 2)
        feature_samples = features.permute(0, 1, 3, 4, 2)
        indices = torch.floor((point_samples + 1.0) * (R / 2.0)).to(torch.long).clamp(0, R - 1)
        batch_ids = torch.arange(B, device=points.device).view(B, 1, 1, 1).expand(B, V, H, W)
        view_ids = torch.arange(V, device=points.device).view(1, V, 1, 1).expand(B, V, H, W)
        keys = indices[..., 0] + R * indices[..., 1] + R * R * indices[..., 2] + R**3 * batch_ids

        flat_valid = valid.reshape(-1)
        keys = keys.reshape(-1)[flat_valid]
        sample_features = feature_samples.reshape(-1, C)[flat_valid]
        sample_weights = weights.reshape(-1)[flat_valid].float()
        sample_confidence = confidence.reshape(-1)[flat_valid].float()
        sample_views = view_ids.reshape(-1)[flat_valid]
        if keys.numel() == 0:
            raise RuntimeError("No valid samples survived voxel indexing")

        reduction_timer = _CudaTimer(points.device, profile)
        reduction_timer.start()
        order = torch.argsort(keys)
        sorted_keys = keys[order]
        unique_keys, inverse = torch.unique_consecutive(sorted_keys, return_inverse=True)
        sorted_weights = sample_weights[order]
        weighted_features = sample_features[order].float() * sorted_weights[:, None]
        accumulated = torch.zeros(unique_keys.numel(), device=points.device, dtype=torch.float32)
        accumulated.index_add_(0, inverse, sorted_weights)
        feature_sum = torch.zeros(unique_keys.numel(), C, device=points.device, dtype=torch.float32)
        feature_sum.index_add_(0, inverse, weighted_features)
        fused_features = feature_sum / accumulated[:, None].clamp_min(1e-8)

        max_confidence = torch.zeros(unique_keys.numel(), device=points.device, dtype=torch.float32)
        max_confidence.scatter_reduce_(0, inverse, sample_confidence[order], reduce="amax", include_self=False)
        sorted_views = sample_views[order]
        key_view = sorted_keys * V + sorted_views
        unique_key_view = torch.unique_consecutive(torch.sort(key_view).values)
        key_for_view = torch.div(unique_key_view, V, rounding_mode="floor")
        voxel_for_view = torch.searchsorted(unique_keys, key_for_view)
        observation_count = torch.zeros(unique_keys.numel(), device=points.device, dtype=torch.float32)
        observation_count.index_add_(0, voxel_for_view, torch.ones_like(voxel_for_view, dtype=torch.float32))
        reduction_ms = reduction_timer.stop()

        local_keys = unique_keys.remainder(R**3)
        batch_index = torch.div(unique_keys, R**3, rounding_mode="floor")
        x_index = local_keys.remainder(R)
        y_index = torch.div(local_keys, R, rounding_mode="floor").remainder(R)
        z_index = torch.div(local_keys, R * R, rounding_mode="floor")
        xyz_index = torch.stack([x_index, y_index, z_index], dim=-1)
        xyz = (xyz_index.float() + 0.5) * (2.0 / R) - 1.0
        valid_views = view_valid_mask.sum(dim=1).float()[batch_index].clamp_min(1)
        observation_fraction = (observation_count / valid_views).clamp(0, 1)
        # ``accumulated`` is already the sum of finite, strictly-positive
        # confidence weights selected above.  Its magnitude expresses quality;
        # it must not also act as an absolute availability threshold.  VGGT
        # confidence is sample-dependent, so a fixed 1e-4 cutoff could turn a
        # valid but weakly-confident object into an entirely unobserved one.
        observation_mask_sparse = torch.isfinite(accumulated) & (accumulated > 0)
        # Retain the old cutoff only as an observability diagnostic.  This lets
        # us identify samples that would have failed under the former policy
        # without discarding their geometric evidence.
        strong_observation_sparse = accumulated > max(self.observation_threshold, 0.0)
        observed_per_batch = torch.zeros(B, device=points.device, dtype=torch.long)
        observed_per_batch.index_add_(
            0, batch_index, observation_mask_sparse.to(dtype=torch.long)
        )
        strong_per_batch = torch.zeros(B, device=points.device, dtype=torch.long)
        strong_per_batch.index_add_(
            0, batch_index, strong_observation_sparse.to(dtype=torch.long)
        )
        conditioning_available = observed_per_batch > 0
        weak_evidence_only = conditioning_available & (strong_per_batch == 0)
        projected = self.feature_projection(fused_features)
        if self.conditioning_layout == "affostruction":
            position = (
                self.affostruction_position(xyz_index).to(projected)
                if self.affostruction_position is not None
                else torch.zeros_like(projected)
            )
            sparse_features = projected + position
        else:
            position = sinusoidal_3d_encoding(xyz, self.positional_frequencies) if self.use_3d_positional_encoding else xyz.new_empty(xyz.shape[0], 0)
            sparse_features = torch.cat(
                [projected, max_confidence[:, None], observation_fraction[:, None], observation_mask_sparse.float()[:, None], position],
                dim=-1,
            )
        spconv_indices = torch.stack([batch_index, z_index, y_index, x_index], dim=-1).to(torch.int32).contiguous()
        spconv_timer = _CudaTimer(points.device, profile)
        spconv_timer.start()
        sparse_tensor = _make_sparse_tensor(
            sparse_features,
            spconv_indices,
            batch_size=B,
            resolution=R,
            required=self.require_spconv,
        )
        if self.sparse_refiner is not None:
            refined = self.sparse_refiner(sparse_tensor)
            sparse_features = sparse_features + refined.features
            sparse_tensor = sparse_tensor.replace_feature(sparse_features)
        spconv_ms = spconv_timer.stop()

        dense = sparse_features.new_zeros(B, R**3, self.condition_dim)
        dense[batch_index, local_keys] = sparse_features
        dense_conf = accumulated.new_zeros(B, R**3, 1)
        dense_conf[batch_index, local_keys, 0] = max_confidence
        dense_obs = torch.zeros(B, R**3, 1, device=points.device, dtype=torch.bool)
        dense_obs[batch_index, local_keys, 0] = observation_mask_sparse
        dense_weight = accumulated.new_zeros(B, R**3, 1)
        dense_weight[batch_index, local_keys, 0] = accumulated
        dense_count = observation_count.new_zeros(B, R**3, 1)
        dense_count[batch_index, local_keys, 0] = observation_count

        return VoxelFusionOutput(
            sparse_tensor=sparse_tensor,
            dense_tokens=dense,
            voxel_indices=spconv_indices,
            voxel_xyz=xyz,
            voxel_confidence=dense_conf,
            observation_mask=dense_obs,
            accumulated_weight=dense_weight,
            observation_count=dense_count,
            valid_mask=dense_obs[..., 0],
            alignment_scale=alignment_scale,
            condition_dim=self.condition_dim,
            conditioning_available=conditioning_available,
            weak_evidence_only=weak_evidence_only,
            timings_ms={"voxel_reduction": reduction_ms, "spconv_construction_refinement": spconv_ms},
        )

    def forward(self, *args, **kwargs):  # type: ignore[override]
        output = self._forward_impl(*args, **kwargs)
        return output

    def _forward_impl(
        self,
        geometry: VGGTGeometryBatch,
        *,
        foreground_masks: torch.Tensor,
        dataset_K: Optional[torch.Tensor] = None,
        dataset_c2w: torch.Tensor,
        canonical_center: torch.Tensor,
        canonical_half_extent: torch.Tensor,
        points_are_trellis_normalized: bool = False,
        profile: bool = False,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> VoxelFusionOutput:
        # This implementation is split only so alignment scale can be attached
        # without changing the public immutable dataclass contract.
        feature_source = geometry.visual_features if spatial_features is None else spatial_features
        features = torch.nan_to_num(
            feature_source.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        B, V, C, Hf, Wf = features.shape
        if C != self.input_feature_dim:
            raise ValueError(f"Spatial feature width mismatch: configured={self.input_feature_dim}, actual={C}")
        if foreground_masks.shape[:2] != (B, V):
            raise ValueError(f"foreground_masks must start [B,V], got {tuple(foreground_masks.shape)}")
        _assert_alignment_inputs(dataset_c2w, canonical_center, canonical_half_extent, B, V)
        if dataset_K is not None and dataset_K.shape != (B, V, 3, 3):
            raise ValueError(f"dataset_K must be [B,V,3,3], got {tuple(dataset_K.shape)}")
        geometry_timer = _CudaTimer(features.device, profile)
        geometry_timer.start()
        masks = _resize_bv_map(foreground_masks.float(), (Hf, Wf)).clamp(0, 1)
        mask_interior = (
            F.max_pool2d(
                (1.0 - masks[:, :, 0]).reshape(B * V, 1, Hf, Wf),
                kernel_size=3,
                stride=1,
                padding=1,
            ).reshape(B, V, Hf, Wf)
            < 0.5
        )
        depth = _resize_bv_map(
            torch.nan_to_num(geometry.depth.float(), nan=0.0, posinf=0.0, neginf=0.0),
            (Hf, Wf),
        )
        point_map = _resize_bv_map(
            torch.nan_to_num(geometry.point_map.float(), nan=0.0, posinf=0.0, neginf=0.0),
            (Hf, Wf),
        )
        point_conf = _resize_bv_map(geometry.point_confidence[:, :, None].float(), (Hf, Wf))[:, :, 0]
        depth_conf = _resize_bv_map(geometry.depth_confidence[:, :, None].float(), (Hf, Wf))[:, :, 0]
        source_hw = (
            tuple(int(value) for value in geometry.image_resolution)
            if dataset_K is None
            else tuple(int(value) for value in foreground_masks.shape[-2:])
        )
        authoritative_K = geometry.intrinsics.float() if dataset_K is None else dataset_K.float()
        K_feature = rescale_intrinsics(authoritative_K, source_hw, (Hf, Wf))
        view_valid = geometry.valid_view_mask[:, :, None, None]
        depth_valid = view_valid & torch.isfinite(depth[:, :, 0]) & (depth[:, :, 0] > 0)
        point_valid = view_valid & torch.isfinite(point_map).all(dim=2)
        if geometry.depth_valid_mask is not None:
            depth_valid &= _resize_validity_mask(geometry.depth_valid_mask, (Hf, Wf))
        if geometry.point_valid_mask is not None:
            point_valid &= _resize_validity_mask(geometry.point_valid_mask, (Hf, Wf))
        camera_valid = geometry.valid_view_mask
        if geometry.camera_valid_mask is not None:
            camera_valid &= geometry.camera_valid_mask.to(device=features.device, dtype=torch.bool)
        foreground = masks[:, :, 0] > 0
        point_valid &= foreground
        depth_valid &= foreground
        point_conf = robust_confidence_probability(point_conf, point_valid)
        depth_conf = robust_confidence_probability(depth_conf, depth_valid)

        if points_are_trellis_normalized:
            normalized_points = point_map
            alignment_scale = torch.ones(B, device=point_map.device, dtype=torch.float32)
            alignment_inlier = point_valid & torch.isfinite(point_map).all(dim=2)
            selected_confidence = point_conf
            selected_valid = point_valid
            alignment_mode = "provided_trellis"
            _, normalized_points, plausible, alignable_fraction = _score_alignment_candidate(
                point_map,
                alignment_inlier,
                point_valid,
                point_conf,
                masks[:, :, 0],
                mask_interior,
                canonical_center.float(),
                canonical_half_extent.float(),
                points_are_normalized=True,
            )
        else:
            point_candidate = None
            depth_candidate = None
            if self.use_vggt_pointmap and bool((point_valid & camera_valid[:, :, None, None]).any()):
                point_alignment_valid = point_valid & camera_valid[:, :, None, None]
                interior_valid = point_alignment_valid & mask_interior
                has_interior = interior_valid.flatten(1).any(dim=1).view(B, 1, 1, 1)
                alignment_valid = torch.where(has_interior, interior_valid, point_alignment_valid)
                try:
                    aligned, scale, inlier = align_vggt_reference_to_dataset(
                        point_map,
                        geometry.extrinsics.float(),
                        dataset_c2w.float(),
                        canonical_center.float(),
                        canonical_half_extent.float(),
                        alignment_valid,
                        point_conf,
                    )
                    point_candidate = _score_alignment_candidate(
                        aligned,
                        inlier,
                        point_alignment_valid,
                        point_conf,
                        masks[:, :, 0],
                        mask_interior,
                        canonical_center.float(),
                        canonical_half_extent.float(),
                    ) + (scale, inlier)
                except NoCanonicalRayIntersectionError:
                    point_candidate = None

            if self.use_vggt_depth and bool(depth_valid.any()):
                depth_points = unproject_depth_batched(depth, K_feature, dataset_c2w.float())
                interior_valid = depth_valid & mask_interior
                has_interior = interior_valid.flatten(1).any(dim=1).view(B, 1, 1, 1)
                alignment_valid = torch.where(has_interior, interior_valid, depth_valid)
                try:
                    aligned, scale, inlier = align_depth_rays_to_dataset(
                        depth_points,
                        dataset_c2w.float(),
                        canonical_center.float(),
                        canonical_half_extent.float(),
                        alignment_valid,
                        depth_conf,
                    )
                    depth_candidate = _score_alignment_candidate(
                        aligned,
                        inlier,
                        depth_valid,
                        depth_conf,
                        masks[:, :, 0],
                        mask_interior,
                        canonical_center.float(),
                        canonical_half_extent.float(),
                    ) + (scale, inlier)
                except NoCanonicalRayIntersectionError:
                    depth_candidate = None

            if point_candidate is None and depth_candidate is None:
                geometry_ms = geometry_timer.stop()
                return self._empty_fusion_output(
                    features,
                    alignment_mode="base_only_no_canonical_rays",
                    geometry_reason="neither point-map nor depth rays support the canonical volume",
                    geometry_ms=geometry_ms,
                )
            if point_candidate is None:
                use_depth = torch.ones(B, device=features.device, dtype=torch.bool)
            elif depth_candidate is None:
                use_depth = torch.zeros(B, device=features.device, dtype=torch.bool)
            else:
                # Both heads are hypotheses. Select the one retaining more
                # confidence-weighted, camera-consistent foreground evidence.
                use_depth = depth_candidate[3] > point_candidate[3]

            template = point_candidate if point_candidate is not None else depth_candidate
            assert template is not None
            if point_candidate is None:
                point_candidate = template
            if depth_candidate is None:
                depth_candidate = template
            choose = use_depth[:, None, None, None, None]
            aligned = torch.where(choose, depth_candidate[0], point_candidate[0])
            normalized_points = torch.where(choose, depth_candidate[1], point_candidate[1])
            plausible = torch.where(use_depth[:, None, None, None], depth_candidate[2], point_candidate[2])
            alignable_fraction = torch.where(use_depth, depth_candidate[3], point_candidate[3])
            alignment_scale = torch.where(use_depth, depth_candidate[4], point_candidate[4])
            alignment_inlier = torch.where(
                use_depth[:, None, None, None], depth_candidate[5], point_candidate[5]
            )
            selected_confidence = torch.where(
                use_depth[:, None, None, None], depth_conf, point_conf
            )
            selected_valid = torch.where(
                use_depth[:, None, None, None], depth_valid, point_valid
            )
            if bool(use_depth.all()):
                alignment_mode = "dataset_camera_depth"
            elif bool((~use_depth).all()):
                alignment_mode = "pointmap_reference"
            else:
                alignment_mode = "mixed_pointmap_depth"

        geometry_ms = geometry_timer.stop()
        if bool((alignable_fraction <= 0).any()) or not torch.isfinite(alignable_fraction).all():
            return self._empty_fusion_output(
                features,
                alignment_mode="base_only_no_alignment_consensus",
                geometry_reason="alignment candidates retained zero reliable foreground evidence",
                geometry_ms=geometry_ms,
            )
        inside_lattice = (
            torch.isfinite(normalized_points).all(dim=2)
            & (normalized_points.abs() <= 1.0).all(dim=2)
        )
        visibility = inside_lattice
        alignment_quality = alignable_fraction[:, None, None, None]
        effective_confidence = selected_confidence * alignment_quality
        confidence_weight = (
            torch.ones_like(selected_confidence)
            if self.fusion_mode == "average_ablation" or not self.use_confidence_weighting
            else selected_confidence
        )
        weights = (
            confidence_weight
            * alignment_quality
            * masks[:, :, 0]
            * visibility.float()
            * geometry.valid_view_mask[:, :, None, None]
        )
        valid = selected_valid & plausible & alignment_inlier & (weights > 0)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        if bool((weights.flatten(1).sum(dim=1) <= 0).any()):
            return self._empty_fusion_output(
                features,
                alignment_mode="base_only_no_voxel_support",
                geometry_reason="aligned evidence lies outside the canonical voxel lattice",
                geometry_ms=geometry_ms,
            )

        if points_are_trellis_normalized:
            aligned_for_depth = (
                normalized_points * canonical_half_extent[:, None, :, None, None]
                + canonical_center[:, None, :, None, None]
            )
        else:
            aligned_for_depth = aligned
        dataset_w2c = torch.linalg.inv(dataset_c2w.float())
        aligned_h = torch.cat(
            [aligned_for_depth.float(), torch.ones(B, V, 1, Hf, Wf, device=features.device)],
            dim=2,
        )
        aligned_camera = torch.einsum("bvij,bvjhw->bvihw", dataset_w2c, aligned_h)
        aligned_depth = aligned_camera[:, :, 2:3]
        depth_supervision_weight = (
            selected_confidence
            * alignment_quality
            * valid.float()
            * (aligned_depth[:, :, 0] > 0).float()
        )
        output = self._reduce_to_voxels(
            normalized_points,
            features,
            effective_confidence,
            weights,
            valid,
            geometry.valid_view_mask,
            alignment_scale,
            profile,
        )
        return replace(
            output,
            alignment_plausible_fraction=alignable_fraction,
            alignment_mode=alignment_mode,
            geometry_available=torch.ones(B, device=features.device, dtype=torch.bool),
            geometry_quality=alignable_fraction,
            geometry_status=("usable" if alignment_mode == "pointmap_reference" else "fallback"),
            geometry_reason=(
                "" if alignment_mode == "pointmap_reference" else "depth hypothesis retained more canonical evidence"
            ),
            aligned_depth=aligned_depth,
            depth_supervision_weight=depth_supervision_weight,
            timings_ms={"voxel_unprojection_alignment": geometry_ms, **output.timings_ms},
        )

    def _empty_fusion_output(
        self,
        features: torch.Tensor,
        *,
        alignment_mode: str,
        geometry_reason: str,
        geometry_ms: Optional[float],
    ) -> VoxelFusionOutput:
        """Represent an object with no defensible geometry as base-only evidence."""
        batch_size = features.shape[0]
        token_count = self.grid_resolution**3
        # Keep every fusion parameter in DDP's graph even when this sample has
        # no usable geometric evidence. The dependency is identically zero.
        graph_zero = features.new_zeros(())
        for parameter in self.parameters():
            if parameter.requires_grad and parameter.numel() > 0:
                graph_zero = graph_zero + parameter.reshape(-1)[0] * 0.0
        dense = features.new_zeros(batch_size, token_count, self.condition_dim) + graph_zero
        scalar = features.new_zeros(batch_size, token_count, 1)
        observation = torch.zeros(
            batch_size, token_count, 1, dtype=torch.bool, device=features.device
        )
        return VoxelFusionOutput(
            sparse_tensor=None,
            dense_tokens=dense,
            voxel_indices=torch.empty(0, 4, dtype=torch.int32, device=features.device),
            voxel_xyz=features.new_empty(0, 3),
            voxel_confidence=scalar,
            observation_mask=observation,
            accumulated_weight=scalar.clone(),
            observation_count=scalar.clone(),
            valid_mask=observation[..., 0],
            alignment_scale=features.new_ones(batch_size),
            condition_dim=self.condition_dim,
            alignment_plausible_fraction=features.new_zeros(batch_size),
            alignment_mode=alignment_mode,
            geometry_available=torch.zeros(batch_size, device=features.device, dtype=torch.bool),
            conditioning_available=torch.zeros(
                batch_size, device=features.device, dtype=torch.bool
            ),
            weak_evidence_only=torch.zeros(
                batch_size, device=features.device, dtype=torch.bool
            ),
            geometry_quality=features.new_zeros(batch_size),
            geometry_status="base_only",
            geometry_reason=geometry_reason,
            aligned_depth=features.new_zeros(batch_size, features.shape[1], 1, *features.shape[-2:]),
            depth_supervision_weight=features.new_zeros(batch_size, features.shape[1], *features.shape[-2:]),
            timings_ms={
                "voxel_unprojection_alignment": geometry_ms,
                "voxel_reduction": None,
                "spconv_construction_refinement": None,
            },
        )


def _score_alignment_candidate(
    aligned_points: torch.Tensor,
    alignment_inlier: torch.Tensor,
    source_valid: torch.Tensor,
    confidence: torch.Tensor,
    foreground: torch.Tensor,
    mask_interior: torch.Tensor,
    canonical_center: torch.Tensor,
    canonical_half_extent: torch.Tensor,
    *,
    points_are_normalized: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a continuous, confidence-weighted alignment hypothesis score."""
    normalized = (
        aligned_points.float()
        if points_are_normalized
        else world_to_trellis_normalized(
            aligned_points.float(), canonical_center, canonical_half_extent
        )
    )
    plausible = (
        torch.isfinite(normalized).all(dim=2)
        & (normalized.abs() <= 1.25).all(dim=2)
        & alignment_inlier
    )
    interior = source_valid & mask_interior & (foreground > 0)
    has_interior = interior.flatten(1).any(dim=1).view(-1, 1, 1, 1)
    expected = torch.where(has_interior, interior, source_valid & (foreground > 0))
    confidence_weight = torch.nan_to_num(
        confidence.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(0, 1)
    weighted = confidence_weight * expected.float()
    has_confidence = weighted.flatten(1).sum(dim=1).view(-1, 1, 1, 1) > 1e-8
    weighted = torch.where(has_confidence, weighted, expected.float())
    retained = (weighted * plausible.float()).flatten(1).sum(dim=1)
    total = weighted.flatten(1).sum(dim=1)
    quality = torch.where(total > 0, retained / total.clamp_min(1e-8), torch.zeros_like(total))
    return aligned_points, normalized, plausible, quality


def unproject_depth_batched(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Fully vectorized OpenCV depth unprojection on the input device."""
    if depth.ndim != 5 or depth.shape[2] != 1:
        raise ValueError(f"depth must be [B,V,1,H,W], got {tuple(depth.shape)}")
    B, V, _, H, W = depth.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=torch.float32),
        torch.arange(W, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    z = depth[:, :, 0].float()
    x_cam = (x.view(1, 1, H, W) - K[..., 0, 2, None, None]) * z / K[..., 0, 0, None, None]
    y_cam = (y.view(1, 1, H, W) - K[..., 1, 2, None, None]) * z / K[..., 1, 1, None, None]
    camera = torch.stack([x_cam, y_cam, z, torch.ones_like(z)], dim=2)
    world = torch.einsum("bvij,bvjhw->bvihw", c2w.float(), camera)
    return world[:, :, :3]


def align_vggt_reference_to_dataset(
    points: torch.Tensor,
    vggt_w2c: torch.Tensor,
    dataset_c2w: torch.Tensor,
    canonical_center: torch.Tensor,
    canonical_half_extent: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve VGGT's scale gauge against the known canonical AABB.

    Rotation and origin are fixed by the reference camera pair. The remaining
    scalar is estimated from foreground ray/AABB entry scales, then constrained
    to the interval that keeps reliable interior points inside a 25% safety
    envelope. This is camera-consistent and avoids the visible-surface bias of
    matching a foreground median depth to the object-center depth. The returned
    mask marks rays that can intersect that safety envelope at a positive scale.
    """
    B, V, _, H, W = points.shape
    reference_index = valid.flatten(2).any(dim=2).to(torch.int64).argmax(dim=1)
    batch = torch.arange(B, device=points.device)
    ref_w2c = vggt_w2c[batch, reference_index]
    ref_dataset_c2w = dataset_c2w[batch, reference_index]
    points_h = torch.cat([points.float(), torch.ones(B, V, 1, H, W, device=points.device)], dim=2)
    camera_points = torch.einsum("bij,bvjhw->bvihw", ref_w2c, points_h)[:, :, :3]
    rotation = ref_dataset_c2w[:, :3, :3]
    origin = ref_dataset_c2w[:, :3, 3]
    directions = torch.einsum("bij,bvjhw->bvihw", rotation, camera_points)
    # Confidence contributes continuously to hypothesis scoring. It must not
    # become another binary support threshold at alignment time.
    reliable = valid & torch.isfinite(confidence)
    entry, exit, hits = _ray_box_scale_interval(
        directions,
        origin,
        canonical_center - canonical_half_extent,
        canonical_center + canonical_half_extent,
    )
    reliable_hits = reliable & hits & torch.isfinite(entry) & torch.isfinite(exit) & (exit > 0)
    plausible_entry, plausible_exit, plausible_hits = _ray_box_scale_interval(
        directions,
        origin,
        canonical_center - 1.25 * canonical_half_extent,
        canonical_center + 1.25 * canonical_half_extent,
    )
    ray_alignable = (
        plausible_hits
        & torch.isfinite(plausible_entry)
        & torch.isfinite(plausible_exit)
        & (plausible_exit > 0)
    )
    plausible_reliable = reliable & ray_alignable
    if bool((plausible_reliable.flatten(1).sum(dim=1) == 0).any()):
        raise NoCanonicalRayIntersectionError(
            "VGGT reference rays do not intersect the canonical safety envelope"
        )
    canonical_available = reliable_hits.flatten(1).any(dim=1)
    canonical_scale = torch.nanquantile(
        torch.where(reliable_hits, entry.clamp_min(0), torch.nan).flatten(1),
        0.9,
        dim=1,
    )
    safety_scale = torch.nanquantile(
        torch.where(plausible_reliable, plausible_entry.clamp_min(0), torch.nan).flatten(1),
        0.9,
        dim=1,
    )
    scale = torch.where(canonical_available, canonical_scale, safety_scale)
    # Select the scale supported by the largest exact overlap of reliable ray
    # intervals. Requiring the intersection of *all* intervals lets one VGGT
    # outlier reject an otherwise coherent multi-view object.
    scale, _ = _maximum_interval_consensus_scale(
        scale,
        plausible_entry.clamp_min(torch.finfo(scale.dtype).eps),
        plausible_exit,
        plausible_reliable,
    )
    if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
        raise NoCanonicalRayIntersectionError("No finite positive VGGT gauge scale")
    aligned = origin[:, None, :, None, None] + scale[:, None, None, None, None] * directions
    plausible_min = canonical_center - 1.25 * canonical_half_extent
    plausible_max = canonical_center + 1.25 * canonical_half_extent
    tolerance = canonical_half_extent * 1e-5 + 1e-6
    alignment_inlier = (
        torch.isfinite(aligned).all(dim=2)
        & (aligned >= (plausible_min - tolerance)[:, None, :, None, None]).all(dim=2)
        & (aligned <= (plausible_max + tolerance)[:, None, :, None, None]).all(dim=2)
    )
    return aligned, scale, alignment_inlier


def align_depth_rays_to_dataset(
    depth_points: torch.Tensor,
    dataset_c2w: torch.Tensor,
    canonical_center: torch.Tensor,
    canonical_half_extent: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align VGGT depth along the authoritative dataset-camera rays."""
    origins = dataset_c2w[:, :, :3, 3]
    directions = depth_points.float() - origins[:, :, :, None, None]
    reliable = valid & torch.isfinite(confidence)
    entry, exit, hits = _ray_box_scale_interval(
        directions,
        origins,
        canonical_center - canonical_half_extent,
        canonical_center + canonical_half_extent,
    )
    reliable_hits = reliable & hits & torch.isfinite(entry) & torch.isfinite(exit) & (exit > 0)
    plausible_entry, plausible_exit, plausible_hits = _ray_box_scale_interval(
        directions,
        origins,
        canonical_center - 1.25 * canonical_half_extent,
        canonical_center + 1.25 * canonical_half_extent,
    )
    plausible_reliable = (
        reliable
        & plausible_hits
        & torch.isfinite(plausible_entry)
        & torch.isfinite(plausible_exit)
        & (plausible_exit > 0)
    )
    if bool((plausible_reliable.flatten(1).sum(dim=1) == 0).any()):
        raise NoCanonicalRayIntersectionError(
            "VGGT depth on dataset-camera rays does not intersect the canonical safety envelope"
        )
    canonical_available = reliable_hits.flatten(1).any(dim=1)
    canonical_scale = torch.nanquantile(
        torch.where(reliable_hits, entry.clamp_min(0), torch.nan).flatten(1),
        0.9,
        dim=1,
    )
    safety_scale = torch.nanquantile(
        torch.where(plausible_reliable, plausible_entry.clamp_min(0), torch.nan).flatten(1),
        0.9,
        dim=1,
    )
    preferred_scale = torch.where(canonical_available, canonical_scale, safety_scale)
    scale, _ = _maximum_interval_consensus_scale(
        preferred_scale,
        plausible_entry.clamp_min(torch.finfo(preferred_scale.dtype).eps),
        plausible_exit,
        plausible_reliable,
    )
    if not torch.isfinite(scale).all() or bool((scale <= 0).any()):
        raise NoCanonicalRayIntersectionError("No finite positive dataset-depth gauge scale")
    aligned = origins[:, :, :, None, None] + scale[:, None, None, None, None] * directions
    plausible_min = canonical_center - 1.25 * canonical_half_extent
    plausible_max = canonical_center + 1.25 * canonical_half_extent
    tolerance = canonical_half_extent * 1e-5 + 1e-6
    inlier = (
        torch.isfinite(aligned).all(dim=2)
        & (aligned >= (plausible_min - tolerance)[:, None, :, None, None]).all(dim=2)
        & (aligned <= (plausible_max + tolerance)[:, None, :, None, None]).all(dim=2)
    )
    return aligned, scale, inlier


def _maximum_interval_consensus_scale(
    preferred_scale: torch.Tensor,
    entry: torch.Tensor,
    exit: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose a positive scale covered by the maximum number of intervals.

    The batched sweep is ``O(P log P)`` from CUDA sorts/searches and introduces
    no pixel/ray Python loop. If the preferred robust scale already attains the
    maximum consensus, it is preserved instead of snapping to an interval edge.
    """
    if (
        entry.shape != exit.shape
        or entry.shape != valid.shape
        or entry.shape[0] != preferred_scale.shape[0]
    ):
        raise ValueError(
            "Scale interval contract mismatch: "
            f"preferred={tuple(preferred_scale.shape)}, entry={tuple(entry.shape)}, "
            f"exit={tuple(exit.shape)}, valid={tuple(valid.shape)}"
        )
    batch_size = entry.shape[0]
    flat_entry = entry.float().reshape(batch_size, -1)
    flat_exit = exit.float().reshape(batch_size, -1)
    flat_valid = (
        valid.reshape(batch_size, -1)
        & torch.isfinite(flat_entry)
        & torch.isfinite(flat_exit)
        & (flat_exit >= flat_entry)
    )
    valid_count = flat_valid.sum(dim=1)
    if bool((valid_count == 0).any()):
        raise RuntimeError("No reliable VGGT ray interval can satisfy the canonical safety envelope")

    infinity = torch.full_like(flat_entry, torch.inf)
    sorted_entry = torch.sort(torch.where(flat_valid, flat_entry, infinity), dim=1).values
    sorted_exit = torch.sort(torch.where(flat_valid, flat_exit, infinity), dim=1).values
    ended_before = torch.searchsorted(sorted_exit.contiguous(), sorted_entry.contiguous(), right=False)
    started = torch.arange(1, sorted_entry.shape[1] + 1, device=entry.device).view(1, -1)
    overlap = started - ended_before
    overlap = torch.where(torch.isfinite(sorted_entry), overlap, overlap.new_full((), -1))
    best_index = overlap.argmax(dim=1)
    best_scale = sorted_entry.gather(1, best_index[:, None])[:, 0]
    best_count = overlap.gather(1, best_index[:, None])[:, 0]

    preferred = preferred_scale.float().reshape(batch_size, 1)
    preferred_count = (
        flat_valid & (flat_entry <= preferred) & (flat_exit >= preferred)
    ).sum(dim=1)
    use_preferred = preferred_count >= best_count
    selected = torch.where(use_preferred, preferred[:, 0], best_scale)
    consensus_fraction = torch.maximum(preferred_count, best_count).float() / valid_count.float()
    return selected.to(preferred_scale.dtype), consensus_fraction


def _validate_alignment_retention(retained_fraction: torch.Tensor) -> None:
    """Require some finite consensus; its magnitude is handled continuously."""
    if not torch.isfinite(retained_fraction).all() or bool((retained_fraction <= 0).any()):
        raise RuntimeError(
            "VGGT-to-TRELLIS alignment retained no reliable geometry: "
            f"retained_fractions={retained_fraction.detach().cpu().tolist()}"
        )


def _ray_box_scale_interval(
    directions: torch.Tensor,
    origin: torch.Tensor,
    box_min: torch.Tensor,
    box_max: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return scalar intervals where ``origin + scale * direction`` is in a box."""
    if origin.shape == (directions.shape[0], 3):
        ray_origin = origin[:, None, :, None, None]
    elif origin.shape == (directions.shape[0], directions.shape[1], 3):
        ray_origin = origin[:, :, :, None, None]
    else:
        raise ValueError(
            f"Ray origin must be [B,3] or [B,V,3], got {tuple(origin.shape)} "
            f"for directions={tuple(directions.shape)}"
        )
    parallel = directions.abs() < 1e-8
    safe = torch.where(parallel, torch.ones_like(directions), directions)
    lower = (box_min[:, None, :, None, None] - ray_origin) / safe
    upper = (box_max[:, None, :, None, None] - ray_origin) / safe
    axis_entry = torch.where(
        parallel, torch.full_like(lower, -torch.inf), torch.minimum(lower, upper)
    )
    axis_exit = torch.where(
        parallel, torch.full_like(upper, torch.inf), torch.maximum(lower, upper)
    )
    entry = axis_entry.amax(dim=2)
    exit = axis_exit.amin(dim=2)
    origin_inside_parallel_slab = (
        (ray_origin >= box_min[:, None, :, None, None])
        & (ray_origin <= box_max[:, None, :, None, None])
    )
    parallel_valid = (~parallel | origin_inside_parallel_slab).all(dim=2)
    hits = parallel_valid & (exit >= entry.clamp_min(0))
    return entry, exit, hits


def world_to_trellis_normalized(points: torch.Tensor, center: torch.Tensor, half_extent: torch.Tensor) -> torch.Tensor:
    """Map dataset object coordinates into TRELLIS-normalized ``[-1,1]^3``."""
    if bool((half_extent <= 0).any()):
        raise ValueError("canonical_half_extent must be strictly positive")
    return (points - center[:, None, :, None, None]) / half_extent[:, None, :, None, None]


def rescale_intrinsics(K: torch.Tensor, source_hw: Tuple[int, int], target_hw: Tuple[int, int]) -> torch.Tensor:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    result = K.clone().float()
    result[..., 0, :] *= float(target_w) / float(source_w)
    result[..., 1, :] *= float(target_h) / float(source_h)
    result[..., 2, :] = torch.tensor([0.0, 0.0, 1.0], device=K.device)
    return result


def robust_confidence_probability(confidence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Quantile-normalize finite VGGT reliabilities without discarding ordering."""
    values = torch.where(valid, confidence.float().clamp(0, 1), torch.nan).flatten(1)
    low = torch.nanquantile(values, 0.05, dim=1).view(-1, 1, 1, 1)
    high = torch.nanquantile(values, 0.95, dim=1).view(-1, 1, 1, 1)
    normalized = (confidence.float() - low) / (high - low).clamp_min(1e-6)
    normalized = torch.where((high - low) > 1e-6, normalized, confidence.float())
    return torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)


def sinusoidal_3d_encoding(xyz: torch.Tensor, num_frequencies: int) -> torch.Tensor:
    if xyz.shape[-1] != 3:
        raise ValueError(f"xyz must end in 3, got {tuple(xyz.shape)}")
    frequencies = (2.0 ** torch.arange(num_frequencies, device=xyz.device, dtype=torch.float32)) * torch.pi
    angles = xyz.float()[..., :, None] * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)


def _resize_bv_map(tensor: torch.Tensor, target: Tuple[int, int]) -> torch.Tensor:
    if tensor.ndim == 4:
        tensor = tensor[:, :, None]
    B, V, C, H, W = tensor.shape
    if (H, W) == target:
        return tensor
    resized = F.interpolate(tensor.reshape(B * V, C, H, W), size=target, mode="bilinear", align_corners=False)
    return resized.reshape(B, V, C, *target)


def _resize_validity_mask(mask: torch.Tensor, target: Tuple[int, int]) -> torch.Tensor:
    """Conservatively resize a per-pixel validity mask without inventing support."""
    if mask.ndim != 4:
        raise ValueError(f"validity mask must be [B,V,H,W], got {tuple(mask.shape)}")
    B, V, H, W = mask.shape
    if (H, W) == target:
        return mask.bool()
    resized = F.interpolate(
        mask.float().reshape(B * V, 1, H, W), size=target, mode="nearest"
    )
    return resized.reshape(B, V, *target) > 0.5


def _assert_alignment_inputs(c2w: torch.Tensor, center: torch.Tensor, half: torch.Tensor, B: int, V: int) -> None:
    if c2w.shape != (B, V, 4, 4) or center.shape != (B, 3) or half.shape != (B, 3):
        raise ValueError(
            f"Alignment contract mismatch: c2w={tuple(c2w.shape)}, center={tuple(center.shape)}, half={tuple(half.shape)}"
        )
    if not torch.isfinite(c2w).all() or not torch.isfinite(center).all() or not torch.isfinite(half).all():
        raise ValueError("Alignment inputs contain NaN/Inf")


def _import_spconv(*, required: bool):
    try:
        import spconv.pytorch as spconv
    except ImportError as exc:
        if required:
            raise ImportError("Production voxel fusion requires installed spconv.pytorch") from exc
        return None
    return spconv


def _make_sparse_tensor(
    features: torch.Tensor,
    indices: torch.Tensor,
    *,
    batch_size: int,
    resolution: int,
    required: bool,
):
    spconv = _import_spconv(required=required)
    if spconv is None:
        return None
    # Installed spconv 2.x uses [batch,z,y,x] indices for 3-D tensors.
    return spconv.SparseConvTensor(
        features=features,
        indices=indices,
        spatial_shape=[resolution, resolution, resolution],
        batch_size=batch_size,
    )


class _CudaTimer:
    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True) if enabled and device.type == "cuda" else None
        self.end_event = torch.cuda.Event(enable_timing=True) if self.start_event is not None else None

    def start(self) -> None:
        if self.start_event is not None:
            self.start_event.record()

    def stop(self) -> Optional[float]:
        if self.start_event is None or self.end_event is None:
            return None
        self.end_event.record()
        self.end_event.synchronize()
        return float(self.start_event.elapsed_time(self.end_event))


__all__ = [
    "ConfidenceSparseVoxelFusion",
    "VoxelFusionOutput",
    "align_vggt_reference_to_dataset",
    "rescale_intrinsics",
    "sinusoidal_3d_encoding",
    "unproject_depth_batched",
    "world_to_trellis_normalized",
]
