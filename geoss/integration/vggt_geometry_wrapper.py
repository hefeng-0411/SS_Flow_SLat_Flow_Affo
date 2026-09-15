from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VGGTGeometryBatch:
    """Typed, rank-local output of one frozen VGGT multi-view forward.

    ``extrinsics`` are OpenCV world-to-camera matrices. ``point_map`` and the
    cameras share VGGT's reference-frame gauge; alignment to the MeshFleet
    object frame is intentionally performed by the voxel fusion engine.
    """

    depth: torch.Tensor
    depth_confidence: torch.Tensor
    point_map: torch.Tensor
    point_confidence: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    camera_to_world: torch.Tensor
    visual_features: torch.Tensor
    valid_view_mask: torch.Tensor
    image_resolution: Tuple[int, int]
    feature_resolution: Tuple[int, int]
    patch_size: int
    depth_valid_mask: Optional[torch.Tensor] = None
    point_valid_mask: Optional[torch.Tensor] = None
    camera_valid_mask: Optional[torch.Tensor] = None
    diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)


class VGGTGeometryWrapper(nn.Module):
    """Frozen real-VGGT extractor; explicit mocks remain test-only."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        vggt_root: Optional[str] = None,
        checkpoint: Optional[str] = None,
        pretrained_name: Optional[str] = None,
        mock: bool = False,
        cache_features: bool = False,
        vggt_image_size: int = 518,
        require_real: bool = False,
    ) -> None:
        super().__init__()
        if require_real and (mock or (model is None and checkpoint is None and pretrained_name is None)):
            raise RuntimeError(
                "Production VGGT requires a real supplied model, checkpoint, or pretrained_name; mock/random VGGT is forbidden."
            )
        self.mock = mock
        self.require_real = bool(require_real)
        self.cache_features = cache_features
        self.vggt_image_size = vggt_image_size
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.model = model
        self.pose_decoder = None
        if self.model is None and not mock:
            self.model, self.pose_decoder = self._load_vggt(vggt_root, checkpoint, pretrained_name)
        elif self.model is not None:
            self.pose_decoder = self._load_pose_decoder(vggt_root)
        if self.model is not None:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.eval()

    def clear_cache(self) -> None:
        """Release per-object frozen predictions after a batch is consumed."""
        self._cache.clear()

    def forward(self, images: torch.Tensor, *, use_cache: bool = False) -> Dict[str, torch.Tensor]:
        """Backward-compatible dictionary API. Production code uses :meth:`extract`."""
        images = self._normalize_input_images(images)
        if self.mock or self.model is None:
            return self._mock_forward(images)
        cache_key = (int(images.data_ptr()), tuple(images.shape), str(images.device))
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        model_images = _resize_for_vggt(images, self.vggt_image_size)
        amp_enabled = model_images.device.type == "cuda"
        amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
        with torch.inference_mode(), torch.autocast(
            device_type=model_images.device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            predictions, tokens, patch_start_idx, raw_last_shape = self._forward_vggt_once(model_images)
            out = self._normalize_predictions(predictions, model_images, output_size=images.shape[-2:])
            if tokens is not None:
                dense = _tokens_to_spatial_features(tokens, model_images.shape[-2:])
                out["vggt_features"] = dense if dense is not None else tokens
                out["vggt_feature_tokens"] = tokens
                out["feature_shape_info"] = {
                    "format": "patch_grid_from_tokens" if dense is not None else "aggregator_patch_tokens",
                    "tokens": list(tokens.shape),
                    "features": list(dense.shape) if dense is not None else None,
                    "patch_start_idx": int(patch_start_idx),
                    "raw_last_tokens": raw_last_shape,
                    "model_image_size": list(model_images.shape[-2:]),
                    "original_image_size": list(images.shape[-2:]),
                }
            else:
                out["vggt_features"] = None
                out["feature_shape_info"] = {"format": "unavailable", "model_image_size": list(model_images.shape[-2:])}
        if self.cache_features or use_cache:
            self._cache[cache_key] = out
        return out

    def extract(
        self,
        images: torch.Tensor,
        *,
        valid_view_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> VGGTGeometryBatch:
        """Return the complete production geometry contract or fail loudly."""
        normalized = self._normalize_input_images(images)
        B, V = normalized.shape[:2]
        if valid_view_mask is None:
            valid_view_mask = torch.ones(B, V, device=normalized.device, dtype=torch.bool)
        else:
            valid_view_mask = valid_view_mask.to(device=normalized.device, dtype=torch.bool)
        if valid_view_mask.shape != (B, V) or bool((valid_view_mask.sum(dim=1) == 0).any()):
            raise ValueError(f"valid_view_mask must contain at least one view per object, got {tuple(valid_view_mask.shape)}")
        output = self.forward(normalized, use_cache=use_cache)
        required = (
            "vggt_depth",
            "vggt_depth_confidence",
            "vggt_pointmap",
            "vggt_point_confidence",
            "vggt_features",
            "vggt_camera",
        )
        missing = [name for name in required if output.get(name) is None]
        if missing:
            raise RuntimeError(f"Real VGGT output is missing required heads/features: {missing}")
        camera = output["vggt_camera"]
        depth = output["vggt_depth"].float()
        point_map = output["vggt_pointmap"].float()
        features = output["vggt_features"]
        if features.ndim != 5:
            raise RuntimeError(f"VGGT spatial feature contract must be [B,V,C,Hf,Wf], got {tuple(features.shape)}")
        camera_valid = _vggt_camera_validity(
            camera["w2c"], camera["c2w"], camera["K"], valid_view_mask
        )
        spatial_valid = valid_view_mask[:, :, None, None]
        depth_valid = spatial_valid & torch.isfinite(depth[:, :, 0]) & (depth[:, :, 0] > 0)
        point_valid = spatial_valid & torch.isfinite(point_map).all(dim=2)
        feature_finite = torch.isfinite(features).all(dim=2)

        # Frozen foundation predictions are evidence, not dataset contracts.
        # Preserve their validity explicitly and sanitize storage so one bad
        # head/pixel cannot spread NaNs through interpolation or attention.
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        point_map = torch.nan_to_num(point_map, nan=0.0, posinf=0.0, neginf=0.0)
        depth_confidence = torch.nan_to_num(
            output["vggt_depth_confidence"].float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0, 1)
        point_confidence = torch.nan_to_num(
            output["vggt_point_confidence"].float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(0, 1)
        features = torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)
        safe_w2c, safe_c2w, safe_K = _sanitize_vggt_camera_bundle(
            camera["w2c"], camera["K"], camera_valid
        )
        return VGGTGeometryBatch(
            # clone outside inference_mode so trainable projections may save
            # these frozen values for their own weight gradients.
            depth=depth.clone(),
            depth_confidence=depth_confidence.clone(),
            point_map=point_map.clone(),
            point_confidence=point_confidence.clone(),
            intrinsics=safe_K.clone(),
            extrinsics=safe_w2c.clone(),
            camera_to_world=safe_c2w.clone(),
            visual_features=features.float().clone(),
            valid_view_mask=valid_view_mask.clone(),
            image_resolution=tuple(int(value) for value in depth.shape[-2:]),
            feature_resolution=tuple(int(value) for value in features.shape[-2:]),
            patch_size=int(getattr(getattr(self.model, "aggregator", None), "patch_size", 14)),
            depth_valid_mask=depth_valid.clone(),
            point_valid_mask=point_valid.clone(),
            camera_valid_mask=camera_valid.clone(),
            diagnostics={
                "depth_valid_fraction": depth_valid.float().flatten(1).mean(dim=1),
                "point_valid_fraction": point_valid.float().flatten(1).mean(dim=1),
                "feature_finite_fraction": feature_finite.float().flatten(1).mean(dim=1),
                "camera_valid_fraction": (
                    camera_valid & valid_view_mask
                ).float().sum(dim=1) / valid_view_mask.float().sum(dim=1).clamp_min(1),
            },
        )

    def _forward_vggt_once(self, images: torch.Tensor):
        if not hasattr(self.model, "aggregator"):
            predictions = self.model(images)
            return predictions, None, 0, None
        aggregated_tokens_list, patch_start_idx = self.model.aggregator(images)
        predictions: Dict[str, torch.Tensor] = {}
        # Keep geometry heads FP32 while using the non-deprecated AMP API.
        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            if getattr(self.model, "camera_head", None) is not None:
                pose_enc_list = self.model.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
            if getattr(self.model, "depth_head", None) is not None:
                depth, depth_conf = self.model.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
            if getattr(self.model, "point_head", None) is not None:
                pts3d, pts3d_conf = self.model.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
        last = aggregated_tokens_list[-1]
        tokens = last[:, :, patch_start_idx:]
        return predictions, tokens, patch_start_idx, list(last.shape)

    def _normalize_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        images: torch.Tensor,
        *,
        output_size: Optional[tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, _, H, W = images.shape
        out_h, out_w = output_size or (H, W)
        out: Dict[str, Any] = {}
        depth = _first_present(predictions, ["depth", "depth_map", "pred_depth", "vggt_depth"])
        out["vggt_depth"] = _resize_b_n_c_h_w(_to_b_n_1_h_w(depth, B, N), (out_h, out_w)) if depth is not None else None
        pointmap = _first_present(predictions, ["world_points", "point_map", "pointmap", "points3d", "vggt_pointmap"])
        out["vggt_pointmap"] = _resize_b_n_c_h_w(_to_b_n_3_h_w(pointmap, B, N), (out_h, out_w)) if pointmap is not None else None
        depth_conf = _first_present(predictions, ["depth_conf"])
        point_conf = _first_present(predictions, ["world_points_conf", "confidence", "conf"])
        selected_conf = point_conf if pointmap is not None and point_conf is not None else depth_conf
        if selected_conf is not None:
            raw_confidence = _resize_confidence(selected_conf, B, N, (out_h, out_w))
            out["vggt_confidence_raw"] = raw_confidence
            out["vggt_confidence"] = _expp1_confidence_to_probability(raw_confidence)
        else:
            out["vggt_confidence_raw"] = None
            out["vggt_confidence"] = None
        if depth_conf is not None:
            raw_depth_conf = _resize_confidence(depth_conf, B, N, (out_h, out_w))
            out["vggt_depth_confidence"] = _expp1_confidence_to_probability(raw_depth_conf)
        if point_conf is not None:
            raw_point_conf = _resize_confidence(point_conf, B, N, (out_h, out_w))
            out["vggt_point_confidence"] = _expp1_confidence_to_probability(raw_point_conf)
        pose = _first_present(predictions, ["pose_enc", "camera", "camera_pose"])
        if pose is not None and self.pose_decoder is not None:
            extrinsic, intrinsic = self.pose_decoder(pose, images.shape[-2:])
            w2c = torch.eye(4, device=images.device, dtype=images.dtype).view(1, 1, 4, 4).repeat(B, N, 1, 1)
            w2c[:, :, :3, :4] = extrinsic.to(images.dtype)
            K = intrinsic.to(images.dtype)
            if (out_h, out_w) != (H, W):
                K = K.clone()
                K[..., 0, :] *= float(out_w) / float(W)
                K[..., 1, :] *= float(out_h) / float(H)
            out["vggt_camera"] = {"w2c": w2c, "c2w": torch.linalg.inv(w2c), "K": K}
        out.setdefault("feature_shape_info", {})
        return out

    def _mock_forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.require_real:
            raise RuntimeError("Mock VGGT cannot execute in production mode")
        B, N, _, H, W = images.shape
        pooled = F.interpolate(images.reshape(B * N, 3, H, W), size=(H // 8, W // 8), mode="bilinear", align_corners=False)
        features = pooled.reshape(B, N, 3, H // 8, W // 8)
        depth = torch.ones(B, N, 1, H, W, device=images.device, dtype=images.dtype)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=images.device, dtype=images.dtype),
            torch.linspace(-1, 1, W, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        pointmap = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).view(1, 1, 3, H, W).expand(B, N, -1, -1, -1)
        return {
            "vggt_features": features,
            "vggt_depth": depth,
            "vggt_pointmap": pointmap,
            "vggt_confidence": torch.ones(B, N, H, W, device=images.device, dtype=images.dtype),
            "feature_shape_info": {"format": "mock_spatial", "features": list(features.shape)},
        }

    @staticmethod
    def _normalize_input_images(images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5:
            raise ValueError(f"VGGTGeometryWrapper expects images [B,N,3,H,W] or [N,3,H,W], got {tuple(images.shape)}")
        if images.shape[2] != 3:
            raise ValueError(f"VGGTGeometryWrapper expects RGB channel at dim 2, got {tuple(images.shape)}")
        return images

    def _load_pose_decoder(self, vggt_root: Optional[str]):
        if vggt_root:
            sys.path.insert(0, str(Path(vggt_root)))
        try:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except Exception:
            return None
        return pose_encoding_to_extri_intri

    def _load_vggt(self, vggt_root: Optional[str], checkpoint: Optional[str], pretrained_name: Optional[str]):
        if vggt_root:
            sys.path.insert(0, str(Path(vggt_root)))
        try:
            from vggt.models.vggt import VGGT
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        except Exception as exc:
            raise ImportError("Could not import VGGT. Use mock=True for dry-run.") from exc
        if pretrained_name is not None and hasattr(VGGT, "from_pretrained"):
            model = VGGT.from_pretrained(pretrained_name)
        else:
            model = VGGT()
            if pretrained_name is not None:
                _load_vggt_pretrained_fallback(model, pretrained_name)
        if checkpoint:
            ckpt_path = Path(checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
            state = torch.load(ckpt_path, map_location="cpu")
            state = _extract_state_dict(state)
            model.load_state_dict(state, strict=True)
        return model, pose_encoding_to_extri_intri


def _vggt_camera_validity(
    w2c: torch.Tensor,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    valid_view_mask: torch.Tensor,
) -> torch.Tensor:
    if w2c.shape[-2:] != (4, 4) or c2w.shape != w2c.shape or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(
            f"Malformed VGGT cameras: w2c={tuple(w2c.shape)}, c2w={tuple(c2w.shape)}, K={tuple(intrinsics.shape)}"
        )
    finite = (
        torch.isfinite(w2c).all(dim=(-2, -1))
        & torch.isfinite(c2w).all(dim=(-2, -1))
        & torch.isfinite(intrinsics).all(dim=(-2, -1))
    )
    identity = torch.eye(4, device=w2c.device, dtype=w2c.dtype)
    safe_w2c = torch.nan_to_num(w2c.float())
    safe_c2w = torch.nan_to_num(c2w.float())
    inverse_error = (safe_w2c @ safe_c2w - identity.float()).abs().amax(dim=(-2, -1))
    rotation = safe_w2c[..., :3, :3]
    rotation_error = (rotation.transpose(-1, -2) @ rotation - identity[:3, :3]).abs().amax(dim=(-2, -1))
    positive_focal = (intrinsics[..., 0, 0] > 0) & (intrinsics[..., 1, 1] > 0)
    return (
        valid_view_mask
        & finite
        & (inverse_error <= 2e-3)
        & (rotation_error <= 2e-3)
        & positive_focal
    )


def _sanitize_vggt_camera_bundle(
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace unusable predicted cameras while retaining a validity mask."""
    B, V = valid.shape
    identity4 = torch.eye(4, device=w2c.device, dtype=torch.float32).view(1, 1, 4, 4)
    identity3 = torch.eye(3, device=w2c.device, dtype=torch.float32).view(1, 1, 3, 3)
    safe_w2c = torch.where(
        valid[:, :, None, None], torch.nan_to_num(w2c.float()), identity4.expand(B, V, -1, -1)
    )
    safe_K = torch.where(
        valid[:, :, None, None],
        torch.nan_to_num(intrinsics.float()),
        identity3.expand(B, V, -1, -1),
    )
    return safe_w2c, torch.linalg.inv(safe_w2c), safe_K


def _to_b_n_1_h_w(depth: torch.Tensor, B: int, N: int) -> torch.Tensor:
    if depth is None:
        return None
    if depth.shape[:3] == (B, N, 1):
        return depth
    if depth.shape[0] == B and depth.shape[1] == N and depth.shape[-1] == 1:
        return depth.permute(0, 1, 4, 2, 3).contiguous()
    raise ValueError(f"Unsupported VGGT depth shape {tuple(depth.shape)}")


def _to_b_n_3_h_w(points: torch.Tensor, B: int, N: int) -> torch.Tensor:
    if points.shape[:3] == (B, N, 3):
        return points
    if points.shape[0] == B and points.shape[1] == N and points.shape[-1] == 3:
        return points.permute(0, 1, 4, 2, 3).contiguous()
    raise ValueError(f"Unsupported VGGT pointmap shape {tuple(points.shape)}")


def _resize_for_vggt(images: torch.Tensor, target_size: int) -> torch.Tensor:
    if target_size <= 0:
        target_size = _round_up_to_multiple(max(images.shape[-2:]), 14)
    if images.shape[-1] == target_size and images.shape[-2] == target_size:
        return images
    B, N, C, H, W = images.shape
    resized = F.interpolate(
        images.reshape(B * N, C, H, W),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(B, N, C, target_size, target_size).contiguous()


def _round_up_to_multiple(value: int, multiple: int) -> int:
    return int((value + multiple - 1) // multiple * multiple)


def _resize_b_n_c_h_w(tensor: Optional[torch.Tensor], output_size: tuple[int, int]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.shape[-2:] == output_size:
        return tensor
    B, N, C, H, W = tensor.shape
    resized = F.interpolate(
        tensor.reshape(B * N, C, H, W).float(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.to(dtype=tensor.dtype).reshape(B, N, C, output_size[0], output_size[1]).contiguous()


def _resize_confidence(conf: torch.Tensor, B: int, N: int, output_size: tuple[int, int]) -> torch.Tensor:
    if conf.shape[:2] == (B, N) and conf.ndim == 4:
        conf = conf.unsqueeze(2)
    elif conf.shape[:3] == (B, N, 1):
        conf = conf.contiguous()
    elif conf.shape[0] == B and conf.shape[1] == N and conf.shape[-1] == 1:
        conf = conf.permute(0, 1, 4, 2, 3).contiguous()
    else:
        return conf
    resized = _resize_b_n_c_h_w(conf, output_size)
    return resized[:, :, 0] if resized is not None else conf[:, :, 0]


def _expp1_confidence_to_probability(confidence: torch.Tensor) -> torch.Tensor:
    """Map VGGT's ``1 + exp(raw)`` confidence to a bounded reliability.

    The inverse relationship ``1 - 1 / confidence`` equals ``sigmoid(raw)``
    and preserves confidence ordering.  Directly clamping expp1 values to
    ``[0,1]`` makes every finite prediction equal to one and removes all
    uncertainty information.
    """
    finite = torch.nan_to_num(confidence.float(), nan=1.0, posinf=1e6, neginf=1.0)
    finite = finite.clamp_min(1.0)
    probability = 1.0 - finite.reciprocal()
    return probability.to(dtype=confidence.dtype).clamp(0.0, 1.0)


def _tokens_to_spatial_features(tokens: torch.Tensor, model_size: tuple[int, int]) -> Optional[torch.Tensor]:
    if tokens.ndim != 4:
        return None
    B, N, T, C = tokens.shape
    side = int(T ** 0.5)
    if side * side != T:
        h = max(1, int(round(model_size[0] / 14)))
        w = max(1, T // h)
        if h * w != T:
            return None
    else:
        h = w = side
    return tokens.permute(0, 1, 3, 2).reshape(B, N, C, h, w).contiguous()


def _first_present(mapping: Dict[str, Any], names: list[str]) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model", "model_state_dict", "module"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return {k.removeprefix("module."): v for k, v in value.items()}
    return {k.removeprefix("module."): v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)}


def _load_vggt_pretrained_fallback(model: nn.Module, pretrained_name: str) -> None:
    """Mirror VGGT demo loading for environments where from_pretrained is unavailable."""
    aliases = {
        "facebook/VGGT-1B": "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
        "VGGT-1B": "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
    }
    url = aliases.get(pretrained_name, pretrained_name)
    if url.startswith("http://") or url.startswith("https://"):
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    else:
        path = Path(url)
        if not path.exists():
            raise FileNotFoundError(
                f"VGGT pretrained source not found: {pretrained_name}. "
                "Use --vggt_pretrained facebook/VGGT-1B or --vggt_checkpoint /path/to/model.pt."
            )
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(_extract_state_dict(state), strict=True)
