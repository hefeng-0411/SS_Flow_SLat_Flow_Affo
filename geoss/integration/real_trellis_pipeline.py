from __future__ import annotations

import contextlib
import os
import sys
from types import MethodType
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from geoss.integration.trellis_ss_hook import (
    DirectConditionedTrellisSSWrapper,
    GeoSSTrellisSSWrapper,
    ss_grid_to_tokens,
    tokens_to_ss_grid,
)
from geoss.io.asset_io import write_internal_mesh
from geoss.metrics.gaussian_metrics import gaussian_statistics
from geoss.slat.integration.trellis_slat_hook import (
    DirectConditionedTrellisSLATWrapper,
    GeoVisTrellisSLATWrapper,
)


def _stage_autocast(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _atomic_asset_write(path: Path, writer: Callable[[Path], object]) -> None:
    """Publish a generated asset only after its complete payload reaches disk."""
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        writer(temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise OSError(f"Asset writer produced an empty file: {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class RealTrellisGeoPipeline:
    """Explicit real TRELLIS image->SS->SLAT->decoder wrapper.

    It avoids the TRELLIS `run()` convenience method so adapter contexts are
    passed into the samplers instead of silently ignored.
    """

    def __init__(self, trellis_root: Optional[str], pipeline_path: str, device: str = "cuda") -> None:
        if trellis_root:
            sys.path.insert(0, str(Path(trellis_root)))
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline
        except ModuleNotFoundError as exc:
            # TRELLIS imports rembg/pymatting at module import time.  Report an
            # environment contract failure explicitly so it cannot be confused
            # with a corrupt Stage-3/4 adapter checkpoint.
            missing = exc.name or "<unknown>"
            raise RuntimeError(
                "TRELLIS runtime dependency is missing from the active Python "
                f"environment: {missing!r} (python={sys.executable}). Install "
                "the project requirements with this same interpreter before "
                "running real inference."
            ) from exc

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(pipeline_path)
        self.pipeline.to(device)
        self.device = torch.device(device)
        self._require_models(
            "sparse_structure_flow_model",
            "sparse_structure_decoder",
            "slat_flow_model",
            "slat_decoder_gs",
        )

    def install_ss_adapter(self, velocity_adapter) -> None:
        self.pipeline.models["sparse_structure_flow_model"] = GeoSSTrellisSSWrapper(
            self.pipeline.models["sparse_structure_flow_model"],
            velocity_adapter,
            use_geoss_adapter=True,
        ).to(self.device)

    def install_slat_adapter(self, velocity_adapter) -> None:
        self.pipeline.models["slat_flow_model"] = GeoVisTrellisSLATWrapper(
            self.pipeline.models["slat_flow_model"],
            velocity_adapter,
            use_geovis_slat=True,
        ).to(self.device)

    def install_direct_ss_flow(self, flow_model) -> None:
        """Install a full-backbone Affostruction SS flow for native sampling."""
        self.pipeline.models["sparse_structure_flow_model"] = (
            DirectConditionedTrellisSSWrapper(flow_model).to(self.device).eval()
        )

    def install_direct_slat_flow(
        self,
        correction_flow,
        *,
        residual_limit: float = 1.0,
    ) -> None:
        """Install a sparse image-flow correction around frozen native SLat."""
        native_flow = self.pipeline.models["slat_flow_model"]
        self.pipeline.models["slat_flow_model"] = (
            DirectConditionedTrellisSLATWrapper(
                native_flow,
                correction_flow,
                residual_limit=residual_limit,
            ).to(self.device).eval()
        )

    def ss_velocity_hook(self, x_t: torch.Tensor, t: torch.Tensor, cond, base_velocity: torch.Tensor, context: Dict[str, torch.Tensor]):
        flow = self.pipeline.models["sparse_structure_flow_model"]
        if not isinstance(flow, GeoSSTrellisSSWrapper):
            raise RuntimeError("ss_velocity_hook requires install_ss_adapter() before sampling.")
        out = flow.velocity_adapter(
            ss_latent_tokens=ss_grid_to_tokens(x_t),
            geo_tokens=context["geo_tokens"],
            geo_confidence=context["geo_confidence"],
            timestep=t,
            v_base=ss_grid_to_tokens(base_velocity),
            voxel_xyz=context.get("ss_voxel_xyz"),
            anchor_xyz=context.get("anchor_xyz"),
            anchor_metadata=context.get("anchor_metadata"),
        )
        return tokens_to_ss_grid(out["v_geo"], tuple(x_t.shape[-3:])), out

    def slat_velocity_hook(self, x_t, t: torch.Tensor, cond, ss_context, base_velocity, context: Dict[str, torch.Tensor]):
        flow = self.pipeline.models["slat_flow_model"]
        if not isinstance(flow, GeoVisTrellisSLATWrapper):
            raise RuntimeError("slat_velocity_hook requires install_slat_adapter() before sampling.")
        return flow(x_t, t, cond, geovis_slat_context=context), dict(flow.last_debug)

    @torch.no_grad()
    def run(
        self,
        images: List,
        *,
        masks: Optional[torch.Tensor] = None,
        mask_aware_crop: bool = False,
        crop_padding: float = 1.2,
        geoss_context: Optional[Dict[str, torch.Tensor]] = None,
        geovis_slat_context: Optional[Dict[str, torch.Tensor]] = None,
        geovis_slat_context_factory: Optional[
            Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ] = None,
        coords_override: Optional[torch.Tensor] = None,
        formats: Iterable[str] = ("gaussian", "mesh"),
        seed: int = 42,
        ss_sampler_params: Optional[dict] = None,
        slat_sampler_params: Optional[dict] = None,
        multi_image_mode: str = "multidiffusion",
        preprocess_images: bool = True,
        ss_autocast_dtype: Optional[torch.dtype] = None,
        slat_autocast_dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, object]:
        images = self._prepare_conditioning_images(
            images,
            preprocess=preprocess_images,
            masks=masks,
            mask_aware_crop=mask_aware_crop,
            crop_padding=crop_padding,
        )
        cond = self.pipeline.get_cond(images)
        num_images = int(cond["cond"].shape[0])
        if num_images > 1:
            if multi_image_mode not in {"multidiffusion", "stochastic"}:
                raise ValueError(f"Unsupported multi_image_mode={multi_image_mode!r}")
            cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        ss_params = ss_sampler_params or {}
        slat_params = slat_sampler_params or {}
        if coords_override is not None:
            coords = self._validate_coords(coords_override)
            ss_latent_grid = None
        else:
            ss_context = _adapter_aware_sampler(
                self.pipeline.sparse_structure_sampler,
                num_images=num_images,
                mode=multi_image_mode,
                context_key="geoss_context",
            ) if num_images > 1 or geoss_context is not None else contextlib.nullcontext()
            with ss_context, _stage_autocast(self.device, ss_autocast_dtype):
                ss_latent_grid, coords = self.sample_sparse_structure_latent(
                    cond,
                    geoss_context=geoss_context,
                    sampler_params=ss_params,
                )
        if geovis_slat_context_factory is not None:
            if geovis_slat_context is not None:
                raise ValueError(
                    "Pass either geovis_slat_context or geovis_slat_context_factory, not both"
                )
            with _stage_autocast(self.device, slat_autocast_dtype):
                geovis_slat_context = geovis_slat_context_factory(coords)
        slat_context = _adapter_aware_sampler(
            self.pipeline.slat_sampler,
            num_images=num_images,
            mode=multi_image_mode,
            context_key="geovis_slat_context",
        ) if num_images > 1 or geovis_slat_context is not None else contextlib.nullcontext()
        with slat_context, _stage_autocast(self.device, slat_autocast_dtype):
            slat = self.sample_slat(cond, coords, geovis_slat_context=geovis_slat_context, sampler_params=slat_params)
        decoded = self.pipeline.decode_slat(slat, list(formats))
        decoded["coords"] = coords
        if ss_latent_grid is not None:
            # Conditioning-generated structure prior for explicit downstream
            # posterior completion.  This is not a dataset/ground-truth latent.
            decoded["ss_latent_grid"] = ss_latent_grid
        decoded["slat"] = slat
        decoded["conditioning_metadata"] = {
            "num_images": num_images,
            "multi_image_mode": multi_image_mode if num_images > 1 else "single_image",
            "mask_aware_crop": bool(mask_aware_crop),
            "crop_padding": float(crop_padding) if mask_aware_crop else None,
        }
        return decoded

    def _prepare_conditioning_images(
        self,
        images,
        *,
        preprocess: bool,
        masks: Optional[torch.Tensor] = None,
        mask_aware_crop: bool = False,
        crop_padding: float = 1.2,
    ):
        if isinstance(images, torch.Tensor):
            if images.ndim == 5:
                if images.shape[0] != 1:
                    raise ValueError(f"TRELLIS inference currently expects one object, got tensor {tuple(images.shape)}")
                images = images[0]
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError(f"TRELLIS conditioning tensor must be [N,3,H,W], got {tuple(images.shape)}")
            images = images.to(device=self.device, dtype=torch.float32).clamp(0.0, 1.0)
            if mask_aware_crop:
                if masks is None:
                    raise ValueError("mask_aware_crop requires conditioning masks.")
                images = mask_aware_trellis_crop(
                    images,
                    masks,
                    output_size=518,
                    padding=crop_padding,
                )
            elif images.shape[-2:] != (518, 518):
                images = torch.nn.functional.interpolate(
                    images, size=(518, 518), mode="bicubic", align_corners=False, antialias=True
                ).clamp(0.0, 1.0)
            return images
        if not isinstance(images, list) or not images:
            raise TypeError("TRELLIS conditioning images must be a non-empty PIL list or tensor batch.")
        return [self.pipeline.preprocess_image(image) for image in images] if preprocess else images

    def _validate_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 4 or coords.numel() == 0:
            raise ValueError(f"Stage-2 sparse coordinates must be non-empty [N,4], got {tuple(coords.shape)}")
        return coords.to(device=self.device, dtype=torch.int32).contiguous()

    @torch.no_grad()
    def generate_sparse_structure_prior(
        self,
        images,
        *,
        masks: Optional[torch.Tensor] = None,
        mask_aware_crop: bool = True,
        crop_padding: float = 1.2,
        seed: int = 42,
        sampler_params: Optional[dict] = None,
        multi_image_mode: str = "multidiffusion",
    ) -> Dict[str, torch.Tensor]:
        """Generate the conditioning-only TRELLIS structure prior without SLAT decoding."""
        prepared = self._prepare_conditioning_images(
            images,
            preprocess=True,
            masks=masks,
            mask_aware_crop=mask_aware_crop,
            crop_padding=crop_padding,
        )
        cond = self.pipeline.get_cond(prepared)
        num_images = int(cond["cond"].shape[0])
        if num_images > 1:
            if multi_image_mode not in {"multidiffusion", "stochastic"}:
                raise ValueError(f"Unsupported multi_image_mode={multi_image_mode!r}")
            cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        context = (
            _adapter_aware_sampler(
                self.pipeline.sparse_structure_sampler,
                num_images=num_images,
                mode=multi_image_mode,
                context_key="geoss_context",
            )
            if num_images > 1
            else contextlib.nullcontext()
        )
        with context:
            latent, coords = self.sample_sparse_structure_latent(
                cond,
                geoss_context=None,
                sampler_params=sampler_params or {},
            )
        return {
            "ss_latent_grid": latent,
            "coords": coords,
            "num_conditioning_images": torch.tensor(num_images, device=latent.device),
        }

    @torch.no_grad()
    def generate_completion_prior(
        self,
        images,
        *,
        masks: Optional[torch.Tensor] = None,
        mask_aware_crop: bool = True,
        crop_padding: float = 1.2,
        seed: int = 42,
        ss_sampler_params: Optional[dict] = None,
        slat_sampler_params: Optional[dict] = None,
        multi_image_mode: str = "multidiffusion",
    ) -> Dict[str, torch.Tensor]:
        """Generate conditioning-only SS geometry and SLAT appearance.

        No TRELLIS decoder is invoked and no dataset latent is consumed.  The
        returned sparse SLAT is a completion prior for RAPC appearance; RAPC's
        geometry prior remains exclusively the generated SS latent.
        """

        prepared = self._prepare_conditioning_images(
            images,
            preprocess=True,
            masks=masks,
            mask_aware_crop=mask_aware_crop,
            crop_padding=crop_padding,
        )
        cond = self.pipeline.get_cond(prepared)
        num_images = int(cond["cond"].shape[0])
        if num_images > 1:
            if multi_image_mode not in {"multidiffusion", "stochastic"}:
                raise ValueError(f"Unsupported multi_image_mode={multi_image_mode!r}")
            cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        ss_context = (
            _adapter_aware_sampler(
                self.pipeline.sparse_structure_sampler,
                num_images=num_images,
                mode=multi_image_mode,
                context_key="geoss_context",
            )
            if num_images > 1
            else contextlib.nullcontext()
        )
        with ss_context:
            ss_latent, coords = self.sample_sparse_structure_latent(
                cond,
                geoss_context=None,
                sampler_params=ss_sampler_params or {},
            )
        slat_context = (
            _adapter_aware_sampler(
                self.pipeline.slat_sampler,
                num_images=num_images,
                mode=multi_image_mode,
                context_key="geovis_slat_context",
            )
            if num_images > 1
            else contextlib.nullcontext()
        )
        with slat_context:
            slat = self.sample_slat(
                cond,
                coords,
                geovis_slat_context=None,
                sampler_params=slat_sampler_params or {},
            )
        features = slat.feats if hasattr(slat, "feats") else slat
        if not isinstance(features, torch.Tensor):
            raise TypeError("TRELLIS completion SLAT must expose tensor features")
        return {
            "ss_latent_grid": ss_latent,
            "coords": coords,
            "slat_feats": features,
            "num_conditioning_images": torch.tensor(
                num_images,
                device=ss_latent.device,
            ),
        }

    def sample_sparse_structure(self, cond: dict, *, geoss_context: Optional[Dict[str, torch.Tensor]], sampler_params: dict) -> torch.Tensor:
        _, coords = self.sample_sparse_structure_latent(
            cond,
            geoss_context=geoss_context,
            sampler_params=sampler_params,
        )
        return coords

    def sample_sparse_structure_latent(
        self,
        cond: dict,
        *,
        geoss_context: Optional[Dict[str, torch.Tensor]],
        sampler_params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the generated SS latent and decoded active coordinates."""
        flow_model = self.pipeline.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        noise = torch.randn(1, flow_model.in_channels, reso, reso, reso, device=self.device)
        params = {**self.pipeline.sparse_structure_sampler_params, **sampler_params}
        sample_kwargs = {**cond, **params, "verbose": True}
        if geoss_context is not None:
            sample_kwargs["geoss_context"] = _to_device(geoss_context, self.device)
        z_s = _sample_final_without_trajectory(
            self.pipeline.sparse_structure_sampler,
            flow_model,
            noise,
            **sample_kwargs,
        )
        decoder = self.pipeline.models["sparse_structure_decoder"]
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        if coords.numel() == 0:
            raise RuntimeError("TRELLIS sparse structure decoder produced zero active voxels.")
        return z_s, coords

    def sample_slat(self, cond: dict, coords: torch.Tensor, *, geovis_slat_context: Optional[Dict[str, torch.Tensor]], sampler_params: dict):
        from trellis.modules import sparse as sp

        flow_model = self.pipeline.models["slat_flow_model"]
        in_channels = _flow_in_channels(flow_model, name="slat_flow_model")
        if isinstance(flow_model, GeoVisTrellisSLATWrapper):
            adapter_channels = int(flow_model.velocity_adapter.slat_dim)
            if adapter_channels != in_channels:
                raise RuntimeError(
                    "Stage-3 adapter/TRELLIS SLat interface mismatch: "
                    f"adapter slat_dim={adapter_channels}, TRELLIS in_channels={in_channels}. "
                    "Use a checkpoint trained for this exact TRELLIS SLat model."
                )
        # The sampler requires only the native TRELLIS latent width here.  The
        # Stage-2 coordinates select the structure; they must not be mistaken
        # for SLat features or silently projected to a different channel count.
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], in_channels, device=self.device),
            coords=coords,
        )
        params = {**self.pipeline.slat_sampler_params, **sampler_params}
        sample_kwargs = {**cond, **params, "verbose": True}
        if geovis_slat_context is not None:
            sample_kwargs["geovis_slat_context"] = _to_device(geovis_slat_context, self.device)
        slat = _sample_final_without_trajectory(
            self.pipeline.slat_sampler,
            flow_model,
            noise,
            **sample_kwargs,
        )
        slat_feats = slat.feats if hasattr(slat, "feats") else slat
        if not isinstance(slat_feats, torch.Tensor) or slat_feats.ndim != 2 or slat_feats.shape[-1] != in_channels:
            shape = list(slat_feats.shape) if isinstance(slat_feats, torch.Tensor) else type(slat_feats).__name__
            raise RuntimeError(
                "TRELLIS SLat sampler returned an invalid latent contract: "
                f"expected [N,{in_channels}], got {shape}."
            )
        std = torch.tensor(self.pipeline.slat_normalization["std"], device=slat.device)[None]
        mean = torch.tensor(self.pipeline.slat_normalization["mean"], device=slat.device)[None]
        if std.shape[-1] != in_channels or mean.shape[-1] != in_channels:
            raise RuntimeError(
                "TRELLIS SLat normalization contract does not match its flow model: "
                f"in_channels={in_channels}, std={tuple(std.shape)}, mean={tuple(mean.shape)}."
            )
        return slat * std + mean

    def save_outputs(
        self,
        outputs: Dict[str, object],
        output_dir: Path,
        *,
        export_textured_glb: bool = False,
    ) -> Dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: Dict[str, object] = {}
        if isinstance(outputs.get("conditioning_metadata"), dict):
            saved["conditioning_metadata"] = dict(outputs["conditioning_metadata"])
        gaussian = outputs.get("gaussian")
        if isinstance(gaussian, list) and gaussian:
            path = output_dir / "asset_gaussian.ply"
            _atomic_asset_write(path, lambda temporary: gaussian[0].save_ply(str(temporary)))
            saved["gaussian_ply"] = str(path)
            saved["gaussian_statistics"] = gaussian_statistics(gaussian[0])
        mesh = outputs.get("mesh")
        if isinstance(mesh, list) and mesh:
            # MeshExtractResult is not an exportable trimesh object.  Preserve
            # the decoder's internal canonical frame for CD/F-score; TRELLIS'
            # public GLB conversion rotates vertices into y-up exchange space.
            internal_path = output_dir / "asset_mesh_internal.ply"
            _atomic_asset_write(
                internal_path,
                lambda temporary: write_internal_mesh(mesh[0], temporary, real_mode=True),
            )
            saved["mesh_internal_ply"] = str(internal_path)
            path = output_dir / "asset_mesh.glb"
            if hasattr(mesh[0], "export"):
                mesh[0].export(str(path))
                saved["mesh_glb"] = str(path)
            elif export_textured_glb and isinstance(gaussian, list) and gaussian:
                from trellis.utils.postprocessing_utils import to_glb

                textured = to_glb(gaussian[0], mesh[0])
                textured.export(str(path))
                saved["mesh_glb"] = str(path)
        # Persist plain tensors, not TRELLIS SparseTensor Python objects. This
        # keeps Stage-2→3 artifacts portable and compatible with weights-only
        # loading during isolated evaluation workers.
        slat = outputs.get("slat")
        slat_feats = slat.feats if hasattr(slat, "feats") else slat
        coords = outputs.get("coords")
        if not isinstance(coords, torch.Tensor) or not isinstance(slat_feats, torch.Tensor):
            raise TypeError("TRELLIS sampler must return tensor coordinates and SLAT features for Stage-2 handoff.")
        latent_path = output_dir / "trellis_latents.pt"
        latent_payload = {
            "coords": coords.detach().cpu().contiguous(),
            "slat": slat_feats.detach().cpu().contiguous(),
        }
        _atomic_asset_write(
            latent_path,
            lambda temporary: torch.save(latent_payload, temporary),
        )
        saved["latents"] = str(latent_path)
        return saved

    def _require_models(self, *names: str) -> None:
        missing = [name for name in names if name not in self.pipeline.models or self.pipeline.models[name] is None]
        if missing:
            raise RuntimeError(f"TRELLIS pipeline is missing required real decoder/flow models: {missing}")


def _to_device(context: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in context.items()}


def _flow_in_channels(flow_model: object, *, name: str) -> int:
    """Read and validate TRELLIS sampler metadata with a diagnostic failure."""
    value = getattr(flow_model, "in_channels", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"TRELLIS {name} must expose a positive integer in_channels for sampling; "
            f"got {value!r} from {type(flow_model).__name__}."
        )
    return value


def mask_aware_trellis_crop(
    images: torch.Tensor,
    masks: torch.Tensor,
    *,
    output_size: int = 518,
    padding: float = 1.2,
    threshold: float = 0.8,
) -> torch.Tensor:
    """Match TRELLIS alpha preprocessing for already-composited tensor views.

    Each foreground bbox becomes a padded square before resizing. Sampling
    outside the source image is black, matching PIL's out-of-bounds crop.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be [N,3,H,W], got {tuple(images.shape)}")
    if masks.ndim == 5:
        if masks.shape[0] != 1:
            raise ValueError("mask-aware TRELLIS crop accepts one object at a time.")
        masks = masks[0]
    if masks.ndim == 3:
        masks = masks[:, None]
    if masks.ndim != 4 or masks.shape[0] != images.shape[0] or masks.shape[1] != 1:
        raise ValueError(
            f"masks must be [N,1,H,W] for images {tuple(images.shape)}, got {tuple(masks.shape)}"
        )
    if output_size < 1 or padding <= 0.0:
        raise ValueError("output_size and padding must be positive")
    if masks.shape[-2:] != images.shape[-2:]:
        masks = torch.nn.functional.interpolate(
            masks.float(),
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    masks = masks.to(device=images.device, dtype=torch.float32)
    height, width = images.shape[-2:]
    axis = torch.linspace(
        -0.5,
        0.5,
        int(output_size),
        device=images.device,
        dtype=torch.float32,
    )
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    cropped = []
    for view in range(images.shape[0]):
        foreground = torch.nonzero(masks[view, 0] > float(threshold), as_tuple=False)
        if foreground.numel() == 0:
            raise RuntimeError(f"Conditioning mask {view} has no foreground for TRELLIS crop.")
        y_min, x_min = foreground.amin(dim=0).float()
        y_max, x_max = foreground.amax(dim=0).float()
        center_x = 0.5 * (x_min + x_max)
        center_y = 0.5 * (y_min + y_max)
        side = torch.maximum(x_max - x_min, y_max - y_min).clamp_min(1.0)
        side = side * float(padding)
        source_x = center_x + xx * side
        source_y = center_y + yy * side
        grid_x = 2.0 * source_x / max(1, width - 1) - 1.0
        grid_y = 2.0 * source_y / max(1, height - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)[None]
        crop = torch.nn.functional.grid_sample(
            images[view : view + 1],
            grid,
            mode="bicubic",
            padding_mode="zeros",
            align_corners=True,
        )
        cropped.append(crop.clamp(0.0, 1.0))
    return torch.cat(cropped, dim=0)


@torch.no_grad()
def _sample_final_without_trajectory(sampler, model, noise, **kwargs):
    """Run TRELLIS Flow-Euler sampling without retaining every ODE state.

    Upstream TRELLIS appends ``pred_x_prev`` and ``pred_x_0`` at every solver
    step even when callers consume only ``samples``.  For large sparse SLATs
    that is linear-in-step VRAM retention.  This path uses the sampler's exact
    ``sample_once`` transition and exact NumPy time grid, but owns only the
    current state.  Unknown sampler families retain their native behavior.
    """

    sample_once = getattr(sampler, "sample_once", None)
    if not callable(sample_once) or not hasattr(sampler, "sigma_min"):
        result = sampler.sample(model, noise, **kwargs)
        return result.samples
    params = dict(kwargs)
    steps = int(params.pop("steps", 50))
    rescale_t = float(params.pop("rescale_t", 1.0))
    params.pop("verbose", None)
    if steps <= 0:
        raise ValueError(f"TRELLIS sampler steps must be positive, got {steps}.")
    if rescale_t <= 0.0:
        raise ValueError(f"TRELLIS rescale_t must be positive, got {rescale_t}.")
    t_seq = np.linspace(1.0, 0.0, steps + 1)
    t_seq = rescale_t * t_seq / (1.0 + (rescale_t - 1.0) * t_seq)
    sample = noise
    for t, t_prev in zip(t_seq[:-1], t_seq[1:]):
        out = sample_once(model, sample, float(t), float(t_prev), **params)
        sample = out.pred_x_prev if hasattr(out, "pred_x_prev") else out["pred_x_prev"]
    return sample


@contextlib.contextmanager
def _adapter_aware_sampler(sampler, *, num_images: int, mode: str, context_key: str):
    """Apply multi-view conditioning while keeping adapters out of CFG's negative branch.

    TRELLIS' native multi-image patch directly invokes ``FlowEulerSampler`` and
    cannot distinguish adapter context between conditional and unconditional
    predictions.  This unified patch averages (or cycles) image-conditioned
    velocities, injects geometry only into those positive predictions, and
    preserves the sampler's configured CFG interval.
    """
    if mode not in {"multidiffusion", "stochastic"}:
        raise ValueError(f"Unsupported multi-image mode {mode!r}")
    old_inference_model = sampler._inference_model
    cursor = {"step": 0}

    def _patched(
        sampler_self,
        model,
        x_t,
        t,
        cond=None,
        neg_cond=None,
        cfg_strength=None,
        cfg_interval=None,
        **kwargs,
    ):
        adapter_context = kwargs.pop(context_key, None)
        if cond is None:
            return _call_trellis_flow_model(model, x_t, t, cond, context_key, adapter_context, kwargs)
        cfg_active = neg_cond is not None and cfg_strength is not None
        if cfg_active and cfg_interval is not None:
            cfg_active = bool(cfg_interval[0] <= t <= cfg_interval[1])
        positive_kwargs = dict(kwargs)
        # CFG amplifies only the learned positive-branch residual.  Native
        # TRELLIS baselines have no GeoVis context/wrapper, so adapter-only
        # control kwargs must never leak into SLatFlowModel.forward().
        if cfg_active and context_key == "geovis_slat_context" and adapter_context is not None:
            amplification = 1.0 + float(cfg_strength)
            if amplification <= 0:
                raise ValueError(f"SLAT CFG residual amplification must be positive, got {amplification}.")
            positive_kwargs["geovis_residual_scale"] = 1.0 / amplification
        count = min(num_images, int(cond.shape[0]))
        if mode == "stochastic" and count > 1:
            index = cursor["step"] % count
            cursor["step"] += 1
            indices = [index]
        else:
            indices = list(range(count))
        predictions = [
            _call_trellis_flow_model(
                model, x_t, t, cond[index : index + 1], context_key, adapter_context, positive_kwargs
            )
            for index in indices
        ]
        pred = predictions[0]
        for prediction in predictions[1:]:
            pred = pred + prediction
        if len(predictions) > 1:
            pred = pred / len(predictions)
        if not cfg_active:
            return pred
        neg_pred = _call_trellis_flow_model(model, x_t, t, neg_cond, context_key, None, kwargs)
        return (1.0 + cfg_strength) * pred - cfg_strength * neg_pred

    sampler._inference_model = MethodType(_patched, sampler)
    try:
        yield sampler
    finally:
        sampler._inference_model = old_inference_model


def _call_trellis_flow_model(model, x_t, t, cond, context_key: str, context, kwargs):
    batch_size = int(x_t.shape[0])
    t_tensor = torch.tensor([1000.0 * float(t)] * batch_size, device=x_t.device, dtype=torch.float32)
    if cond is not None and cond.shape[0] == 1 and batch_size > 1:
        cond = cond.repeat(batch_size, *([1] * (cond.ndim - 1)))
    model_kwargs = dict(kwargs)
    if context_key == "geovis_slat_context" and context is None:
        # Defensive boundary: this keyword belongs to
        # GeoVisTrellisSLATWrapper, not native TRELLIS SLatFlowModel.  It may
        # be present only on a context-bearing conditional adapter branch.
        model_kwargs.pop("geovis_residual_scale", None)
    if context is not None:
        model_kwargs[context_key] = context
    return model(x_t, t_tensor, cond, **model_kwargs)
