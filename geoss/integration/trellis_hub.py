from __future__ import annotations

import os
import json
from pathlib import Path

import torch


def resolve_local_hf_snapshot(
    model_or_path: str,
    cache_root: str | os.PathLike[str],
    *,
    required_file: str,
    validate_trellis_pipeline: bool = False,
) -> str:
    """Resolve a complete local Hugging Face snapshot without network access."""

    explicit = Path(model_or_path).expanduser()
    candidates: list[Path] = []
    if explicit.is_dir():
        candidates.append(explicit)
    elif "/" in model_or_path:
        repository = (
            Path(cache_root).expanduser()
            / f"models--{model_or_path.replace('/', '--')}"
        )
        ref = repository / "refs" / "main"
        if ref.is_file():
            revision = ref.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(repository / "snapshots" / revision)
        snapshots = repository / "snapshots"
        if snapshots.is_dir():
            candidates.extend(
                sorted(
                    (path for path in snapshots.iterdir() if path.is_dir()),
                    key=lambda path: path.stat().st_mtime_ns,
                    reverse=True,
                )
            )
        # Some pre-existing downloads use a repository-root materialization.
        candidates.append(repository)
    else:
        raise FileNotFoundError(
            f"Expected a local model directory or Hugging Face repo id, got {model_or_path!r}"
        )

    seen: set[str] = set()
    failures: list[str] = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        required = candidate / required_file
        if not required.is_file():
            failures.append(f"{candidate}: missing {required_file}")
            continue
        if validate_trellis_pipeline:
            try:
                _validate_local_trellis_pipeline(candidate)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                failures.append(f"{candidate}: {exc}")
                continue
        return str(candidate.resolve())

    detail = "; ".join(failures) if failures else "no local candidates"
    raise FileNotFoundError(
        f"No complete local snapshot for {model_or_path!r} under "
        f"{Path(cache_root).expanduser()}: {detail}. Production training does not download implicitly."
    )


def _validate_local_trellis_pipeline(snapshot: Path) -> None:
    """Verify that every model referenced by pipeline.json is materialized."""

    pipeline_file = snapshot / "pipeline.json"
    payload = json.loads(pipeline_file.read_text(encoding="utf-8"))
    models = payload["args"]["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("pipeline.json has no model mapping")
    for name, relative in models.items():
        prefix = snapshot / str(relative)
        config = prefix.with_suffix(".json")
        weights = prefix.with_suffix(".safetensors")
        if not config.is_file() or not weights.is_file():
            raise FileNotFoundError(
                f"incomplete model {name!r}: expected {config.name} and {weights.name}"
            )


def configure_trellis_hub(args) -> dict[str, str | None]:
    """Resolve TRELLIS' DINO dependency locally before model construction.

    Native TRELLIS calls ``torch.hub.load('facebookresearch/dinov2', ...)``.
    Even with a populated torch cache that API can query GitHub to resolve the
    default branch.  Rewriting only that repository call to ``source='local'``
    makes training reproducible when the verified checkout is already present.
    """

    hub_dir = resolve_torch_hub_dir(args)
    if hub_dir is not None:
        torch.hub.set_dir(str(hub_dir))
        os.environ["TORCH_HUB_DIR"] = str(hub_dir)
        os.environ.setdefault(
            "TORCH_HOME", str(hub_dir.parent if hub_dir.name == "hub" else hub_dir)
        )
    dinov2_repo = resolve_dinov2_repo(args, hub_dir)
    if dinov2_repo is not None:
        patch_torch_hub_for_local_dinov2(dinov2_repo)
    return {
        "torch_hub_dir": str(hub_dir) if hub_dir is not None else None,
        "dinov2_repo": str(dinov2_repo) if dinov2_repo is not None else None,
    }


def resolve_torch_hub_dir(args) -> Path | None:
    candidates: list[Path] = []
    for value in (
        getattr(args, "torch_hub_dir", None),
        os.environ.get("TORCH_HUB_DIR"),
    ):
        if value:
            candidates.append(Path(value).expanduser())
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        home = Path(torch_home).expanduser()
        candidates.extend([home / "hub", home])
    candidates.extend(
        [
            Path.home() / ".cache" / "torch" / "hub",
            Path("/mnt/sda/hf/.cache/torch/hub"),
            Path("/mnt/sda3/yu/checkpoints/hub/hub"),
            Path("/mnt/sda3/yu/checkpoints/hub"),
        ]
    )
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "facebookresearch_dinov2_main" / "hubconf.py").exists():
            return candidate
        nested = candidate / "hub"
        if (nested / "facebookresearch_dinov2_main" / "hubconf.py").exists():
            return nested
        if candidate.exists():
            return candidate
    return None


def resolve_dinov2_repo(args, hub_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    explicit = getattr(args, "dinov2_repo", None)
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if hub_dir is not None:
        candidates.append(hub_dir / "facebookresearch_dinov2_main")
        candidates.extend(sorted(hub_dir.glob("facebookresearch_dinov2*")))
    for candidate in candidates:
        if (candidate / "hubconf.py").exists():
            return candidate
    return None


def patch_torch_hub_for_local_dinov2(local_repo: Path) -> None:
    if getattr(torch.hub, "_geoss_local_dinov2_patch", False):
        return
    original_load = torch.hub.load

    def load(repo_or_dir, model, *args, **kwargs):
        if str(repo_or_dir).rstrip("/") == "facebookresearch/dinov2":
            local_kwargs = dict(kwargs)
            local_kwargs.pop("trust_repo", None)
            local_kwargs.pop("force_reload", None)
            return original_load(
                str(local_repo), model, *args, source="local", **local_kwargs
            )
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load
    torch.hub._geoss_local_dinov2_patch = True
