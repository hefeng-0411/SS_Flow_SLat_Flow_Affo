#!/usr/bin/env python3
"""Multi-GPU held-out asset evaluation and aggregation for Affostruction runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geoss.datasets.meshfleet_trellis_dataset import MeshFleetTrellisDataset
from geoss.metrics.geometry_metrics import geometry_metrics
from scripts.evaluate_meshfleet_sequence import _aggregate


METHODS = ("original_trellis", "ss_only", "slat_only", "ss_slat")


def main() -> None:
    args = _build_parser().parse_args()
    uids = _read_manifest(Path(args.uid_manifest))
    if args.max_samples > 0:
        uids = uids[: args.max_samples]
    if args.aggregate_only:
        _aggregate_results(args, uids)
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Held-out Gaussian evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    methods = tuple(args.methods)
    shard = [(method, uid) for method in methods for uid in uids]
    shard = shard[rank::world_size]
    completed = failed = skipped = 0
    for method, uid in shard:
        asset_dir = Path(args.output_root) / method / uid
        inference_metrics = asset_dir / "metrics.json"
        eval_dir = asset_dir / "evaluation"
        metrics_path = eval_dir / "geovis_slat_metrics.json"
        if not inference_metrics.is_file():
            _write_failure(eval_dir, method, uid, "missing inference metrics")
            failed += 1
            continue
        inference = json.loads(inference_metrics.read_text(encoding="utf-8"))
        if inference.get("status") != "ok":
            _write_failure(
                eval_dir,
                method,
                uid,
                f"inference status is {inference.get('status')!r}",
            )
            failed += 1
            continue
        if (
            not args.overwrite
            and metrics_path.is_file()
            and _completed_evaluation(metrics_path)
        ):
            skipped += 1
            continue
        gaussian = asset_dir / "asset_gaussian.ply"
        mesh = asset_dir / "asset_mesh_internal.ply"
        command = [
            sys.executable,
            "scripts/eval_geovis_slat.py",
            "--input_dir",
            str(asset_dir),
            "--output_dir",
            str(eval_dir),
            "--ablation",
            method,
            "--real_eval",
            "--render_eval",
            "true",
            "--gaussian_ply",
            str(gaussian),
            "--pred_mesh",
            str(mesh),
            "--inference_metrics",
            str(inference_metrics),
            "--meshfleet_root",
            args.data_root,
            "--meshfleet_split",
            args.split,
            "--meshfleet_uid",
            uid,
            "--num_views",
            str(args.eval_num_views),
            "--conditioning_num_views",
            str(args.num_views),
            "--image_size",
            str(args.image_size),
            "--conditioning_view_set",
            args.conditioning_view_set,
            "--eval_view_set",
            args.eval_view_set,
            "--geometry_samples",
            str(args.geometry_samples),
            "--geometry_seed",
            str(args.geometry_seed),
            "--fscore_threshold",
            str(args.fscore_threshold),
            "--device",
            f"cuda:{local_rank}",
            "--save_visuals",
            str(args.save_visuals).lower(),
        ]
        if args.category:
            command += ["--meshfleet_category", args.category]
        eval_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds,
                check=False,
            )
            (eval_dir / "evaluation.log").write_text(result.stdout, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(
                    f"asset evaluator exited {result.returncode}; see {eval_dir / 'evaluation.log'}"
                )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics.update(_evaluate_ss_occupancy(args, uid, asset_dir))
            metrics["inference_latency_seconds"] = inference.get("latency_seconds")
            metrics["inference_peak_allocated_cuda_gb"] = inference.get(
                "peak_allocated_cuda_gb"
            )
            _atomic_json(metrics_path, metrics)
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "evaluation_complete",
                        "rank": rank,
                        "method": method,
                        "uid": uid,
                        "progress": f"{completed + failed + skipped}/{len(shard)}",
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            _write_failure(
                eval_dir,
                method,
                uid,
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )
            failed += 1
            print(
                json.dumps(
                    {
                        "event": "evaluation_failed",
                        "rank": rank,
                        "method": method,
                        "uid": uid,
                        "progress": f"{completed + failed + skipped}/{len(shard)}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
            }
        ),
        flush=True,
    )


def _evaluate_ss_occupancy(args, uid: str, asset_dir: Path) -> dict:
    prediction_path = asset_dir / "predicted_ss_occ.npz"
    if not prediction_path.is_file():
        return {"ss_metrics_valid": False, "ss_metrics_reason": "missing_prediction"}
    dataset = MeshFleetTrellisDataset(
        args.data_root,
        split=args.split,
        category=args.category,
        num_views=1,
        image_size=16,
        render_set=args.eval_view_set,
        repeat_views_if_insufficient=False,
        uid_manifest=[uid],
        occ_resolution=args.occ_resolution,
        load_3d_modalities=True,
    )
    sample = dataset.get_by_uid(uid)
    if "gt_occ" not in sample:
        return {"ss_metrics_valid": False, "ss_metrics_reason": "missing_gt_occupancy"}
    pred = _canonical_occupancy(
        torch.from_numpy(np.load(prediction_path, allow_pickle=False)["occ"]),
        "prediction",
    )
    gt = _canonical_occupancy(sample["gt_occ"], "ground_truth")
    if pred.shape != gt.shape:
        return {
            "ss_metrics_valid": False,
            "ss_metrics_reason": f"shape_mismatch:{tuple(pred.shape)}!={tuple(gt.shape)}",
        }
    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()
    union = (pred | gt).sum().float().clamp_min(1)
    pred_surface = _surface_points(pred)
    gt_surface = _surface_points(gt)
    result = {
        "ss_IoU": float(tp / union),
        "ss_Dice": float(2 * tp / (2 * tp + fp + fn).clamp_min(1)),
        "ss_precision": float(tp / (tp + fp).clamp_min(1)),
        "ss_recall": float(tp / (tp + fn).clamp_min(1)),
        "ss_predicted_occupied_voxels": int(pred.sum()),
        "ss_gt_occupied_voxels": int(gt.sum()),
        "ss_metrics_valid": True,
    }
    if pred_surface.numel() and gt_surface.numel():
        surface = geometry_metrics(
            pred_surface,
            gt_surface,
            threshold=1.5 / args.occ_resolution,
            real_mode=True,
            chunk_size=4096,
        )
        result.update({f"ss_surface_{key}": value for key, value in surface.items()})
    return result


def _canonical_occupancy(value: torch.Tensor, name: str) -> torch.Tensor:
    occupancy = value.bool()
    while occupancy.ndim > 3 and occupancy.shape[0] == 1:
        occupancy = occupancy[0]
    if occupancy.ndim != 3:
        raise ValueError(
            f"{name} occupancy must be [D,H,W] with optional singleton leading axes, "
            f"got {tuple(value.shape)}"
        )
    return occupancy


def _completed_evaluation(metrics_path: Path) -> bool:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return metrics.get("official_metrics") is True and metrics.get("ss_metrics_valid") is True


def _surface_points(occupancy: torch.Tensor) -> torch.Tensor:
    value = occupancy[None, None].float()
    eroded = -torch.nn.functional.max_pool3d(-value, 3, stride=1, padding=1)
    surface = occupancy & (eroded[0, 0] < 0.5)
    indices = torch.nonzero(surface, as_tuple=False).float()
    if not indices.numel():
        return indices
    xyz = indices[:, [2, 1, 0]]
    return (xyz + 0.5) / occupancy.shape[-1] - 0.5


def _aggregate_results(args, uids: list[str]) -> None:
    rows = []
    for index, uid in enumerate(uids):
        for method in args.methods:
            asset_dir = Path(args.output_root) / method / uid
            eval_path = asset_dir / "evaluation" / "geovis_slat_metrics.json"
            inference_path = asset_dir / "metrics.json"
            row = {
                "index": index,
                "uid": uid,
                "ablation": method,
                "population_manifested": True,
            }
            if not eval_path.is_file():
                row.update({"status": "failed", "error": "missing evaluation metrics"})
                rows.append(row)
                continue
            metrics = json.loads(eval_path.read_text(encoding="utf-8"))
            if metrics.get("status") == "failed":
                row.update({"status": "failed", "error": metrics.get("error")})
                rows.append(row)
                continue
            row["status"] = "ok"
            row["asset_official_metrics"] = metrics.get("official_metrics") is True
            row["asset_evaluation_protocol"] = metrics.get("evaluation_protocol")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[f"asset_{key}"] = value
            if inference_path.is_file():
                inference = json.loads(inference_path.read_text(encoding="utf-8"))
                row["latency_seconds"] = inference.get("latency_seconds")
                row["peak_vram_gb"] = inference.get("peak_allocated_cuda_gb")
            rows.append(row)
    summary = _aggregate(
        rows,
        expected_indices=list(range(len(uids))),
        expected_ablations=list(args.methods),
    )
    output_root = Path(args.output_root)
    _atomic_json(output_root / "summary.json", summary)
    with (output_root / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    fieldnames = sorted({key for row in rows for key in row})
    with (output_root / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.print_full_summary:
        print(json.dumps(summary, indent=2), flush=True)
    else:
        print(
            json.dumps(
                {
                    "summary": str(output_root / "summary.json"),
                    "rows_jsonl": str(output_root / "rows.jsonl"),
                    "rows_csv": str(output_root / "rows.csv"),
                    "expected_object_count": len(uids),
                    "by_ablation": {
                        method: {
                            key: values[key]
                            for key in ("num_ok", "num_failed", "num_missing", "complete")
                        }
                        for method, values in summary["by_ablation"].items()
                    },
                },
                indent=2,
            ),
            flush=True,
        )


def _write_failure(
    eval_dir: Path, method: str, uid: str, error: str, trace: str | None = None
) -> None:
    _atomic_json(
        eval_dir / "geovis_slat_metrics.json",
        {
            "status": "failed",
            "ablation": method,
            "uid": uid,
            "official_metrics": False,
            "error": error,
            "traceback": trace,
        },
    )


def _read_manifest(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    uids = payload.get("uids") if isinstance(payload, dict) else payload
    if not isinstance(uids, list) or not all(isinstance(uid, str) for uid in uids):
        raise ValueError(f"Invalid UID manifest: {path}")
    return uids


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shard held-out rendering/geometry metrics over torchrun GPUs"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--uid-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Evaluate and aggregate only these manifested ablations.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--category")
    parser.add_argument("--conditioning-view-set", choices=("renders", "renders_cond"), default="renders")
    parser.add_argument("--eval-view-set", choices=("renders_eval_70", "renders_eval_90"), default="renders_eval_70")
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--eval-num-views", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--occ-resolution", type=int, default=64)
    parser.add_argument("--geometry-samples", type=int, default=100000)
    parser.add_argument("--geometry-seed", type=int, default=20260720)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--save-visuals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--print-full-summary", action="store_true")
    return parser


if __name__ == "__main__":
    main()
