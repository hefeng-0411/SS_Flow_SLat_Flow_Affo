#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/mnt/sda/hf/MVG/Base/SS_Flow_SLat_Flow_Affo"
PYTHON_BIN="/mnt/sda/hf/miniconda3/envs/trellis/bin/python"
TORCHRUN_BIN="/mnt/sda/hf/miniconda3/envs/trellis/bin/torchrun"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
PROCESS_COUNT="${NPROC_PER_NODE:-${#VISIBLE_GPU_ARRAY[@]}}"
if [[ "${PROCESS_COUNT}" -ne "${#VISIBLE_GPU_ARRAY[@]}" ]]; then
  echo "NPROC_PER_NODE=${PROCESS_COUNT} must equal visible GPU count ${#VISIBLE_GPU_ARRAY[@]}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CC="${CC:-/usr/bin/gcc-10}"
export CXX="${CXX:-/usr/bin/g++-10}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/sda/hf/.cache/torch_extensions/geoss_gsplat14_cu118_gcc10}"
export MAX_JOBS="${MAX_JOBS:-4}"

DATA_ROOT="${DATA_ROOT:-/mnt/sda/hf/MVG/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97}"
DEPTH_ROOT="${DEPTH_ROOT:-${DATA_ROOT}/affostruction_depth}"
UID_MANIFEST="${UID_MANIFEST:-outputs/affostruction_dataset_audit/test_evaluation_uids.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/affostruction_evaluation/test}"
SS_CHECKPOINT="${SS_CHECKPOINT:-outputs/affostruction_ss_official/affostruction_official_ss_last.pt}"
SLAT_CHECKPOINT="${SLAT_CHECKPOINT:-outputs/affostruction_slat_fp32/geovis_slat_adapter_last.pt}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-false}"
SAVE_VISUALS="${SAVE_VISUALS:-false}"
INFERENCE_METHODS="${INFERENCE_METHODS:-original_trellis ss_only}"
RUN_EVALUATION="${RUN_EVALUATION:-true}"
MINIMUM_FREE_DISK_GB="${MINIMUM_FREE_DISK_GB:-2.0}"
ESTIMATED_ASSET_MB="${ESTIMATED_ASSET_MB:-48}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-false}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

METHODS_TO_RUN=()
if [[ "${INFERENCE_METHODS}" != "none" ]]; then
  read -r -a METHODS_TO_RUN <<< "${INFERENCE_METHODS}"
fi
for METHOD in "${METHODS_TO_RUN[@]}"; do
  case "${METHOD}" in
    original_trellis|ss_only|slat_only|ss_slat) ;;
    *) echo "Unsupported INFERENCE_METHODS entry: ${METHOD}" >&2; exit 2 ;;
  esac
done
if [[ "${RUN_EVALUATION}" != "true" && "${RUN_EVALUATION}" != "false" ]]; then
  echo "RUN_EVALUATION must be true or false" >&2
  exit 2
fi
if [[ "${PREFLIGHT_ONLY}" != "true" && "${PREFLIGHT_ONLY}" != "false" ]]; then
  echo "PREFLIGHT_ONLY must be true or false" >&2
  exit 2
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  OVERWRITE_FLAG="--overwrite"
else
  OVERWRITE_FLAG="--no-overwrite"
fi
if [[ "${SAVE_VISUALS}" == "true" ]]; then
  VISUALS_FLAG="--save-visuals"
else
  VISUALS_FLAG="--no-save-visuals"
fi

EVAL_METHODS="${METHODS_TO_RUN[*]}" EVAL_SS_CHECKPOINT="${SS_CHECKPOINT}" \
EVAL_SLAT_CHECKPOINT="${SLAT_CHECKPOINT}" "${PYTHON_BIN}" - <<'PY'
import os
import torch
methods = os.environ["EVAL_METHODS"].split()
checks = []
if any(method in {"ss_only", "ss_slat"} for method in methods):
    checks.append((
        os.environ["EVAL_SS_CHECKPOINT"],
        {
            "affostruction_official_ss_flow_768x12_v1": "official_fp32_master_fp16_amp_v1",
            "affostruction_direct_ss_cross_attention_v1": "fp32_master_bf16_forward_v2",
        },
    ))
if any(method in {"slat_only", "ss_slat"} for method in methods):
    checks.append((
        os.environ["EVAL_SLAT_CHECKPOINT"],
        {"vggt_affostruction_sparse_residual_slat_flow_768x12_v1": "frozen_trellis_sparse_image_flow_fp32_master_v1"},
    ))
for path, supported in checks:
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    architecture = state.get("architecture_version")
    assert architecture in supported, (path, architecture, sorted(supported))
    assert state.get("optimization_numerics_version") == supported[architecture]
    assert int(state["step"]) > 0, (path, state.get("step"))
    del state
print("Affostruction evaluation checkpoint preflight passed")
PY

EVAL_DATA_ROOT="${DATA_ROOT}" EVAL_UID_MANIFEST="${UID_MANIFEST}" \
EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}" EVAL_MAX_SAMPLES="${MAX_SAMPLES}" \
EVAL_METHODS="${METHODS_TO_RUN[*]}" EVAL_MIN_FREE_GB="${MINIMUM_FREE_DISK_GB}" \
EVAL_ESTIMATED_ASSET_MB="${ESTIMATED_ASSET_MB}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import shutil
from pathlib import Path

project = Path.cwd()
manifest_path = project / os.environ["EVAL_UID_MANIFEST"]
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
uids = payload.get("uids") if isinstance(payload, dict) else payload
maximum = int(os.environ["EVAL_MAX_SAMPLES"])
if maximum > 0:
    uids = uids[:maximum]
output = project / os.environ["EVAL_OUTPUT_ROOT"]
methods = os.environ["EVAL_METHODS"].split()

def complete(method, uid):
    directory = output / method / uid
    metric_path = directory / "metrics.json"
    required = (
        directory / "asset_gaussian.ply",
        directory / "asset_mesh_internal.ply",
        directory / "predicted_ss_occ.npz",
        directory / "trellis_latents.pt",
    )
    if not metric_path.is_file() or not all(p.is_file() and p.stat().st_size > 0 for p in required):
        return False
    try:
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return metric.get("status") == "ok" and metric.get("method") == method and metric.get("uid") == uid

missing = {method: sum(not complete(method, uid) for uid in uids) for method in methods}
missing_total = sum(missing.values())
estimated_gib = missing_total * float(os.environ["EVAL_ESTIMATED_ASSET_MB"]) / 1024.0
reserve_gib = float(os.environ["EVAL_MIN_FREE_GB"])
free_gib = shutil.disk_usage(output).free / 2**30
report = {
    "objects": len(uids),
    "methods_to_run": methods,
    "missing_by_method": missing,
    "estimated_new_output_gib": estimated_gib,
    "free_gib": free_gib,
    "safety_reserve_gib": reserve_gib,
}
(output / "resume_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if free_gib < estimated_gib + reserve_gib:
    raise SystemExit(
        f"Insufficient storage: need approximately {estimated_gib + reserve_gib:.2f} GiB "
        f"including reserve, only {free_gib:.2f} GiB is available."
    )
PY

if [[ "${PREFLIGHT_ONLY}" == "true" ]]; then
  echo "Preflight complete; no inference or evaluation was started"
  exit 0
fi

COMMON_INFERENCE_ARGS=(
  --data-root "${DATA_ROOT}"
  --depth-root "${DEPTH_ROOT}"
  --uid-manifest "${UID_MANIFEST}"
  --output-root "${OUTPUT_ROOT}"
  --split test
  --conditioning-view-set renders
  --num-views 8
  --conditioning-image-size 518
  --occ-resolution 64
  --max-samples "${MAX_SAMPLES}"
  --seed 42
  --multi-image-mode multidiffusion
  --ss-steps 12
  --ss-cfg-strength 7.5
  --slat-steps 12
  --slat-cfg-strength 3.0
  --ss-amp-dtype fp16
  --slat-amp-dtype fp16
  --slat-spconv-algo native
  --minimum-free-disk-gb "${MINIMUM_FREE_DISK_GB}"
  --ss-checkpoint "${SS_CHECKPOINT}"
  --slat-checkpoint "${SLAT_CHECKPOINT}"
  --trellis-root /mnt/sda/hf/MVG/Base/TRELLIS
  --affostruction-root /mnt/sda/hf/MVG/Base/Affostruction
  --trellis-model-path microsoft/TRELLIS-image-large
  --vggt-root /mnt/sda/hf/MVG/Base/vggt
  --vggt-pretrained facebook/VGGT-1B
  --hf-cache-root /mnt/sda/hf/.cache/huggingface/hub
  --torch-hub-dir /mnt/sda/hf/.cache/torch/hub
  --dinov2-repo /mnt/sda/hf/.cache/torch/hub/facebookresearch_dinov2_main
  "${OVERWRITE_FLAG}"
)

for METHOD in "${METHODS_TO_RUN[@]}"; do
  echo "Starting distributed inference: ${METHOD}"
  "${TORCHRUN_BIN}" --standalone --nproc_per_node="${PROCESS_COUNT}" \
    scripts/infer_affostruction_batch.py \
    --method "${METHOD}" \
    "${COMMON_INFERENCE_ARGS[@]}"
done

if [[ "${RUN_EVALUATION}" == "false" ]]; then
  echo "Requested inference stages complete; RUN_EVALUATION=false"
  exit 0
fi

EVAL_UID_MANIFEST="${UID_MANIFEST}" EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}" \
EVAL_MAX_SAMPLES="${MAX_SAMPLES}" EVAL_METHODS="${METHODS_TO_RUN[*]}" \
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

project = Path.cwd()
payload = json.loads((project / os.environ["EVAL_UID_MANIFEST"]).read_text(encoding="utf-8"))
uids = payload.get("uids") if isinstance(payload, dict) else payload
maximum = int(os.environ["EVAL_MAX_SAMPLES"])
if maximum > 0:
    uids = uids[:maximum]
root = project / os.environ["EVAL_OUTPUT_ROOT"]
missing = []
for method in os.environ["EVAL_METHODS"].split():
    for uid in uids:
        directory = root / method / uid
        metric_path = directory / "metrics.json"
        try:
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            missing.append((method, uid, "missing_or_invalid_metrics"))
            continue
        required = (directory / "asset_gaussian.ply", directory / "asset_mesh_internal.ply", directory / "predicted_ss_occ.npz")
        if metric.get("status") != "ok" or not all(path.is_file() and path.stat().st_size > 0 for path in required):
            missing.append((method, uid, "incomplete_assets"))
if missing:
    (root / "inference_incomplete.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
    raise SystemExit(f"Inference population is incomplete ({len(missing)} entries); evaluation will not start.")
print(f"Inference population validation passed: {len(uids)} objects x {len(os.environ['EVAL_METHODS'].split())} methods")
PY

"${PYTHON_BIN}" -c "import gsplat; from gsplat.cuda._backend import _C; assert _C is not None; print('gsplat backend ready:', gsplat.__version__)"

echo "Starting distributed held-out rendering and geometry evaluation"
"${TORCHRUN_BIN}" --standalone --nproc_per_node="${PROCESS_COUNT}" \
  scripts/evaluate_affostruction_batch.py \
  --data-root "${DATA_ROOT}" \
  --uid-manifest "${UID_MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --methods "${METHODS_TO_RUN[@]}" \
  --split test \
  --conditioning-view-set renders \
  --eval-view-set renders_eval_70 \
  --num-views 8 \
  --eval-num-views 12 \
  --image-size 256 \
  --occ-resolution 64 \
  --geometry-samples 100000 \
  --geometry-seed 20260720 \
  --fscore-threshold 0.01 \
  --max-samples "${MAX_SAMPLES}" \
  --timeout-seconds 1800 \
  "${VISUALS_FLAG}" \
  "${OVERWRITE_FLAG}"

"${PYTHON_BIN}" scripts/evaluate_affostruction_batch.py \
  --data-root "${DATA_ROOT}" \
  --uid-manifest "${UID_MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --methods "${METHODS_TO_RUN[@]}" \
  --split test \
  --max-samples "${MAX_SAMPLES}" \
  --aggregate-only

EVAL_OUTPUT_ROOT="${OUTPUT_ROOT}" EVAL_METHODS="${METHODS_TO_RUN[*]}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

summary_path = Path(os.environ["EVAL_OUTPUT_ROOT"]) / "summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
incomplete = {
    method: values
    for method, values in summary["by_ablation"].items()
    if not values.get("complete") or values.get("num_failed") or values.get("num_missing")
}
if incomplete:
    raise SystemExit(f"Evaluation population is incomplete: {incomplete}")
print(f"Evaluation population validation passed for: {os.environ['EVAL_METHODS']}")
PY

echo "Evaluation complete: ${OUTPUT_ROOT}/summary.json"
