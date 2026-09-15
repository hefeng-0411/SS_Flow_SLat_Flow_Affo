#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROJECT_ROOT="/mnt/sda/hf/MVG/Base/SS_Flow_SLat_Flow_Affo"
PYTHON_BIN="/mnt/sda/hf/miniconda3/envs/trellis/bin/python"
TORCHRUN_BIN="/mnt/sda/hf/miniconda3/envs/trellis/bin/torchrun"
PROCESS_COUNT="${NPROC_PER_NODE:-4}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="${CC:-/usr/bin/gcc-10}"
export CXX="${CXX:-/usr/bin/g++-10}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/sda/hf/.cache/torch_extensions/geoss_gsplat14_cu118_gcc10}"
export MAX_JOBS="${MAX_JOBS:-4}"
STAGE="${1:-all}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/sda/hf/MVG/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97}"
DEPTH_ROOT="${DEPTH_ROOT:-${DATASET_ROOT}/affostruction_depth}"
OFFICIAL_CHECKPOINT_DIR="${OFFICIAL_CHECKPOINT_DIR:-/mnt/sda/hf/MVG/Base/Affostruction/checkpoints/reconstruction}"
OFFICIAL_CHECKPOINT="${OFFICIAL_CHECKPOINT:-${OFFICIAL_CHECKPOINT_DIR}/model.safetensors}"
AFFOSTRUCTION_ROOT="${AFFOSTRUCTION_ROOT:-/mnt/sda/hf/MVG/Base/Affostruction}"
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$PROJECT_ROOT"

run_ss() {
  if [[ ! -s "${OFFICIAL_CHECKPOINT}" ]]; then
    echo "Official initialization missing: ${OFFICIAL_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ ! -d "${DEPTH_ROOT}/train" || ! -d "${DEPTH_ROOT}/test" ]]; then
    echo "Exact RGBD cache missing under ${DEPTH_ROOT}" >&2
    echo "Run: ./train_ddp_affostruction.sh cache_depth" >&2
    exit 2
  fi
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$PROCESS_COUNT" \
    scripts/train_affostruction_ss_official.py \
    --config configs/affostruction_ss_official.yaml \
    --affostruction-root "${AFFOSTRUCTION_ROOT}" \
    --trellis-root /mnt/sda/hf/MVG/Base/TRELLIS \
    --meshfleet-root /mnt/sda/hf/MVG/Base/MeshFleet \
    --dataset-root "${DATASET_ROOT}" \
    --depth-root "${DEPTH_ROOT}" \
    --initial-checkpoint "${OFFICIAL_CHECKPOINT}" \
    --output-dir outputs/affostruction_ss_official \
    --precision fp16 \
    --batch-size "${SS_BATCH_SIZE:-8}" \
    --gradient-checkpointing \
    "$@"
}

run_slat() {
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$PROCESS_COUNT" \
    scripts/train_geovis_slat.py \
    --config configs/real_train_slat_only.yaml \
    --real_train \
    --affostruction_root "${AFFOSTRUCTION_ROOT}" \
    --meshfleet_root "${DATASET_ROOT}" \
    --trellis_root /mnt/sda/hf/MVG/Base/TRELLIS \
    --trellis_model_path microsoft/TRELLIS-image-large \
    --vggt_root /mnt/sda/hf/MVG/Base/vggt \
    --vggt_pretrained facebook/VGGT-1B \
    --output_dir outputs/affostruction_slat_fp32 \
    --amp true \
    --amp_dtype fp16 \
    --gradient_checkpointing true \
    "$@"
}

cache_depth() {
  for SPLIT_NAME in train test; do
    "$TORCHRUN_BIN" --standalone --nproc_per_node="$PROCESS_COUNT" \
      scripts/cache_affostruction_depth.py \
      --data-root "${DATASET_ROOT}" \
      --output-root "${DEPTH_ROOT}" \
      --split "${SPLIT_NAME}" \
      --render-set renders \
      --image-size 224 \
      --minimum-mask-iou 0.90
  done
}

case "$STAGE" in
  ss)
    run_ss "$@"
    ;;
  slat)
    run_slat "$@"
    ;;
  cache_depth)
    cache_depth
    ;;
  all)
    if [[ $# -ne 0 ]]; then
      "$PYTHON_BIN" -c "raise SystemExit('Stage-specific arguments require stage ss or slat')"
    fi
    cache_depth
    run_ss
    run_slat
    ;;
  *)
    "$PYTHON_BIN" -c "raise SystemExit('Usage: train_ddp_affostruction.sh {cache_depth|ss|slat|all} [stage arguments]')"
    ;;
esac
