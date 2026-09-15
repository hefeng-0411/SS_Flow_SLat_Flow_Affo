#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

PROJECT_ROOT="/mnt/sda/hf/MVG/Base/SS_Flow_SLat_Flow_Affo"
cd "$PROJECT_ROOT"

exec ./train_ddp_affostruction.sh ss \
  --min-views 1 \
  --max-views 8 \
  --grid-resolution 16 \
  --num-workers 8 \
  --persistent-workers \
  --pin-memory "$@"
