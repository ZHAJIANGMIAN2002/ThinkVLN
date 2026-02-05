#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH=${1:-"config/ar_training.yaml"}
NUM_GPUS=${2:-$(nvidia-smi --list-gpus | wc -l)}
MASTER_PORT=${3:-29500}

echo "Training with ${NUM_GPUS} GPU(s)"

if [ "${NUM_GPUS}" -gt 1 ]; then
  echo "Multi-GPU training with torchrun"
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m thinkvln.engine.ar_trainer \
    --config "${CONFIG_PATH}"
else
  echo "Single-GPU training"
  python -m thinkvln.engine.ar_trainer \
    --config "${CONFIG_PATH}"
fi

