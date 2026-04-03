#!/bin/bash
set -euo pipefail

GPUS="${GPUS:-1}"
CONFIG_FILE="${CONFIG_FILE:-config/watcher_sft.yaml}"
MASTER_PORT="${MASTER_PORT:-29621}"
EXTRA_ARGS=("${@:1}")

export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')

echo "Watcher SFT training"
echo "GPUs: $GPUS (count=$NUM_GPUS)"
echo "Config: $CONFIG_FILE"

run_cmd() {
  if [ "${CONDA_DEFAULT_ENV:-}" = "vln" ]; then
    "$@"
  else
    conda run --no-capture-output -n vln "$@"
  fi
}

if [ "$NUM_GPUS" -gt 1 ]; then
  run_cmd torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    thinkvln/engine/watcher_sft_trainer.py \
    --config "$CONFIG_FILE" \
    "${EXTRA_ARGS[@]}"
else
  run_cmd python \
    thinkvln/engine/watcher_sft_trainer.py \
    --config "$CONFIG_FILE" \
    "${EXTRA_ARGS[@]}"
fi
