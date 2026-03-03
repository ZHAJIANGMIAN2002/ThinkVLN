#!/bin/bash
set -e

# ThinkVLN Actor Evaluation Script
# 
# Usage:
#   bash scripts/eval_thinkvln_actor.sh [MODEL_PATH] [CONFIG_FILE] [GPUS] [EXTRA_ARGS...]
#
# Examples:
#   bash scripts/eval_thinkvln_actor.sh
#   bash scripts/eval_thinkvln_actor.sh outputs/actor/lora/run-2-3-11-07
#   bash scripts/eval_thinkvln_actor.sh outputs/actor/lora/run-2-3-11-07 config/eval_config.yaml
#   bash scripts/eval_thinkvln_actor.sh outputs/actor/lora/run-2-3-11-07 config/eval_config.yaml 0,1,2,3
#   bash scripts/eval_thinkvln_actor.sh outputs/actor/lora/run-2-3-11-07 config/eval_config.yaml 0,1 --batch_size 64

# Default configuration
DEFAULT_CONFIG="config/eval_config.yaml"
DEFAULT_MODEL_PATH="outputs/actor/lora/run-3-2-seperate"
DEFAULT_GPUS="4"

# Parse arguments
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"
CONFIG_FILE="${2:-$DEFAULT_CONFIG}"
GPUS="${3:-$DEFAULT_GPUS}"
EXTRA_ARGS=("${@:4}")

# GPU configuration
export CUDA_VISIBLE_DEVICES="$GPUS"
NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')
MASTER_PORT="${MASTER_PORT:-29611}"

echo "=========================================="
echo "ThinkVLN Actor Model Evaluation"
echo "=========================================="
echo "Model path:   $MODEL_PATH"
echo "Config file:  $CONFIG_FILE"
echo "Using GPU(s): $GPUS (count=$NUM_GPUS)"
echo "=========================================="
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file does not exist: $CONFIG_FILE"
    exit 1
fi

# Run evaluation
echo "Starting evaluation..."
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        thinkvln/engine/evaluate.py \
        --config "$CONFIG_FILE" \
        --model_path "$MODEL_PATH" \
        "${EXTRA_ARGS[@]}"
else
    python thinkvln/engine/evaluate.py \
        --config "$CONFIG_FILE" \
        --model_path "$MODEL_PATH" \
        "${EXTRA_ARGS[@]}"
fi

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
