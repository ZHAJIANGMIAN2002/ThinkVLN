#!/bin/bash
set -e

# ThinkVLN Actor Evaluation Script
# 
# Usage:
#   bash scripts/eval_thinkvln_actor.sh [MODEL_PATH] [CONFIG_FILE]
#
# Examples:
#   bash scripts/eval_thinkvln_actor.sh
#   bash scripts/eval_thinkvln_actor.sh outputs/actor/lora/run-2-3-11-07
#   bash scripts/eval_thinkvln_actor.sh outputs/actor/lora/run-2-3-11-07 config/eval_config.yaml

# Default configuration
DEFAULT_CONFIG="config/eval_config.yaml"
DEFAULT_MODEL_PATH="outputs/actor/lora/run-2-3-11-07"

# Parse arguments
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"
CONFIG_FILE="${2:-$DEFAULT_CONFIG}"

# GPU configuration
export GPUS="4"  # Single GPU for evaluation
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "=========================================="
echo "ThinkVLN Actor Model Evaluation"
echo "=========================================="
echo "Model path:   $MODEL_PATH"
echo "Config file:  $CONFIG_FILE"
echo "Using GPU:    $GPUS"
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
python thinkvln/engine/evaluate.py \
    --config "$CONFIG_FILE" \
    --model_path "$MODEL_PATH"

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
