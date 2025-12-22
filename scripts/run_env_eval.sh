#!/bin/bash
# ThinkVLN Environment Evaluation Script

# Set default values
MODEL_PATH="${1:-outputs/lora/train_2025-12-18-12-19-49}"
HABITAT_CONFIG="${2:-config/vln_r2r.yaml}"
EVAL_SPLIT="${3:-val_unseen}"
OUTPUT_PATH="${4:-./results/env_eval}"

echo "Starting ThinkVLN Environment Evaluation"
echo "Model: $MODEL_PATH"
echo "Config: $HABITAT_CONFIG"
echo "Split: $EVAL_SPLIT"
echo "Output: $OUTPUT_PATH"
echo ""

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    thinkvln/model/env_eval.py \
    --model_path "$MODEL_PATH" \
    --habitat_config_path "$HABITAT_CONFIG" \
    --eval_split "$EVAL_SPLIT" \
    --output_path "$OUTPUT_PATH" \
    --device cuda

