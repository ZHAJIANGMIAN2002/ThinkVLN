#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <episode_key>"
  echo "Example: $0 17DRP5sb8fy_517"
  exit 1
fi

EPISODE_KEY="$1"

export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 thinkvln/eval/close_eval.py \
  --model_type thinkvln_actor \
  --model_path outputs/actor/lora/run-3-3 \
  --ladder_mode subtask \
  --memory_num_history_images 6 \
  --done_threshold 0.85 \
  --summary_full_path data/trajectory_data/R2R_back/summary_full.jsonl \
  --habitat_config_path config/vln_r2r.yaml \
  --eval_split train \
  --sample_rate 1.0 \
  --target_episode_key "${EPISODE_KEY}" \
  --enable_step_debug \
  --step_debug_format html \
  --output_path results/env_eval/run-3-3

echo "Debug HTML generated at: results/env_eval/run-3-3/debug_${EPISODE_KEY}/index.html"
