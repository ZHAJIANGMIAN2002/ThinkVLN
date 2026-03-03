#!/bin/bash
set -euo pipefail

# Close-loop subtask evaluation for ThinkVLNActor LoRA checkpoint.


export CUDA_VISIBLE_DEVICES=0,1,2,3


torchrun --nproc_per_node=4 thinkvln/eval/close_eval.py \
  --model_type thinkvln_actor \
  --model_path outputs/actor/lora/run-3-2-seperate/checkpoint-200 \
  --ladder_mode subtask \
  --summary_full_path data/trajectory_data/R2R_back/summary_full.jsonl \
  --habitat_config_path config/vln_r2r.yaml \
  --eval_split train \
  --sample_rate 0.1 \
  --output_path results/env_eval/run-3-2-seperate_subtask
