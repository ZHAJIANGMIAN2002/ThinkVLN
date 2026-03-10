#!/bin/bash

# Close-loop subtask evaluation for ThinkVLNActor LoRA checkpoint.


export CUDA_VISIBLE_DEVICES=7

STARTUP_SCAN_TURNS=0
RECOVERY_TURN_STEPS=0

torchrun --nproc_per_node=1 thinkvln/eval/close_eval.py \
  --model_type thinkvln_actor \
  --model_path outputs/actor/lora/run-3-3 \
  --ladder_mode subtask \
  --memory_num_history_images 6 \
  --done_threshold 0.85 \
  --summary_full_path data/trajectory_data/R2R_back/summary_full.jsonl \
  --habitat_config_path config/vln_r2r.yaml \
  --eval_split train \
  --sample_rate 0.1 \
  --startup_scan_turns "${STARTUP_SCAN_TURNS}" \
  --recovery_turn_steps "${RECOVERY_TURN_STEPS}" \
  --output_path results/env_eval/run-3-9-no-stuck
