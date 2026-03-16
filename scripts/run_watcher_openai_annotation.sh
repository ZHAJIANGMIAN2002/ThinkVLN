#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
MAX_RETRIES="${MAX_RETRIES:-4}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"

cmd=(
  python -m thinkvln.datagen.generation.watcher_openai_annotation
  --gt_image_root data/trajectory_data/R2R_back/r2r
  --bundle_root results/watcher_rollout_train_full_v2
  --manifest_file results/watcher_rollout_train_full_v2/manifest/watcher_rollout_manifest.jsonl
  --output_file results/watcher_rollout_train_full_v2/watcher_openai_annotations.jsonl
  --summary_full_path data/trajectory_data/R2R_back/summary_full.jsonl
  --image_stride 3
  --max_samples 10
  --request_timeout "${REQUEST_TIMEOUT}"
  --max_retries "${MAX_RETRIES}"
  --reasoning_effort "${REASONING_EFFORT}"
  --skip_missing_gt
  --debug_html_dir results/watcher_openai_debug_html
)

"${cmd[@]}"
