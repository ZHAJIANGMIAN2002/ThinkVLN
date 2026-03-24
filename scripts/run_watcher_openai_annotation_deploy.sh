#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAMPLE_COUNT="${SAMPLE_COUNT:-50}"
MAX_WORKERS="${MAX_WORKERS:-8}"
IMAGE_STRIDE="${IMAGE_STRIDE:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
MAX_RETRIES="${MAX_RETRIES:-4}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
RANDOM_SEED="${RANDOM_SEED:-}"

MANIFEST_FILE="${MANIFEST_FILE:-results/watcher_rollout_train_full_v2/manifest/watcher_rollout_manifest.jsonl}"
OUT_DIR="${OUT_DIR:-results/watcher_rollout_train_full_v2}"
DEBUG_DIR_BASE="${DEBUG_DIR_BASE:-results/watcher_openai_debug_html_deploy}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_DIR}/watcher_openai_annotations_deploy_${TS}.jsonl"
DEBUG_DIR="${DEBUG_DIR_BASE}_${TS}"

cmd=(
  python "thinkvln/datagen/generation/watcher_openai_annotation_deploy.py"
  --gt_video_root "/mnt/swx/ThinkVLN/data/trajectory_data/R2R_back/images"
  --bundle_root "results/watcher_rollout_train_full_v2"
  --manifest_file "${MANIFEST_FILE}"
  --output_file "${OUT_FILE}"
  --summary_full_path "data/trajectory_data/R2R_back/summary_full.jsonl"
  --image_stride "${IMAGE_STRIDE}"
  --max_samples "${SAMPLE_COUNT}"
  --max_workers "${MAX_WORKERS}"
  --request_timeout "${REQUEST_TIMEOUT}"
  --max_retries "${MAX_RETRIES}"
  --reasoning_effort "${REASONING_EFFORT}"
  --skip_missing_gt
  --deploy_mode
  --shuffle
  --debug_html_dir "${DEBUG_DIR}"
)

if [[ -n "${RANDOM_SEED}" ]]; then
  cmd+=(--seed "${RANDOM_SEED}")
fi

echo "[run] output_file=${OUT_FILE}"
echo "[run] debug_html=${DEBUG_DIR}/index.html"
echo "[run] max_workers=${MAX_WORKERS} sample_count=${SAMPLE_COUNT}"

"${cmd[@]}"
