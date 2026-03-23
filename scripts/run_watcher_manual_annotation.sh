#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MAX_SAMPLES="${MAX_SAMPLES:-10}"
IMAGE_STRIDE="${IMAGE_STRIDE:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
MAX_RETRIES="${MAX_RETRIES:-4}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
SHUFFLE="${SHUFFLE:-1}"
SEED="${SEED:-42}"
RESUME="${RESUME:-1}"

OUT_FILE="${OUT_FILE:-results/watcher_rollout_train_full_v2/watcher_manual_done_annotations.jsonl}"
WORK_DIR="${WORK_DIR:-results/watcher_manual_review}"

cmd=(
  python -m thinkvln.datagen.generation.watcher_manual_done_annotation
  --gt_image_root "data/trajectory_data/R2R_back/r2r"
  --bundle_root "results/watcher_rollout_train_full_v2"
  --manifest_file "results/watcher_rollout_train_full_v2/manifest/watcher_rollout_manifest.jsonl"
  --output_file "${OUT_FILE}"
  --summary_full_path "data/trajectory_data/R2R_back/summary_full.jsonl"
  --work_dir "${WORK_DIR}"
  --image_stride "${IMAGE_STRIDE}"
  --max_samples "${MAX_SAMPLES}"
  --request_timeout "${REQUEST_TIMEOUT}"
  --max_retries "${MAX_RETRIES}"
  --reasoning_effort "${REASONING_EFFORT}"
  --skip_missing_gt
)

if [[ "${SHUFFLE}" == "1" ]]; then
  cmd+=(--shuffle --seed "${SEED}")
fi
if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
fi

echo "[run] manual done annotation"
echo "[run] output_file=${OUT_FILE}"
echo "[run] work_dir=${WORK_DIR}"
echo "[run] max_samples=${MAX_SAMPLES}"

"${cmd[@]}"
