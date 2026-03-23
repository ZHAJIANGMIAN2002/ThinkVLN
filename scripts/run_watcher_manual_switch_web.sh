#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-results/watcher_rollout_train_full_v2}"
MANIFEST_FILE="${MANIFEST_FILE:-${BUNDLE_ROOT}/manifest/watcher_rollout_manifest.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${BUNDLE_ROOT}/watcher_manual_switch_annotations.jsonl}"
SUMMARY_FULL_PATH="${SUMMARY_FULL_PATH:-data/trajectory_data/R2R_back/summary_full.jsonl}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
PAGE_SIZE="${PAGE_SIZE:-20}"
IMAGE_STRIDE="${IMAGE_STRIDE:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TITLE="${TITLE:-Watcher Manual Switch Annotation}"
SHUFFLE="${SHUFFLE:-0}"
SEED="${SEED:-42}"

cmd=(
  python -m thinkvln.datagen.generation.watcher_manual_switch_annotation
  --bundle_root "${BUNDLE_ROOT}"
  --manifest_file "${MANIFEST_FILE}"
  --output_file "${OUTPUT_FILE}"
  --summary_full_path "${SUMMARY_FULL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --page_size "${PAGE_SIZE}"
  --image_stride "${IMAGE_STRIDE}"
  --title "${TITLE}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ "${SHUFFLE}" == "1" ]]; then
  cmd+=(--shuffle --seed "${SEED}")
fi

echo "[run] manual switch web"
echo "[run] bundle_root=${BUNDLE_ROOT}"
echo "[run] manifest_file=${MANIFEST_FILE}"
echo "[run] output_file=${OUTPUT_FILE}"
echo "[run] url=http://${HOST}:${PORT}"

"${cmd[@]}"
