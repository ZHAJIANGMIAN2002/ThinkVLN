#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-results/watcher_rollout_train_full_v2}"
MANIFEST_FILE="${MANIFEST_FILE:-${BUNDLE_ROOT}/manifest/watcher_rollout_manifest.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${BUNDLE_ROOT}/watcher_manual_switch_annotations.jsonl}"
DB_FILE="${DB_FILE:-${BUNDLE_ROOT}/watcher_manual_switch_web.sqlite3}"
SUMMARY_FULL_PATH="${SUMMARY_FULL_PATH:-data/trajectory_data/R2R_back/summary_full.jsonl}"

BIND_HOST="${BIND_HOST:-0.0.0.0}"
PORT="${PORT:-8010}"
PAGE_SIZE="${PAGE_SIZE:-20}"
IMAGE_STRIDE="${IMAGE_STRIDE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TITLE="${TITLE:-Watcher Manual Switch Annotation}"
SHUFFLE="${SHUFFLE:-0}"
SEED="${SEED:-42}"
CLAIM_LEASE_SECONDS="${CLAIM_LEASE_SECONDS:-1800}"
BOOTSTRAP_ADMIN_USER="${BOOTSTRAP_ADMIN_USER:-admin}"
BOOTSTRAP_ADMIN_PASSWORD="${BOOTSTRAP_ADMIN_PASSWORD:-${MANUAL_SWITCH_ADMIN_PASSWORD:-}}"

cmd=(
  python -m thinkvln.datagen.generation.watcher_manual_switch_annotation
  --bundle_root "${BUNDLE_ROOT}"
  --manifest_file "${MANIFEST_FILE}"
  --output_file "${OUTPUT_FILE}"
  --database_file "${DB_FILE}"
  --summary_full_path "${SUMMARY_FULL_PATH}"
  --host "${BIND_HOST}"
  --port "${PORT}"
  --page_size "${PAGE_SIZE}"
  --image_stride "${IMAGE_STRIDE}"
  --claim_lease_seconds "${CLAIM_LEASE_SECONDS}"
  --bootstrap_admin_user "${BOOTSTRAP_ADMIN_USER}"
  --title "${TITLE}"
)

if [[ -n "${BOOTSTRAP_ADMIN_PASSWORD}" ]]; then
  cmd+=(--bootstrap_admin_password "${BOOTSTRAP_ADMIN_PASSWORD}")
fi

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
echo "[run] db_file=${DB_FILE}"
echo "[run] url=http://${BIND_HOST}:${PORT}"

"${cmd[@]}"
