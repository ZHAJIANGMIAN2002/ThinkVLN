#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAMPLE_COUNT="${SAMPLE_COUNT:-30}"
IMAGE_STRIDE="${IMAGE_STRIDE:4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
MAX_RETRIES="${MAX_RETRIES:-4}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
RANDOM_SEED="${RANDOM_SEED:-}"

MANIFEST_SRC="${MANIFEST_SRC:-results/watcher_rollout_train_full_v2/manifest/watcher_rollout_manifest.jsonl}"
MANIFEST_RANDOM="${MANIFEST_RANDOM:-results/watcher_rollout_train_full_v2/manifest/watcher_rollout_manifest.random${SAMPLE_COUNT}.jsonl}"
OUT_DIR="${OUT_DIR:-results/watcher_rollout_train_full_v2}"
DEBUG_DIR_BASE="${DEBUG_DIR_BASE:-results/watcher_openai_debug_html_random${SAMPLE_COUNT}}"

python - "$MANIFEST_SRC" "$MANIFEST_RANDOM" "$SAMPLE_COUNT" "$RANDOM_SEED" <<'PY'
import random
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
sample_count = max(0, int(sys.argv[3]))
seed_text = sys.argv[4].strip()

if not src.exists():
    raise SystemExit(f"manifest not found: {src}")

lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
if not lines:
    raise SystemExit(f"manifest is empty: {src}")

if seed_text:
    random.seed(int(seed_text))

k = min(sample_count, len(lines))
selected = random.sample(lines, k)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(selected) + "\n", encoding="utf-8")
print(f"[random-manifest] src={src} total={len(lines)} selected={k} dst={dst}")
PY

TS="$(date +%Y%m%d_%H%M%S)"
OUT_FILE="${OUT_DIR}/watcher_openai_annotations_random${SAMPLE_COUNT}_${TS}.jsonl"
DEBUG_DIR="${DEBUG_DIR_BASE}_${TS}"

echo "[run] output_file=${OUT_FILE}"
echo "[run] debug_html=${DEBUG_DIR}/index.html"

python -m thinkvln.datagen.generation.watcher_openai_annotation \
  --gt_image_root data/trajectory_data/R2R_back/r2r \
  --bundle_root results/watcher_rollout_train_full_v2 \
  --manifest_file "${MANIFEST_RANDOM}" \
  --output_file "${OUT_FILE}" \
  --summary_full_path data/trajectory_data/R2R_back/summary_full.jsonl \
  --image_stride "${IMAGE_STRIDE}" \
  --max_samples "${SAMPLE_COUNT}" \
  --request_timeout "${REQUEST_TIMEOUT}" \
  --max_retries "${MAX_RETRIES}" \
  --reasoning_effort "${REASONING_EFFORT}" \
  --skip_missing_gt \
  --debug_html_dir "${DEBUG_DIR}"
