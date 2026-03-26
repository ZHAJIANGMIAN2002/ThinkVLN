#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
BUNDLE_ROOT="results/watcher_rollout_train_streamvln_rgb_v2"
MANIFEST_DIR="${BUNDLE_ROOT}/manifest"

common_args=(
  --summary_full_path data/trajectory_data/R2R_back/summary_full.jsonl
  --habitat_config_path config/vln_r2r.yaml
  --model_path model_weights/streamvln
  --bundle_root "${BUNDLE_ROOT}"
  --num_pivots 4
  --num_rollouts 1
  --min_rollout_steps 6
  --max_rollout_steps 12
  --episode_sample_rate 1.0
  --seed 42
  --resume
)

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun \
    --standalone \
    --nproc_per_node "${NUM_GPUS}" \
    -m thinkvln.datagen.generation.watcher_rollout_generation \
    "${common_args[@]}"

  python - "${MANIFEST_DIR}" <<'PY'
import glob
import json
import sys
from pathlib import Path

manifest_dir = Path(sys.argv[1])
manifest_dir.mkdir(parents=True, exist_ok=True)
rank_files = sorted(glob.glob(str(manifest_dir / "watcher_rollout_manifest.rank*.jsonl")))
merged_file = manifest_dir / "watcher_rollout_manifest.jsonl"

seen_ids = set()
rows = []
for path in rank_files:
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                sample_id = str((json.loads(line) or {}).get("sample_id", ""))
            except Exception:
                sample_id = line
            dedup_key = sample_id or line
            if dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)
            rows.append(line)

with open(merged_file, "w", encoding="utf-8") as handle:
    for line in rows:
        handle.write(line + "\n")

print(f"[merge] files={len(rank_files)} rows={len(rows)} -> {merged_file}")
PY
else
  python -m thinkvln.datagen.generation.watcher_rollout_generation \
    "${common_args[@]}"
fi
