#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/run_subtask_summary_pipeline.sh \
#   [trajectory_dir] [split_output_file] [determination_output_file] [summary_full_file]

ROOT_DIR="/mnt/swx/ThinkVLN"
TRAJ_DIR="${1:-$ROOT_DIR/data/trajectory_data/R2R_back}"
SPLIT_OUT="${2:-$ROOT_DIR/data/subtask_splits/R2R/subtask_splits_test.jsonl}"
DETERM_OUT="${3:-$ROOT_DIR/data/subtask_determination_results/R2R/subtask_determination_test.jsonl}"
SUMMARY_FULL="${4:-$ROOT_DIR/data/trajectory_data/R2R_back/summary_full_test.jsonl}"
MAX_WORKERS_SPLIT="${MAX_WORKERS_SPLIT:-8}"
MAX_WORKERS_DET="${MAX_WORKERS_DET:-4}"

cd "$ROOT_DIR"

if [ -f "/home/swx/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source /home/swx/miniforge3/etc/profile.d/conda.sh
  if [ "${CONDA_DEFAULT_ENV:-}" != "vln" ]; then
    # Some conda deactivate hooks reference optional vars; avoid nounset failure.
    set +u
    conda activate vln || true
    set -u
  fi
fi

mkdir -p "$(dirname "$SPLIT_OUT")" "$(dirname "$DETERM_OUT")" "$(dirname "$SUMMARY_FULL")"

echo "[1/3] Running subtask split..."
python -m thinkvln.datagen.generation.subtask_split \
  --trajectory_dir "$TRAJ_DIR" \
  --output_file "$SPLIT_OUT" \
  --max_workers "$MAX_WORKERS_SPLIT"

echo "[2/3] Running subtask determination..."
python -m thinkvln.datagen.generation.subtask_determination \
  --trajectory_dir "$TRAJ_DIR" \
  --subtask_splits_file "$SPLIT_OUT" \
  --output_file "$DETERM_OUT" \
  --max_workers "$MAX_WORKERS_DET"

echo "[3/3] Merging summary_full..."
python - "$TRAJ_DIR" "$SPLIT_OUT" "$DETERM_OUT" "$SUMMARY_FULL" <<'PY'
import json
import os
import sys

trajectory_dir, split_file, determination_file, summary_full_file = sys.argv[1:]


def get_episode_key(scene_id, episode_id):
    return f"{scene_id}_{episode_id}" if scene_id else str(episode_id)


def load_summary(path):
    episodes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def load_jsonl_by_key(path):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("episode_key")
            if not key and "scene_id" in record and "episode_id" in record:
                key = get_episode_key(record.get("scene_id"), record.get("episode_id"))
            if key:
                data[key] = record
    return data


summary_path = os.path.join(trajectory_dir, "summary.json")
if not os.path.exists(summary_path):
    raise FileNotFoundError(f"summary.json not found: {summary_path}")

episodes = load_summary(summary_path)
split_map = load_jsonl_by_key(split_file)
det_map = load_jsonl_by_key(determination_file)

merged = 0
skipped = 0
with open(summary_full_file, "w", encoding="utf-8") as out_f:
    for ep in episodes:
        episode_id = ep.get("id")
        scene_id = ep.get("scene_id")
        episode_key = get_episode_key(scene_id, episode_id)
        split_rec = split_map.get(episode_key)
        det_rec = det_map.get(episode_key)
        if not split_rec or not det_rec:
            skipped += 1
            continue

        instruction = split_rec.get("instruction")
        if not instruction:
            instructions = ep.get("instructions")
            instruction = instructions[0] if isinstance(instructions, list) and instructions else instructions

        rec = {
            "episode_key": episode_key,
            "episode_id": episode_id,
            "id": episode_id,
            "scene_id": scene_id,
            "trajectory_id": ep.get("trajectory_id"),
            "instruction": instruction,
            "plan": split_rec.get("plan", []),
            "actions": ep.get("actions", []),
            "video": ep.get("video", ""),
            "num_subtasks": det_rec.get("num_subtasks"),
            "num_frames": det_rec.get("num_frames"),
            "keyframes": det_rec.get("keyframes"),
            "subtask_sequence": det_rec.get("subtask_sequence"),
            "subtask_counts": det_rec.get("subtask_counts"),
            "transition_points": det_rec.get("transition_points"),
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        merged += 1

print(f"summary_full merged: {merged}")
print(f"summary_full skipped (missing split/determination): {skipped}")
print(f"summary_full path: {summary_full_file}")
PY

echo "Done."

