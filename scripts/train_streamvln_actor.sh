#!/usr/bin/env bash

set -euo pipefail

CONFIG_FILE=${1:-"config/streamvln_actor_train.yaml"}

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file does not exist: $CONFIG_FILE"
  exit 1
fi

mapfile -t CFG_VALUES < <(python - "$CONFIG_FILE" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
runtime = config.get("runtime") or {}
print(runtime.get("gpus", "0"))
print(runtime.get("master_port", 29512))
PY
)

GPUS="${CFG_VALUES[0]}"
MASTER_PORT="${MASTER_PORT:-${CFG_VALUES[1]}}"

export CUDA_VISIBLE_DEVICES="$GPUS"
NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')

echo "=========================================="
echo "StreamVLN Actor Training"
echo "=========================================="
echo "Config file:  $CONFIG_FILE"
echo "Using GPU(s): $GPUS (count=$NUM_GPUS)"
echo "Master port:  $MASTER_PORT"
echo "=========================================="

if [ "$NUM_GPUS" -gt 1 ]; then
  torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    scripts/train_streamvln_actor.py \
    --config "$CONFIG_FILE"
else
  python scripts/train_streamvln_actor.py --config "$CONFIG_FILE"
fi
