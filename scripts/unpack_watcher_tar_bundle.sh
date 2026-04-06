#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: cd /mnt/swx/ThinkVLN && bash scripts/unpack_watcher_tar_bundle.sh <tar_bundle_root> <bundle_root>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
python3 thinkvln/datagen/generation/extract_tar_shard_folder.py \
  --input_root "$1" \
  --output_root "$2"

echo "[next] bundle_root=$2"
echo "[next] manifest_file=$2/manifest/watcher_rollout_manifest.jsonl"
