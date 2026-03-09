#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <html_file_or_dir> [port] [host]"
  echo "Example: $0 results/env_eval/run-3-3/debug_17DRP5sb8fy_517/index.html 8000 127.0.0.1"
  exit 1
fi

TARGET="$1"
PORT="${2:-8000}"
HOST="${3:-127.0.0.1}"

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Error: port must be a number, got '$PORT'"
  exit 1
fi

if [[ -f "$TARGET" ]]; then
  SERVE_DIR="$(cd "$(dirname "$TARGET")" && pwd)"
  PAGE_NAME="$(basename "$TARGET")"
elif [[ -d "$TARGET" ]]; then
  SERVE_DIR="$(cd "$TARGET" && pwd)"
  PAGE_NAME="index.html"
else
  echo "Error: path not found: $TARGET"
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: python interpreter not found"
  exit 1
fi

echo "Serving directory: $SERVE_DIR"
echo "Open: http://$HOST:$PORT/$PAGE_NAME"
echo "Press Ctrl+C to stop."

cd "$SERVE_DIR"
exec "$PYTHON_BIN" -m http.server "$PORT" --bind "$HOST"
