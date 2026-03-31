#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <input-video> <output-dir>" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_VIDEO="$1"
OUTPUT_DIR="$2"
CONFIG_PATH="$ROOT_DIR/config.out3_clean.yaml"

mkdir -p "$OUTPUT_DIR"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.venv/bin/activate"
fi

python -m bat_tracker.cli \
  --input "$INPUT_VIDEO" \
  --output "$OUTPUT_DIR" \
  --config "$CONFIG_PATH"
