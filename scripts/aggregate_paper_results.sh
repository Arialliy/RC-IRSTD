#!/usr/bin/env bash
set -euo pipefail
LODO_ROOT="${1:?Usage: $0 /path/to/lodo/output [output_dir]}"
OUTPUT_DIR="${2:-$LODO_ROOT/paper_tables}"
python -m rc_irstd.pipelines.aggregate_results \
  --lodo-root "$LODO_ROOT" \
  --output-dir "$OUTPUT_DIR"
