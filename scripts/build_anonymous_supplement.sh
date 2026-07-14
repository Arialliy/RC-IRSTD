#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:-$ROOT/dist/RC_IRSTD_Anonymous_Supplement.zip}"
mkdir -p "$(dirname "$OUTPUT")"
python -m rc_irstd.pipelines.build_supplement \
  --source-root "$ROOT" \
  --output "$OUTPUT"
