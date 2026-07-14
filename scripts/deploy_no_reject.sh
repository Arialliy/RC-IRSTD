#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORE_DIR=${1:?Usage: $0 SCORE_DIR CALIBRATOR_CHECKPOINT OUTPUT_JSON [SOURCE_REFERENCE]}
CHECKPOINT=${2:?Usage: $0 SCORE_DIR CALIBRATOR_CHECKPOINT OUTPUT_JSON [SOURCE_REFERENCE]}
OUTPUT=${3:?Usage: $0 SCORE_DIR CALIBRATOR_CHECKPOINT OUTPUT_JSON [SOURCE_REFERENCE]}
SOURCE_REFERENCE=${4:-}
ARGS=(
  python -m rc_irstd.pipelines.apply_calibrator
  --score-directory "$SCORE_DIR"
  --checkpoint "$CHECKPOINT"
  --context-size "${CONTEXT_SIZE:-32}"
  --budget "${PIXEL_BUDGET:-1e-5}"
  --device "${DEVICE:-cuda}"
  --output "$OUTPUT"
)
if [[ -n "$SOURCE_REFERENCE" ]]; then
  ARGS+=(--source-reference "$SOURCE_REFERENCE")
fi
cd "$ROOT"
exec "${ARGS[@]}"
