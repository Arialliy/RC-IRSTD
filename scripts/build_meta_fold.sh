#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORE_DIR=${1:?Usage: $0 SCORE_DIR OUTPUT_META [SOURCE_REFERENCE]}
OUTPUT_META=${2:?Usage: $0 SCORE_DIR OUTPUT_META [SOURCE_REFERENCE]}
SOURCE_REFERENCE=${3:-}
ARGS=(
  python -m rc_irstd.pipelines.build_meta_dataset
  --score-directory "$SCORE_DIR"
  --output "$OUTPUT_META"
  --budget "${BUDGET_LOOSE:-1e-4}"
  --budget "${BUDGET_MID:-1e-5}"
  --budget "${BUDGET_STRICT:-1e-6}"
  --context-size "${CONTEXT_SIZE:-32}"
  --horizon "${QUERY_SIZE:-64}"
  --protocol "${PROTOCOL:-auto}"
  --background-sample-limit "${BACKGROUND_SAMPLE_LIMIT:-65536}"
  --split-role "${SPLIT_ROLE:-official_train_meta}"
  --seed "${SEED:-42}"
)
if [[ -n "$SOURCE_REFERENCE" ]]; then
  ARGS+=(--source-reference "$SOURCE_REFERENCE")
fi
cd "$ROOT"
exec "${ARGS[@]}"
