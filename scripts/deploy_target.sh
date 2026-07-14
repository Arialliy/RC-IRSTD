#!/usr/bin/env bash
set -euo pipefail

# Export unlabeled target scores, estimate an operating point from a past-only
# warm-up window, and apply it to future images.
#
# Usage:
#   ./scripts/deploy_target.sh DATASET SPLIT DETECTOR_PT CURVE_PT OUTPUT_DIR

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 DATASET_DIR SPLIT DETECTOR_CHECKPOINT CURVE_CHECKPOINT OUTPUT_DIR" >&2
  exit 2
fi
DATASET_DIR="$1"
SPLIT="$2"
DETECTOR_CHECKPOINT="$3"
CURVE_CHECKPOINT="$4"
OUTPUT_DIR="$5"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCORE_DIR="$OUTPUT_DIR/scores"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR"

python -m rc_irstd.pipelines.export_scores \
  --dataset-dir "$DATASET_DIR" \
  --split "$SPLIT" \
  --detector mshnet \
  --checkpoint "$DETECTOR_CHECKPOINT" \
  --inference-mode "${INFERENCE_MODE:-native_pad}" \
  --stride-multiple "${STRIDE_MULTIPLE:-32}" \
  --normalization "${NORMALIZATION:-imagenet}" \
  --dataset-type "${DATASET_TYPE:-iid_images}" \
  --no-include-mask \
  --device "${DEVICE:-cuda}" \
  --output-dir "$SCORE_DIR"

python -m rc_irstd.pipelines.run_deployment \
  --score-dir "$SCORE_DIR" \
  --curve-checkpoint "$CURVE_CHECKPOINT" \
  --warmup-size "${WARMUP_SIZE:-32}" \
  --update-every "${UPDATE_EVERY:-0}" \
  --pixel-budget "${PIXEL_BUDGET:-1e-6}" \
  --peak-budget "${PEAK_BUDGET:-1.0}" \
  --offset-index "${OFFSET_INDEX:-0}" \
  --ood-threshold "${OOD_THRESHOLD:-8.0}" \
  --device "${DEVICE:-cuda}" \
  --output-dir "$OUTPUT_DIR/deployment"
