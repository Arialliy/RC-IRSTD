#!/usr/bin/env bash
set -euo pipefail

# Compatibility launcher for the self-contained TwoStage reference package.
# Its NPZ/checkpoint schema must never be mixed with the strict flat v5 path.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_META=${1:?Usage: $0 TRAIN_META VAL_META OUTPUT_DIR [extra args...]}
VAL_META=${2:?Usage: $0 TRAIN_META VAL_META OUTPUT_DIR [extra args...]}
OUTPUT_DIR=${3:?Usage: $0 TRAIN_META VAL_META OUTPUT_DIR [extra args...]}
shift 3

export CUDA_VISIBLE_DEVICES="${RC_CALIBRATOR_GPU:-0}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
default_project_python="$(dirname "$ROOT")/BasicIRSTD/infrarenet/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$default_project_python" ]]; then
    PYTHON_BIN="$default_project_python"
  else
    PYTHON_BIN="python"
  fi
fi
if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 2
  fi
else
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: PYTHON_BIN command was not found" >&2
    exit 2
  fi
fi

exec "$PYTHON_BIN" -m rc_irstd.pipelines.train_calibrator \
  --train-meta "$TRAIN_META" \
  --val-meta "$VAL_META" \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --lr "${LR:-0.001}" \
  --lambda-violation "${LAMBDA_VIOLATION:-4.0}" \
  --lambda-utility "${LAMBDA_UTILITY:-1.0}" \
  --lambda-oracle "${LAMBDA_ORACLE:-0.10}" \
  --lambda-smoothness "${LAMBDA_SMOOTHNESS:-0.01}" \
  --pixel-temperature "${PIXEL_TEMPERATURE:-0.10}" \
  --object-temperature "${OBJECT_TEMPERATURE:-0.20}" \
  --device "${DEVICE:-cuda}" \
  --seed "${SEED:-42}" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
