#!/usr/bin/env bash
set -euo pipefail

# Train the final domain-tail-separation MSHNet on balanced source-domain batches.
#
# Usage:
#   ./scripts/train_detector_mshnet.sh /data/NUAA-SIRST /data/NUDT-SIRST /data/IRSTD-1K
#   SOURCE_DATASETS=/data/A:/data/B ./scripts/train_detector_mshnet.sh
#
# Extra Python arguments follow "--":
#   ./scripts/train_detector_mshnet.sh /data/A /data/B -- --epochs 40 --lr 0.02

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/outputs/detector_mshnet}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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

DATASETS=()
EXTRA_ARGS=()
SEEN_SEPARATOR=0
for value in "$@"; do
  if [[ "$value" == "--" && "$SEEN_SEPARATOR" -eq 0 ]]; then
    SEEN_SEPARATOR=1
    continue
  fi
  if [[ "$SEEN_SEPARATOR" -eq 0 ]]; then
    DATASETS+=("$value")
  else
    EXTRA_ARGS+=("$value")
  fi
done

if [[ ${#DATASETS[@]} -eq 0 && -n "${SOURCE_DATASETS:-}" ]]; then
  IFS=':' read -r -a DATASETS <<< "$SOURCE_DATASETS"
fi
if [[ ${#DATASETS[@]} -lt 1 ]]; then
  echo "Usage: $0 /data/sourceA [/data/sourceB ...] [-- extra-options]" >&2
  exit 2
fi
for dataset in "${DATASETS[@]}"; do
  if [[ ! -d "$dataset" ]]; then
    echo "ERROR: dataset directory does not exist: $dataset" >&2
    exit 2
  fi
done

PER_DOMAIN_BATCH="${PER_DOMAIN_BATCH:-2}"
BATCH_SIZE="$((PER_DOMAIN_BATCH * ${#DATASETS[@]}))"
COMMAND=(
  "$PYTHON_BIN" -m rc_irstd.pipelines.train_detector
  --train-split "${TRAIN_SPLIT:-train}"
  --detector mshnet
  --base-loss auto
  --resize "${RESIZE_H:-256}" "${RESIZE_W:-256}"
  --normalization "${NORMALIZATION:-imagenet}"
  --dataset-type "${DATASET_TYPE:-iid_images}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS:-400}"
  --warm-epoch "${WARM_EPOCH:-5}"
  --optimizer "${OPTIMIZER:-adagrad}"
  --lr "${LR:-0.05}"
  --weight-decay "${WEIGHT_DECAY:-0.0}"
  --detector-objective domain_tail_separation
  --lambda-sep "${LAMBDA_SEP:-0.20}"
  --separation-margin "${SEPARATION_MARGIN:-1.0}"
  --background-tail-fraction "${BACKGROUND_TAIL_FRACTION:-0.05}"
  --object-top-fraction "${OBJECT_TOP_FRACTION:-0.25}"
  --hard-object-fraction "${HARD_OBJECT_FRACTION:-0.25}"
  --risk-start-epoch "${RISK_START_EPOCH:-5}"
  --risk-ramp-epochs "${RISK_RAMP_EPOCHS:-10}"
  --peak-kernel "${PEAK_KERNEL:-5}"
  --exclusion-radius "${EXCLUSION_RADIUS:-2}"
  --worst-gamma "${WORST_GAMMA:-10.0}"
  --auxiliary-weight "${AUXILIARY_WEIGHT:-1.0}"
  --pixel-budget "${SELECTION_PIXEL_BUDGET:-1e-5}"
  --peak-budget "${SELECTION_PEAK_BUDGET:-5.0}"
  --num-workers "${NUM_WORKERS:-4}"
  --device "${DEVICE:-cuda}"
  --amp
  --deterministic
  --seed "${SEED:-42}"
  --output-dir "$RUN_ROOT"
)
if [[ "${ENGINEERING_SMOKE_NO_VALIDATION:-0}" == "1" ]]; then
  COMMAND+=(--engineering-smoke-no-validation)
  if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
    COMMAND+=(--max-train-steps "$MAX_TRAIN_STEPS")
  fi
else
  COMMAND+=(--val-split "${VAL_SPLIT:-val}")
fi
for dataset in "${DATASETS[@]}"; do
  COMMAND+=(--source-dataset "$dataset")
done
COMMAND+=("${EXTRA_ARGS[@]}")

cd "$ROOT"
printf 'Launching:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
exec "${COMMAND[@]}"
