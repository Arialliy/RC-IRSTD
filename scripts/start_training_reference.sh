#!/usr/bin/env bash
set -euo pipefail

# Explicit compatibility dispatcher for rc_irstd/.  These commands use the
# reference NPZ/checkpoint contracts and are not interchangeable with flat v5.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-}"
shift || true
default_project_python="$(dirname "$ROOT")/BasicIRSTD/infrarenet/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$default_project_python" ]]; then
    PYTHON_BIN="$default_project_python"
  else
    PYTHON_BIN="python"
  fi
fi
if [[ "$PYTHON_BIN" != */* ]]; then
  PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN is not executable" >&2
  exit 2
fi

case "$MODE" in
  detector)
    exec "$ROOT/scripts/train_detector_mshnet.sh" "$@"
    ;;
  calibrator)
    exec "$ROOT/scripts/train_calibrator_reference.sh" "$@"
    ;;
  build-meta)
    exec "$ROOT/scripts/build_meta_fold.sh" "$@"
    ;;
  deploy)
    exec "$ROOT/scripts/deploy_no_reject.sh" "$@"
    ;;
  validate)
    exec "$ROOT/scripts/validate_two_stage_release.sh" "$@"
    ;;
  legacy-lodo)
    CONFIG="${1:-$ROOT/configs/lodo_example.yaml}"
    shift || true
    exec "$PYTHON_BIN" \
      -m rc_irstd.pipelines.run_lodo --config "$CONFIG" "$@"
    ;;
  smoke)
    exec "$ROOT/scripts/smoke_two_stage_no_reject.sh" "$@"
    ;;
  *)
    cat >&2 <<EOF
Reference compatibility usage:
  $0 detector /data/sourceA /data/sourceB [...] [-- extra detector options]
  $0 build-meta SCORE_DIR OUTPUT_META [SOURCE_REFERENCE]
  $0 calibrator TRAIN_META VAL_META OUTPUT_DIR [extra options]
  $0 deploy SCORE_DIR CALIBRATOR_CHECKPOINT OUTPUT_JSON [SOURCE_REFERENCE]
  $0 validate [work-directory]
  $0 smoke [work-directory]
  $0 legacy-lodo configs/lodo_example.yaml [--outer-target TARGET] [--dry-run]
EOF
    exit 2
    ;;
esac
