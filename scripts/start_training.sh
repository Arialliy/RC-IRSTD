#!/usr/bin/env bash
set -euo pipefail

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
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

case "$MODE" in
  detector)
    exec "$ROOT/scripts/train_rc_3gpu.sh" "$@"
    ;;
  calibrator)
    exec "$ROOT/scripts/train_calibrator_risk_aligned.sh" "$@"
    ;;
  export-scores)
    exec "$PYTHON_BIN" -m evaluation.export_score_maps "$@"
    ;;
  export-labels)
    exec "$PYTHON_BIN" -m evaluation.export_label_maps "$@"
    ;;
  build-source-reference)
    exec "$PYTHON_BIN" -m rc.build_source_reference "$@"
    ;;
  build-meta)
    exec "$PYTHON_BIN" -m rc.build_meta_episodes "$@"
    ;;
  online)
    exec "$PYTHON_BIN" -m rc.online_adapter "$@"
    ;;
  audit)
    exec "$PYTHON_BIN" -m scripts.audit_aaai_protocol "$@"
    ;;
  reference)
    exec "$ROOT/scripts/start_training_reference.sh" "$@"
    ;;
  *)
    cat >&2 <<EOF
Usage:
  $0 detector [strict scripts.train_multisource_tail options]
  $0 export-scores [strict evaluation.export_score_maps options]
  $0 export-labels [strict evaluation.export_label_maps options]
  $0 build-source-reference [strict rc.build_source_reference options]
  $0 build-meta [strict rc.build_meta_episodes options]
  $0 calibrator [strict rc.train_calibrator_risk_aligned options]
  $0 online [strict rc.online_adapter options]
  $0 audit [strict scripts.audit_aaai_protocol options]
  $0 reference <mode> [...]  # compatibility/synthetic smoke only
EOF
    exit 2
    ;;
esac
