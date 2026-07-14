#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${1:-/tmp/rc_irstd_two_stage_smoke}
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
exec "$PYTHON_BIN" -m rc_irstd.pipelines.smoke_two_stage \
  --output-dir "$OUT" \
  --epochs "${EPOCHS:-3}"
