#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# This is the claim-bearing schema-v5 no-Reject calibrator.  The older
# reference-package trainer has a deliberately separate launcher:
# scripts/train_calibrator_reference.sh.
export CUDA_VISIBLE_DEVICES="${RC_CALIBRATOR_GPU:-0}"
default_project_python="$(dirname "$repo_root")/BasicIRSTD/infrarenet/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_candidate="$PYTHON_BIN"
elif [[ -x "$default_project_python" ]]; then
  python_candidate="$default_project_python"
else
  python_candidate="python"
fi
if [[ "$python_candidate" == */* ]]; then
  if [[ ! -x "$python_candidate" ]]; then
    echo "ERROR: PYTHON_BIN is not executable: $python_candidate" >&2
    exit 2
  fi
  python_bin="$python_candidate"
else
  python_bin="$(command -v "$python_candidate" || true)"
  if [[ -z "$python_bin" ]]; then
    echo "ERROR: PYTHON_BIN command was not found: $python_candidate" >&2
    exit 2
  fi
fi
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"
exec "$python_bin" -m rc.train_calibrator_risk_aligned \
  --device cuda \
  "$@"
