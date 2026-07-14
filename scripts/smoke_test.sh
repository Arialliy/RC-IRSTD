#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$ROOT/outputs/smoke}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -m rc_irstd.pipelines.smoke --work-dir "$WORK" --clean
python -m pytest -q
echo "Smoke test completed: $WORK"
