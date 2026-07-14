#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-/tmp/rc_irstd_release_validation}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
python -m compileall -q rc_irstd tests
for script in scripts/*.sh; do bash -n "$script"; done
python -m pytest -q
python -m rc_irstd.pipelines.smoke --work-dir "$WORK" --clean
printf 'Release validation passed. Smoke artifacts: %s\n' "$WORK"
