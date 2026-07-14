#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${1:-$ROOT/validation/release_smoke}
cd "$ROOT"
python -m compileall -q rc_irstd model losses rc tests
for script in scripts/*.sh; do bash -n "$script"; done
pytest -q
EPOCHS=${SMOKE_EPOCHS:-3} bash scripts/smoke_two_stage_no_reject.sh "$OUT"
bash scripts/mshnet_integration_test.sh
