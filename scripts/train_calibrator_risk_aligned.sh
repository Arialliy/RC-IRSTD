#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Stage 1 uses all three physical GPUs through train_rc_3gpu.sh.  Stage 2 is a
# small meta-calibrator and intentionally runs on one explicitly selected GPU;
# distribute independent outer folds over 0/1/2 rather than DataParallel-ing
# this tiny model.  Override RC_CALIBRATOR_GPU for each fold launcher.
export CUDA_VISIBLE_DEVICES="${RC_CALIBRATOR_GPU:-0}"
python_bin="${PYTHON_BIN:-python}"

exec "$python_bin" -m rc.train_calibrator_risk_aligned \
  --device cuda \
  "$@"
