#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Physical GPUs 0,1,2 become logical CUDA devices 0,1,2.  A per-domain batch
# of three lets each DataParallel replica receive the same interleaved domain
# mixture; callers may override it with another multiple of three.
export CUDA_VISIBLE_DEVICES=0,1,2
python_bin="${PYTHON_BIN:-python}"

exec "$python_bin" -m scripts.train_multisource_tail \
  --device cuda \
  --data-parallel \
  --batch-per-domain 3 \
  --risk-objective margin \
  "$@"
