#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -z "${RC_STAGE1_GPU:-}" ]]; then
  echo "RC_STAGE1_GPU must be set to one physical GPU ID: 0, 1, or 2" >&2
  exit 2
fi
case "$RC_STAGE1_GPU" in
  0|1|2) ;;
  *)
    echo "RC_STAGE1_GPU must be 0, 1, or 2; got: $RC_STAGE1_GPU" >&2
    exit 2
    ;;
esac

for argument in "$@"; do
  case "$argument" in
    --data-parallel)
      echo "train_stage1_single_gpu.sh forbids --data-parallel" >&2
      exit 2
      ;;
    --device|--device=*)
      echo "train_stage1_single_gpu.sh freezes --device cuda" >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$RC_STAGE1_GPU"
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
    echo "PYTHON_BIN is not an executable file: $python_candidate" >&2
    exit 2
  fi
  # Preserve the final virtualenv symlink so Python still discovers pyvenv.cfg.
  python_bin="$(realpath -s "$python_candidate")"
else
  python_bin="$(command -v "$python_candidate" || true)"
  if [[ -z "$python_bin" ]]; then
    echo "PYTHON_BIN command was not found: $python_candidate" >&2
    exit 2
  fi
fi
export PYTHON_BIN="$python_bin"

"$python_bin" - <<'PY'
import os
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(
        f"Selected interpreter cannot import torch: {sys.executable}: {exc}"
    ) from exc

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected interpreter")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        "single-GPU Stage 1 requires exactly one visible CUDA device; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, "
        f"observed {torch.cuda.device_count()}"
    )
print(f"Validated interpreter: {sys.executable}")
print(
    f"Validated torch: {torch.__version__}; physical GPU "
    f"{os.environ['CUDA_VISIBLE_DEVICES']} is logical cuda:0"
)
PY

# Do not pass --data-parallel: each process owns one physical GPU and can be
# launched independently for a baseline, proposed objective, or ablation.
exec "$python_bin" -m scripts.train_multisource_tail \
  --device cuda \
  "$@"
