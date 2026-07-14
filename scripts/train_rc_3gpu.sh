#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Physical GPUs 0,1,2 become logical CUDA devices 0,1,2.  A per-domain batch
# of three lets each DataParallel replica receive the same interleaved domain
# mixture; callers may override it with another multiple of three.
for argument in "$@"; do
  case "$argument" in
    --device|--device=*)
      echo "train_rc_3gpu.sh freezes --device cuda" >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES=0,1,2
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
  # Preserve the final virtualenv symlink: resolving it to /usr/bin/python can
  # detach Python from its pyvenv.cfg and silently lose project packages.
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
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(
        f"Selected interpreter cannot import torch: {sys.executable}: {exc}"
    ) from exc

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected interpreter")
if torch.cuda.device_count() != 3:
    raise SystemExit(
        "train_rc_3gpu.sh requires exactly three visible CUDA devices after "
        f"CUDA_VISIBLE_DEVICES=0,1,2; observed {torch.cuda.device_count()}"
    )
print(f"Validated interpreter: {sys.executable}")
print(f"Validated torch: {torch.__version__}; visible CUDA devices: 3")
PY

exec "$python_bin" -m scripts.train_multisource_tail \
  --device cuda \
  --data-parallel \
  --batch-per-domain 3 \
  "$@"
