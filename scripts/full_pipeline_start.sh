#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/lodo_example.yaml}"
shift || true
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
exec python -m rc_irstd.pipelines.run_lodo --config "$CONFIG" "$@"
