#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/lodo_example.yaml}"
shift || true
python -m rc_irstd.pipelines.run_lodo --config "${CONFIG}" "$@"
