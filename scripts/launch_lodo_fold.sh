#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/lodo_example.yaml}"
shift || true

# Examples:
#   ./scripts/launch_lodo_fold.sh configs/paper.yaml
#   ./scripts/launch_lodo_fold.sh configs/paper.yaml --outer-target RealScene-ISTD
#   ./scripts/launch_lodo_fold.sh configs/paper.yaml --stages detector export episodes
python -m rc_irstd.pipelines.run_lodo --config "$CONFIG" "$@"
