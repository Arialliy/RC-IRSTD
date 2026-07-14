#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install --upgrade pip
python -m pip install -e "${ROOT}[dev]"
echo "Installed rc-irstd from ${ROOT}"
