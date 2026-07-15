#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/thesis_final_fullres.yaml}"
python -m disp_quality.cli.train --config "$CONFIG"
