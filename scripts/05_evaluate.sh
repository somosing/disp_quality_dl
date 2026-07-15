#!/usr/bin/env bash
set -euo pipefail
python -m disp_quality.cli.evaluate \
  --checkpoint outputs/thesis_final_fullres/best.pth \
  --split-file splits/train_val.json \
  --split-name val \
  --output-dir evaluation_output
