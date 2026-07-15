#!/usr/bin/env bash
set -euo pipefail
python -m disp_quality.cli.infer \
  --checkpoint outputs/thesis_final_fullres/best.pth \
  --disparity /path/to/scene/disparity.png \
  --roi-mask /path/to/scene/roi_mask.png \
  --output-dir inference_output
