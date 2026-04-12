#!/usr/bin/env bash
set -e

PROJECT_ROOT=~/Master_Thesis/software/disp_quality_dl
DATA_ROOT=/home/msingh/Master_Thesis/dataset/raft_data
YOLO_MODEL=/home/msingh/Master_Thesis/dataset/x_wing_yolo/weights/07_sept_best.pt

cd "$PROJECT_ROOT"
source .venv/bin/activate

for ds in \
  02_oct_2025_db_schenker_raft \
  03_12_25_db_schenker \
  06_nov_2025_db_schenker \
  11_sept_2025_raben \
  17_sept_2025_db_schenker_raft \
  24_nov_2025_db_schenker_raft \
  25_nov_2025_db_schenker_raft
do
  echo "Processing $ds ..."
  PYTHONPATH=. python scripts/visualize_targets.py \
    --dataset-root "$DATA_ROOT/$ds" \
    --intensity-glob "**/kl_depth_intensity_0.png" \
    --sensor-glob "**/kl_depth_disparity_0.png" \
    --raft-glob "**/raft_disp.npy" \
    --yolo-model "$YOLO_MODEL" \
    --num-samples 999999 \
    --output-root "outputs/target_debug/$ds"
done
