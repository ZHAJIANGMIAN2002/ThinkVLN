#!/bin/bash
set -euo pipefail

source /home/swx/miniforge3/etc/profile.d/conda.sh
conda activate vln

TRAJ_DIR=/mnt/swx/ThinkVLN/data/trajectory_data/ScaleVLN_back
SPLIT_OUT=/mnt/swx/ThinkVLN/data/subtask_splits/ScaleVLN/subtask_splits.jsonl
DETERM_OUT=/mnt/swx/ThinkVLN/data/subtask_determination_results/ScaleVLN/subtask_determination.jsonl
SUMMARY_FULL=/mnt/swx/ThinkVLN/data/trajectory_data/ScaleVLN_back/summary_full.jsonl
COT_OUT=/mnt/swx/ThinkVLN/data/cot_dataset/scalevln_cot_dataset.jsonl
FRAME_BASE_DIR=/mnt/nvme/swx/dataset/scalevln

python /mnt/swx/ThinkVLN/third_party/thinkvln_back/cot_data/cot_pipeline_deploy.py \
  --trajectory_dir "${TRAJ_DIR}" \
  --split_output_file "${SPLIT_OUT}" \
  --determination_output_file "${DETERM_OUT}" \
  --summary_full_file "${SUMMARY_FULL}" \
  --cot_output_file "${COT_OUT}" \
  --frame_base_dir "${FRAME_BASE_DIR}" \
  --use_frames_only
