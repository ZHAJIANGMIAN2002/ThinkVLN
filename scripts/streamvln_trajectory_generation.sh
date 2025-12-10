#!/bin/bash
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

set -x
umask 000

# # Original multi-GPU version with timestamp
# TIME=$(date +%Y%m%d_%H%M)
# DATASET=R2R
# CONFIG_PATH=config/vln_r2r.yaml
# OUTPUT_PATH=data/trajectory_data/${DATASET}/${TIME}
# DATA_PATH=None  # Set to None to use default dataset path
#
# mkdir -p ${OUTPUT_PATH}
# torchrun --nproc_per_node=8 --master_port=$MASTER_PORT streamvln/streamvln_trajectory_generation.py \
#     --dataset ${DATASET} \
#     --config_path ${CONFIG_PATH} \
#     --output_path ${OUTPUT_PATH} \
#     --data_path ${DATA_PATH} \
#     > ${OUTPUT_PATH}/log.log 2>&1

# Single GPU version
DATASET=R2R
CONFIG_PATH=config/vln_r2r.yaml
OUTPUT_PATH=data/trajectory_data/${DATASET}_back
DATA_PATH=None
NUM_WORKERS=8  # Number of parallel workers, adjust based on your CPU cores

mkdir -p ${OUTPUT_PATH}
python streamvln/streamvln_trajectory_generation.py \
    --dataset ${DATASET} \
    --config_path ${CONFIG_PATH} \
    --output_path ${OUTPUT_PATH} \
    --world_size ${NUM_WORKERS}