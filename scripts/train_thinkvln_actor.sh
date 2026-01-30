#!/bin/bash
set -e


# export GPUS="0,1,2,3,4,5,6,7"
export GPUS="4,5,6,7"
CONFIG_FILE="config/sft_training.yaml"

NUM_GPUS=$(echo "$GPUS" | awk -F',' '{print NF}')

echo "正在启动训练..."
echo "使用显卡: $GPUS (共 $NUM_GPUS 张)"
echo "配置文件: $CONFIG_FILE"

# 限制显卡可见性 (这是最稳妥的方法，DeepSpeed 只能看到你指定的这些卡)
export CUDA_VISIBLE_DEVICES="$GPUS"

# 启动 DeepSpeed
torchrun --nproc_per_node="$NUM_GPUS" \
    thinkvln/engine/sft_trainer.py \
    --config "$CONFIG_FILE"