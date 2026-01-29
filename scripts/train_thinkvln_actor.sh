#!/bin/bash
set -e
CONFIG_FILE="${1:-config/sft_training.yaml}"
NUM_GPUS="${2:-8}"
METHOD="${3:-deepspeed}"

[ ! -f "$CONFIG_FILE" ] && echo "Config not found: $CONFIG_FILE" && exit 1

case $METHOD in
    single) python thinkvln/engine/sft_trainer.py --config "$CONFIG_FILE" ;;
    deepspeed) deepspeed --num_gpus=$NUM_GPUS thinkvln/engine/sft_trainer.py --config "$CONFIG_FILE" ;;
    torchrun) torchrun --nproc_per_node=$NUM_GPUS thinkvln/engine/sft_trainer.py --config "$CONFIG_FILE" ;;
    *) echo "Invalid method: $METHOD. Use single/deepspeed/torchrun" && exit 1 ;;
esac
