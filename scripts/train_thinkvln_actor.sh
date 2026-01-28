#!/bin/bash
# ThinkVLN Actor Training Launch Script
#
# This script provides convenient ways to launch training with different configurations.
# It supports single-GPU, multi-GPU with Accelerate, DeepSpeed, and torchrun.

set -e  # Exit on error

# Default values
CONFIG_FILE="config/sft_training.yaml"
NUM_GPUS=8
LAUNCH_METHOD="accelerate"  # Options: single, accelerate, deepspeed, torchrun

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --method)
            LAUNCH_METHOD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE       Path to config YAML (default: config/sft_training.yaml)"
            echo "  --num_gpus N        Number of GPUs to use (default: 8)"
            echo "  --method METHOD     Launch method: single, accelerate, deepspeed, torchrun (default: accelerate)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Single GPU"
            echo "  $0 --method single"
            echo ""
            echo "  # Multi-GPU with Accelerate (recommended)"
            echo "  $0 --method accelerate --num_gpus 8"
            echo ""
            echo "  # Multi-GPU with DeepSpeed"
            echo "  $0 --method deepspeed --num_gpus 8"
            echo ""
            echo "  # Multi-GPU with torchrun"
            echo "  $0 --method torchrun --num_gpus 8"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "============================================"
echo "ThinkVLN Actor Training"
echo "============================================"
echo "Config file: $CONFIG_FILE"
echo "Launch method: $LAUNCH_METHOD"
echo "Number of GPUs: $NUM_GPUS"
echo "============================================"
echo ""

# Set environment variables for WandB (optional)
# export WANDB_PROJECT=thinkvln
# export WANDB_ENTITY=your-username
# export WANDB_RUN_NAME=custom-run-name

case $LAUNCH_METHOD in
    single)
        echo "Launching single-GPU training..."
        python thinkvln/engine/sft_trainer.py --config "$CONFIG_FILE"
        ;;
    
    accelerate)
        echo "Launching multi-GPU training with Accelerate..."
        
        # Check if accelerate config exists
        ACCELERATE_CONFIG="config/accelerate_config.yaml"
        if [ ! -f "$ACCELERATE_CONFIG" ]; then
            echo "Warning: Accelerate config not found: $ACCELERATE_CONFIG"
            echo "Using default accelerate settings..."
            accelerate launch \
                --num_processes=$NUM_GPUS \
                --multi_gpu \
                --mixed_precision=bf16 \
                thinkvln/engine/sft_trainer.py --config "$CONFIG_FILE"
        else
            # Update num_processes in accelerate config if needed
            accelerate launch \
                --config_file "$ACCELERATE_CONFIG" \
                --num_processes=$NUM_GPUS \
                thinkvln/engine/sft_trainer.py --config "$CONFIG_FILE"
        fi
        ;;
    
    deepspeed)
        echo "Launching multi-GPU training with DeepSpeed..."
        deepspeed --num_gpus=$NUM_GPUS \
            thinkvln/engine/sft_trainer.py --config "$CONFIG_FILE"
        ;;
    
    torchrun)
        echo "Launching multi-GPU training with torchrun..."
        torchrun --nproc_per_node=$NUM_GPUS \
            --master_port=29500 \
            thinkvln/engine/sft_trainer.py --config "$CONFIG_FILE"
        ;;
    
    *)
        echo "Error: Unknown launch method: $LAUNCH_METHOD"
        echo "Valid options: single, accelerate, deepspeed, torchrun"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Training completed!"
echo "============================================"
