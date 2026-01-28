# ThinkVLN SFT Trainer

Supervised fine-tuning (SFT) trainer for ThinkVLN actor model with hybrid action prediction and chain-of-thought generation.

## Features

- **Mixed Batch Training**: Handles action and CoT samples in the same batch
- **Distributed Training**: Support for multi-GPU training via Accelerate, DeepSpeed, or torchrun
- **WandB Logging**: Comprehensive experiment tracking and visualization
- **Train/Val Split**: Automatic dataset splitting with validation metrics
- **Memory Efficient**: Gradient checkpointing, mixed precision (bf16/fp16)
- **Flexible Configuration**: YAML-based configuration system

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install torch transformers accelerate deepspeed wandb pyyaml

# Optional: Install flash-attention for faster training
pip install flash-attn --no-build-isolation
```

### 2. Setup WandB (Optional but Recommended)

```bash
# Login to WandB
wandb login

# Set project/entity (optional, can also be set in config)
export WANDB_PROJECT=thinkvln
export WANDB_ENTITY=your-username
```

### 3. Configure Training

Edit `config/sft_training.yaml` to set your training parameters:

```yaml
model:
  model_name_or_path: "Qwen/Qwen3-VL-2B"
  num_query_tokens: 4

data:
  data_root: "data/trajectory_data/R2R_back"
  action_data_path: "summary_full.jsonl"
  cot_data_path: "../cot_dataset/cot_dataset_100_answer.jsonl"
  val_split_ratio: 0.1  # 10% validation split

training:
  output_dir: "checkpoints/thinkvln_actor"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-5
  bf16: true
  deepspeed: "scripts/zero2.json"

logging:
  report_to: ["wandb"]
  run_name: "thinkvln-actor-sft"
```

### 4. Launch Training

#### Option 1: Using the Launch Script (Recommended)

```bash
# Multi-GPU with Accelerate (recommended)
bash scripts/train_thinkvln_actor.sh --method accelerate --num_gpus 8

# Multi-GPU with DeepSpeed
bash scripts/train_thinkvln_actor.sh --method deepspeed --num_gpus 8

# Single GPU
bash scripts/train_thinkvln_actor.sh --method single

# Custom config
bash scripts/train_thinkvln_actor.sh --config path/to/custom_config.yaml
```

#### Option 2: Direct Command

```bash
# Single GPU
python thinkvln/engine/sft_trainer.py --config config/sft_training.yaml

# Multi-GPU with Accelerate
accelerate launch --config_file config/accelerate_config.yaml \
    thinkvln/engine/sft_trainer.py --config config/sft_training.yaml

# Multi-GPU with DeepSpeed
deepspeed --num_gpus=8 thinkvln/engine/sft_trainer.py \
    --config config/sft_training.yaml

# Multi-GPU with torchrun
torchrun --nproc_per_node=8 thinkvln/engine/sft_trainer.py \
    --config config/sft_training.yaml
```

## Configuration Guide

### Model Configuration

```yaml
model:
  model_name_or_path: "Qwen/Qwen3-VL-2B"  # Base model to fine-tune
  num_query_tokens: 4                      # Query tokens for action prediction
  num_action_classes: 4                    # Number of action classes (0-3)
  action_loss_weight: 1.0                  # Weight for action loss
  progress_loss_weight: 1.0                # Weight for progress loss
```

### Data Configuration

```yaml
data:
  data_root: "data/trajectory_data/R2R_back"        # Root data directory
  action_data_path: "summary_full.jsonl"            # Action trajectory data
  cot_data_path: "../cot_dataset/cot_dataset.jsonl" # Chain-of-thought data
  action_cot_ratio: 0.5                             # 50% action, 50% CoT
  val_split_ratio: 0.1                              # 10% validation split
```

### Training Configuration

```yaml
training:
  # Output and logging
  output_dir: "checkpoints/thinkvln_actor"
  logging_steps: 10
  save_steps: 1000
  save_total_limit: 3
  
  # Training schedule
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  
  # Optimizer
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 500
  lr_scheduler_type: "cosine"
  
  # Mixed precision
  bf16: true  # Use bfloat16 (A100/H100)
  fp16: false # Use float16 (V100)
  
  # Memory optimization
  gradient_checkpointing: true
  
  # Evaluation
  evaluation_strategy: "steps"
  eval_steps: 1000
  
  # Distributed training
  deepspeed: "scripts/zero2.json"
  ddp_backend: "nccl"
```

### WandB Logging Configuration

```yaml
logging:
  report_to: ["wandb"]              # Can also include "tensorboard"
  run_name: "thinkvln-actor-sft"    # Custom run name
  logging_dir: null                  # Optional local log directory
```

## Multi-GPU Training

### Accelerate (Recommended)

Accelerate provides the easiest way to run distributed training:

```bash
# First time: Create accelerate config
accelerate config

# Or use the provided config
accelerate launch --config_file config/accelerate_config.yaml \
    thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
```

### DeepSpeed

DeepSpeed enables training very large models with ZeRO optimization:

```bash
# ZeRO-2 (recommended for models that fit in memory)
deepspeed --num_gpus=8 thinkvln/engine/sft_trainer.py \
    --config config/sft_training.yaml

# Make sure your config has: training.deepspeed: "scripts/zero2.json"
```

### PyTorch torchrun

Standard PyTorch distributed launcher:

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
    thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
```

## Hardware Recommendations

### 8x A100 (80GB)
- `per_device_train_batch_size: 4`
- `gradient_accumulation_steps: 4`
- `bf16: true`
- `gradient_checkpointing: false` (optional)
- Effective batch size: 128

### 8x V100 (32GB)
- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 8`
- `fp16: true`
- `gradient_checkpointing: true`
- `deepspeed: scripts/zero3.json` (for larger models)
- Effective batch size: 128

### 8x RTX 3090/4090 (24GB)
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 16`
- `bf16: true` (if supported)
- `gradient_checkpointing: true`
- `deepspeed: scripts/zero3.json`
- Effective batch size: 128

## Monitoring Training

### WandB Dashboard

Access your training metrics at: https://wandb.ai/your-username/thinkvln

Metrics logged:
- `train/loss`: Combined training loss
- `train/action_loss`: Action classification loss
- `train/progress_loss`: Progress regression loss
- `train/lm_loss`: Language modeling loss (CoT samples)
- `eval/loss`: Validation loss
- `learning_rate`: Current learning rate
- `epoch`: Current epoch

### Local Logs

Training logs are saved to:
- `{output_dir}/training_metrics.pt`: Training statistics
- `{output_dir}/checkpoint-*/loss_history.pt`: Loss history per checkpoint
- `{output_dir}/runs/`: TensorBoard logs (if enabled)

## Troubleshooting

### Out of Memory (OOM)

1. Enable gradient checkpointing: `gradient_checkpointing: true`
2. Reduce batch size: `per_device_train_batch_size: 1`
3. Increase gradient accumulation: `gradient_accumulation_steps: 16`
4. Use DeepSpeed ZeRO-3: `deepspeed: scripts/zero3.json`
5. Use mixed precision: `bf16: true` or `fp16: true`

### WandB Not Logging

1. Check if wandb is installed: `pip install wandb`
2. Login to WandB: `wandb login`
3. Set project/entity:
   ```bash
   export WANDB_PROJECT=thinkvln
   export WANDB_ENTITY=your-username
   ```
4. Verify config has: `logging.report_to: ["wandb"]`

### Distributed Training Issues

1. **Port already in use**: Change port in launch command
   ```bash
   torchrun --master_port=29501 ...
   ```

2. **NCCL timeout**: Increase timeout
   ```bash
   export NCCL_TIMEOUT=3600
   ```

3. **Multi-node training**: Configure in `config/accelerate_config.yaml`

## Next Steps

1. **Complete Dataset Implementation**: The dataset/collator API is defined but needs full implementation
2. **Evaluation**: Add task-specific evaluation metrics
3. **Inference**: Use the trained model for navigation tasks

## Advanced Usage

### Resume from Checkpoint

```yaml
training:
  resume_from_checkpoint: "checkpoints/thinkvln_actor/checkpoint-5000"
```

### Custom Validation Split

Set `val_split_ratio: 0` and provide separate validation data files in the dataset implementation.

### Hyperparameter Tuning

Use WandB Sweeps for automated hyperparameter search. See `config/wandb_sweep.yaml` (to be created).

## Citation

If you use ThinkVLN in your research, please cite:

```bibtex
@inproceedings{thinkvln2024,
  title={ThinkVLN: Reasoning-Enhanced Vision-and-Language Navigation},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```
