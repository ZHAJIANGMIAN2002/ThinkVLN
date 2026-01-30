# SFT Training Pipeline Guide

This guide covers how to run the ThinkVLN supervised fine-tuning (SFT) training pipeline and what configuration options are available.

## Training Overview

The SFT trainer supports hybrid training with:
- **Action Mode**: Predicts next actions and progress values
- **CoT Mode**: Generates reasoning text for action determination
- **Mixed Batches**: Combines both modes in a single training batch

## Quick Start

### 1. Single GPU Training (Debug Mode)

Use the VSCode debugger with the "SFT Training - Debug" configuration:
1. Open Debug panel (Ctrl+Shift+D)
2. Select "SFT Training - Debug" from dropdown
3. Press F5 to start

Or via terminal:
```bash
python thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
```

### 2. Multi-GPU Training (Using torchrun)

Use the VSCode debugger with the "SFT Training - Multi-GPU (torchrun)" configuration, or:

```bash
torchrun --nproc_per_node=8 thinkvln/engine/sft_trainer.py \
  --config config/sft_training.yaml
```

### 3. Multi-GPU Training (Using DeepSpeed)

```bash
deepspeed --num_gpus=8 thinkvln/engine/sft_trainer.py \
  --config config/sft_training.yaml
```

## Configuration Guide

Edit `config/sft_training.yaml` to customize your training:

### Model Configuration
```yaml
model:
  model_name_or_path: "Qwen/Qwen3-VL-2B"      # Model to fine-tune
  num_query_tokens: 4                          # Learnable query tokens
  num_action_classes: 4                        # Action vocab size
  action_loss_weight: 1.0                      # Action loss weight
  progress_loss_weight: 1.0                    # Progress loss weight
```

**Key Options:**
- `model_name_or_path`: Use Qwen3VL-2B for small, 7B for larger model
- `num_query_tokens`: Append to input sequence in action mode (keep 4)
- `num_action_classes`: 4 = {stop, forward, turn_left, turn_right}

### Data Configuration
```yaml
data:
  data_root: "data/"                                           # Base data directory
  action_data_path: "trajectory_data/R2R_back/summary_full.jsonl"
  cot_data_path: "cot_dataset/cot_dataset_answer.jsonl"
  val_split_ratio: 0.1                         # 10% validation split
```

**Data Preparation Steps:**
1. **Prepare trajectory data** (JSONL format):
   - Required fields: `episode_key`, `actions`, `subtask_sequence`, `instruction`, `plan`
   - Each line = one trajectory example
   - Place in: `data/trajectory_data/R2R_back/summary_full.jsonl`

2. **Prepare CoT data** (JSONL format):
   - Required fields: `frame_key`, `episode_key`, `instruction`, `plan`, `answer` (reasoning)
   - Each line = one CoT example
   - Place in: `data/cot_dataset/cot_dataset_answer.jsonl`

3. **Generate CoT data** (if not available):
   ```bash
   python thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
   ```

### Training Configuration
```yaml
training:
  output_dir: "checkpoints/thinkvln_actor"    # Checkpoint directory
  num_train_epochs: 3                          # Training duration
  
  # Batch size (adjust based on GPU memory)
  per_device_train_batch_size: 2              # Per GPU batch size
  gradient_accumulation_steps: 8              # Effective BS = 2×8×N_gpus
  
  # Optimizer
  learning_rate: 2.0e-5
  weight_decay: 0.01
  warmup_steps: 500
  max_grad_norm: 1.0
  lr_scheduler_type: "cosine"
  
  # Mixed precision
  bf16: true                                   # Use bfloat16 (A100/H100)
  fp16: false                                  # Use float16 (V100)
  
  # Memory optimization
  gradient_checkpointing: true                # Save memory at cost of speed
  
  # Logging & checkpointing
  logging_steps: 10
  save_steps: 1000
  save_total_limit: 3                         # Keep only last 3 checkpoints
  
  # Evaluation
  evaluation_strategy: "steps"                # "no", "steps", or "epoch"
  eval_steps: 1000
  
  # DeepSpeed
  deepspeed: "scripts/zero2.json"             # or zero3.json for large models
```

**Hardware-Specific Recommendations:**

For **8x A100 (80GB)**:
```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
bf16: true
gradient_checkpointing: false
```

For **8x V100 (32GB)**:
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
fp16: true
gradient_checkpointing: true
deepspeed: "scripts/zero3.json"
```

For **8x RTX 3090/4090 (24GB)**:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
bf16: true
gradient_checkpointing: true
deepspeed: "scripts/zero3.json"
```

### Logging Configuration
```yaml
logging:
  report_to: ["wandb"]                        # Also supports "tensorboard"
  run_name: "thinkvln-actor-sft"              # WandB run name
  logging_dir: null                           # Log directory (auto-set)
```

## Common Configuration Scenarios

### Scenario 1: Quick Test on Single GPU
```yaml
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
save_steps: 10
logging_steps: 2
evaluation_strategy: "no"
output_dir: "checkpoints/test"
```

### Scenario 2: Production Training on 8x A100
```yaml
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-5
warmup_steps: 1000
save_steps: 500
bf16: true
deepspeed: "scripts/zero2.json"
```

### Scenario 3: Memory-Constrained Training
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
gradient_checkpointing: true
deepspeed: "scripts/zero3.json"
bf16: true
num_train_epochs: 2
```

## VSCode Debug Configurations

### 1. SFT Training - Debug
- Single GPU debugging
- Uses default config: `config/sft_training.yaml`
- GPU: 0
- Best for: Development and debugging

### 2. SFT Training - Custom Config
- Interactive config selection
- Choose GPU devices
- Best for: Testing different configurations

### 3. SFT Training - Multi-GPU (torchrun)
- Multi-GPU training with torchrun
- 2 GPUs (edit `nproc_per_node` for more)
- Best for: Multi-GPU debugging

### 4. Dataset Test - Verify Data Loading
- Test dataset loading without training
- Verify data format and integrity
- Best for: Diagnosing data issues

### 5. CoT Generation Pipeline
- Generate CoT data from trajectories
- Interactive input/output paths
- Best for: Data preparation

### 6. Environment Evaluation
- Evaluate trained model on environments
- Model path and output directory inputs
- Best for: Model evaluation

## Running Training from Terminal

### Single GPU
```bash
python thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
```

### Multi-GPU with torchrun
```bash
torchrun --nproc_per_node=8 \
  --master_port=29500 \
  thinkvln/engine/sft_trainer.py \
  --config config/sft_training.yaml
```

### Multi-GPU with DeepSpeed
```bash
deepspeed --num_gpus=8 thinkvln/engine/sft_trainer.py \
  --config config/sft_training.yaml \
  --deepspeed scripts/zero2.json
```

### Override Config from CLI
```bash
python thinkvln/engine/sft_trainer.py \
  --config config/sft_training.yaml \
  --output_dir checkpoints/custom_run \
  --num_train_epochs 5 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 4
```

## Data Format Reference

### Action Data (trajectory_data/R2R_back/summary_full.jsonl)
```json
{
  "episode_key": "train_0",
  "instruction": "Walk forward...",
  "plan": "First go straight...",
  "actions": [1, 1, 3, 0],
  "subtask_sequence": ["navigate", "stop"]
}
```

### CoT Data (cot_dataset/cot_dataset_answer.jsonl)
```json
{
  "frame_key": "train_0_frame_0",
  "episode_key": "train_0",
  "instruction": "Walk forward...",
  "plan": "First go straight...",
  "answer": "Looking at the scene, I see a path ahead. I should go forward."
}
```

## Monitoring Training

### WandB Dashboard
All training metrics are logged to WandB by default:
- Training/validation loss
- Learning rate schedule
- GPU memory usage
- Throughput (samples/sec)

### Tensorboard
```bash
tensorboard --logdir=checkpoints/thinkvln_actor
```

### Local Logs
Check `checkpoints/thinkvln_actor/trainer_state.json` for training progress.

## Troubleshooting

### CUDA Out of Memory
1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing: true`
4. Use DeepSpeed ZeRO-3: `deepspeed: scripts/zero3.json`

### Slow Data Loading
1. Increase `dataloader_num_workers` (default: 4)
2. Set `dataloader_pin_memory: true`
3. Use SSD for data instead of network drive

### Training Not Starting
1. Verify data files exist: `data/trajectory_data/R2R_back/summary_full.jsonl`
2. Check data format matches expected schema
3. Test dataset loading with "Dataset Test" debug config

### Model Not Converging
1. Check learning rate (default: 2e-5)
2. Verify data quality and labels
3. Monitor gradients via WandB
4. Try longer warmup: increase `warmup_steps`

## Resume Training

Training automatically resumes from latest checkpoint if it exists:
```bash
python thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
# Will resume from: checkpoints/thinkvln_actor/checkpoint-*/
```

To start fresh:
```bash
rm -rf checkpoints/thinkvln_actor
python thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
```

## Performance Tips

1. **Batch Size**: Start with 2-4, adjust based on available VRAM
2. **Gradient Accumulation**: Larger accumulation = more stable training
3. **Learning Rate**: Start with 2e-5, adjust by 2x based on convergence
4. **Warmup**: Use 5-10% of total steps
5. **Checkpointing**: Enable gradient checkpointing if OOM
6. **Mixed Precision**: Use bf16 on A100/H100, fp16 on V100

## Next Steps

1. Prepare your data in JSONL format
2. Update paths in `config/sft_training.yaml`
3. Test with "Dataset Test" debug config
4. Start training with "SFT Training - Debug" config
5. Monitor progress on WandB dashboard
6. Evaluate model with "Environment Evaluation" config
