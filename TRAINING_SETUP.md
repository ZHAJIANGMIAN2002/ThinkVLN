# ThinkVLN Actor Training Setup

Complete implementation of the SFT trainer for ThinkVLN actor model with hybrid action prediction and chain-of-thought generation.

## ✅ Implementation Status

### Completed Components

1. **Model Architecture** (`thinkvln/models/thinkvln_actor.py`)
   - ✅ ThinkVLNActor extends Qwen3VL with action/progress heads
   - ✅ Hybrid training: action mode + CoT mode in same forward pass
   - ✅ Query token embedding injection (handled by thinkvln_model.py)
   - ✅ Automatic loss routing based on sample type

2. **Dataset API** (`thinkvln/dataset/dataset.py`)
   - ✅ API contract defined with clear documentation
   - ✅ Expected input/output formats specified
   - ✅ Action and CoT sample handling defined
   - ⏳ Full implementation (to be done in separate task)

3. **Trainer** (`thinkvln/engine/sft_trainer.py`)
   - ✅ ThinkVLNTrainingArguments with model/data/training configs
   - ✅ ThinkVLNSFTTrainer with custom compute_loss
   - ✅ WandB logging integration
   - ✅ Train/val split support
   - ✅ Distributed training support (Accelerate, DeepSpeed, torchrun)
   - ✅ YAML-based configuration system
   - ✅ Loss component tracking and logging

4. **Configuration** (`config/`)
   - ✅ `sft_training.yaml`: Comprehensive training configuration
   - ✅ `accelerate_config.yaml`: Multi-GPU training config
   - ✅ Example DeepSpeed configs (zero2.json, zero3.json)

5. **Launch Scripts**
   - ✅ `scripts/train_thinkvln_actor.sh`: Convenient training launcher
   - ✅ Supports single-GPU, multi-GPU (Accelerate/DeepSpeed/torchrun)

6. **Documentation**
   - ✅ `thinkvln/engine/README.md`: Complete training guide
   - ✅ Configuration examples and hardware recommendations
   - ✅ Troubleshooting guide

## 📋 Key Design Decisions

### Query Token Handling

**IMPORTANT**: Query tokens are added by the **data collator**, NOT the model.

```
Data Flow:
1. Collator: Adds query token IDs to input_ids (for action samples)
   input_ids = torch.cat([text_ids, query_token_ids], dim=1)

2. Model (thinkvln_model.py): Injects query embeddings automatically
   - Detects query token IDs in input_ids
   - Replaces with learned embeddings

3. Model (thinkvln_actor.py): Extracts hidden states from query positions
   query_hidden = hidden_states[:, -num_query_tokens:, :]
   - Computes action and progress predictions
```

### Loss Computation

The model handles loss weighting and combination internally:

```python
# In thinkvln_actor.py forward():
if action_labels is not None:
    # Action mode: compute action + progress loss
    total_loss = action_loss * weight_a + progress_loss * weight_p
else:
    # CoT mode: compute only LM loss
    total_loss = lm_loss

# Trainer just returns the combined loss
```

### Mixed Batch Training

Action and CoT samples can be in the same batch:

```python
# Collator output for mixed batch:
{
    "input_ids": [batch_size, seq_len],       # Mixed: some with query tokens, some without
    "action_labels": [num_action_samples, 4], # Only for action samples
    "labels": [num_cot_samples, seq_len],     # Only for CoT samples
}

# Model handles routing automatically based on action_labels presence
```

## 🚀 Quick Start

### 1. Setup

```bash
pip install torch transformers deepspeed wandb pyyaml
wandb login
export WANDB_PROJECT=thinkvln
```

### 2. Configure

Edit `config/sft_training.yaml`:

```yaml
model:
  model_name_or_path: "Qwen/Qwen3-VL-2B"
  
data:
  data_root: "data/trajectory_data/R2R_back"
  action_data_path: "summary_full.jsonl"
  cot_data_path: "../cot_dataset/cot_dataset_100_answer.jsonl"
  val_split_ratio: 0.1
  
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

### 3. Train

```bash
bash scripts/train_thinkvln_actor.sh config/sft_training.yaml 8 deepspeed
deepspeed --num_gpus=8 thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
```

## 📊 Data Format

### Action Data (summary_full.jsonl)

```json
{
  "episode_key": "17DRP5sb8fy_10154",
  "actions": [-1, 2, 2, 2, 2, 1, 1, 1, 2, 1, ...],
  "subtask_sequence": [1, 1, 1, 1, 1, 1, 1, 1, 1, ...],
  "instruction": "Walk forward in the direction...",
  "plan": ["Walk forward toward the dining room.", ...],
  "num_frames": 40
}
```

Images: `data/trajectory_data/R2R_back/{episode_key}/{episode_key}_{frame_idx:06d}.jpg`

### CoT Data (cot_dataset_*.jsonl)

```json
{
  "frame_key": "17DRP5sb8fy_10154_000035",
  "episode_key": "17DRP5sb8fy_10154",
  "instruction": "Walk forward in the direction...",
  "plan": "1. Walk forward...\n2. Veer right...",
  "action": 1,
  "answer": "[localization]\nI am standing...\n\n[reason]\n...\n\n[action]\nforward"
}
```

Images: `data/trajectory_data/R2R_back/{episode_key}/{frame_key}.jpg`

## 🔧 Next Steps (Dataset Implementation)

To complete the training system, implement in `thinkvln/dataset/dataset.py`:

### 1. ThinkVLNDataset.__init__()
- Load action data from JSONL
- Load CoT data from JSONL
- Mix samples according to action_cot_ratio
- Create index mapping

### 2. ThinkVLNDataset.__getitem__()
- Return raw sample dict with:
  - data_type: "action" or "cot"
  - episode_key, frame_key
  - instruction, plan
  - For action: actions, subtask_sequence
  - For CoT: reasoning text

### 3. ThinkVLNDataCollator.__call__()

**For Action Samples:**
```python
# 1. Create prompt
prompt = f"Based on the observation and subgoal '{subgoal}', predict the next 4 actions."

# 2. Load image
image_path = f"{data_root}/{episode_key}/{episode_key}_{frame_idx:06d}.jpg"
image = load_image(image_path)

# 3. Process with Qwen3VL processor
inputs = processor(text=prompt, images=[image], return_tensors="pt")
input_ids = inputs["input_ids"]

# 4. CRITICAL: Append query token IDs
query_tokens = torch.full((1, num_query_tokens), query_token_id, dtype=torch.long)
input_ids = torch.cat([input_ids, query_tokens], dim=1)

# 5. Create labels
action_labels = torch.tensor(actions[frame_idx:frame_idx+4])  # Next 4 actions
progress_labels = compute_progress(subtask_sequence, frame_idx)  # 0.0 to 1.0

# 6. Return
return {
    "input_ids": input_ids,
    "action_labels": action_labels,
    "progress_labels": progress_labels,
    "labels": None  # No LM loss for action samples
}
```

**For CoT Samples:**
```python
# 1. Create prompt with reasoning
prompt = f"Based on the observation and subgoal '{subgoal}', think step by step.\n{reasoning}"

# 2. Load image
image_path = f"{data_root}/{episode_key}/{frame_key}.jpg"
image = load_image(image_path)

# 3. Process with Qwen3VL processor
inputs = processor(text=prompt, images=[image], return_tensors="pt")

# 4. Create labels for LM loss (shifted input_ids)
labels = inputs["input_ids"].clone()
labels[labels == processor.pad_token_id] = -100

# 5. Return (NO query tokens for CoT!)
return {
    "input_ids": inputs["input_ids"],
    "labels": labels,
    "action_labels": None,
    "progress_labels": None
}
```

### 4. Progress Computation

```python
def compute_progress(subtask_sequence, frame_idx):
    """
    Compute progress for next 4 frames based on subtask completion.
    
    Progress = 0.0 at subtask start, 1.0 at subtask completion/transition
    """
    progress = []
    current_subtask = subtask_sequence[frame_idx]
    
    for i in range(4):
        if frame_idx + i >= len(subtask_sequence):
            progress.append(1.0)
        else:
            next_subtask = subtask_sequence[frame_idx + i]
            if next_subtask != current_subtask:
                # Subtask transition
                progress.append(1.0)
            else:
                # Within subtask - compute based on position
                subtask_start = subtask_sequence.index(current_subtask)
                subtask_end = len([s for s in subtask_sequence if s == current_subtask])
                progress_val = (frame_idx + i - subtask_start) / subtask_end
                progress.append(progress_val)
    
    return torch.tensor(progress, dtype=torch.float)
```

## 📈 Monitoring

### WandB Metrics
- `train/loss`: Combined loss
- `train/action_loss`: Action classification loss
- `train/progress_loss`: Progress regression loss  
- `train/lm_loss`: Language modeling loss (CoT)
- `eval/loss`: Validation loss
- `learning_rate`: Current LR
- `epoch`: Current epoch

### Checkpoints
- Saved to `{output_dir}/checkpoint-{step}/`
- Includes model weights, optimizer state, loss history
- Keep last N checkpoints (configurable)

## 🔍 Action Encoding

```python
ACTION_MAPPING = {
    0: "stop",         # Stop navigation
    1: "forward",      # Move forward
    2: "turn_left",    # Turn left 15 degrees
    3: "turn_right"    # Turn right 15 degrees
}
```

## 📚 References

- Model: `thinkvln/models/thinkvln_actor.py`
- Trainer: `thinkvln/engine/sft_trainer.py`
- Dataset API: `thinkvln/dataset/dataset.py`
- Config: `config/sft_training.yaml`
- Guide: `thinkvln/engine/README.md`
