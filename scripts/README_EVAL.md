# ThinkVLN Actor Evaluation Guide

This guide explains how to evaluate trained ThinkVLN actor models.

## Quick Start

### 1. Default Evaluation (Using Config)

Evaluate the default model specified in `config/eval_config.yaml`:

```bash
bash scripts/eval_thinkvln_actor.sh
```

### 2. Evaluate Specific Model

Evaluate a specific checkpoint:

```bash
bash scripts/eval_thinkvln_actor.sh outputs/actor/lora/run-2-3-11-07
```

### 3. Custom Configuration

Use a custom evaluation config:

```bash
bash scripts/eval_thinkvln_actor.sh outputs/actor/lora/run-2-3-11-07 config/my_eval_config.yaml
```

## Configuration File

The evaluation configuration file (`config/eval_config.yaml`) contains:

### Model Settings
- `model_path`: Path to trained checkpoint (LoRA adapter or full model)
- `base_model_path`: Path to base Qwen3VL model (REQUIRED for LoRA checkpoints)
- `device`: Device to use ("cuda" or "cpu")

### Data Settings
- `image_root`: Root directory for R2R images
- `action_data_path`: Path to action prediction data (JSONL)
- `cot_data_path`: Path to Chain-of-Thought data (JSONL)
- `num_query_tokens`: Number of query tokens (must match training)
- `sample_ratio`: Proportion of data to use (1.0 = all, 0.1 = 10% for quick testing)

### Evaluation Settings
- `batch_size`: Batch size per device
- `num_workers`: Number of dataloader workers
- `metrics`:
  - `progress_metric`: "l1", "mse", or "huber"
  - `action_metric`: "accuracy"
- `output_dir`: Directory to save results

## Advanced Usage

### Direct Python Call

For more control, call the evaluation script directly:

```bash
# Evaluate LoRA checkpoint
python thinkvln/engine/evaluate.py \
    --config config/eval_config.yaml \
    --model_path outputs/actor/lora/run-2-3-11-07 \
    --base_model_path /mnt/swx/ThinkVLN/model_weights/qwen3vl-2 \
    --batch_size 32 \
    --progress_metric l1

# Evaluate full model (no base_model_path needed)
python thinkvln/engine/evaluate.py \
    --config config/eval_config.yaml \
    --model_path outputs/actor/final_model \
    --batch_size 32
```

### Override Configuration

Command-line arguments override config file values:

```bash
python thinkvln/engine/evaluate.py \
    --config config/eval_config.yaml \
    --batch_size 32 \
    --device cuda:1 \
    --progress_metric mse
```

## Output

Evaluation results are saved to the output directory (default: `eval_results/`):

### Metrics Saved
- `eval_loss`: Overall evaluation loss
- `eval_action_accuracy`: Action classification accuracy
- `eval_progress_l1`: Progress L1 error (or MSE/Huber)
- `eval_lm_loss`: Language modeling loss (for CoT samples)
- `eval_perplexity`: Perplexity (for CoT samples)

### Output Files
- `eval_results.json`: JSON file with all metrics
- Console output: Formatted metric display

## Example Output

```
================================================================================
Evaluation Results:
================================================================================
  eval_action_accuracy: 0.8523
  eval_lm_loss: 1.2345
  eval_loss: 0.4567
  eval_perplexity: 3.4321
  eval_progress_l1: 0.0234
================================================================================
Results saved to eval_results/eval_results.json
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config or via command line
- Use CPU: `--device cpu`

### Model Not Found
- Check model path exists: `ls outputs/actor/lora/run-2-3-11-07`
- Verify it's a valid checkpoint (contains `adapter_config.json` for LoRA or `config.json` for full model)
- For LoRA checkpoints: Ensure `base_model_path` is set in config or via `--base_model_path`

### Data Path Issues
- Ensure `image_root` points to R2R images
- Verify data paths in config are correct
- Check file permissions

## GPU Selection

Edit `scripts/eval_thinkvln_actor.sh` to change GPU:

```bash
export GPUS="4"  # Use GPU 4
export GPUS="0"  # Use GPU 0
```

For multiple GPUs, evaluation typically only needs one GPU.
