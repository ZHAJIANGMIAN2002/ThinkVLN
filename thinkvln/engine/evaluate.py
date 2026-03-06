#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ThinkVLN Model Evaluation Script

Standalone evaluation script for trained ThinkVLN actor models.
Supports both full fine-tuned models and LoRA adapters.

Usage:
    python thinkvln/engine/evaluate.py \
        --model_path outputs/actor/checkpoint-1000 \
        --config config/eval_config.yaml \
        --batch_size 16 \
        --output_dir eval_results

    torchrun --nproc_per_node=4 thinkvln/engine/evaluate.py \
        --config config/eval_config.yaml \
        --model_path outputs/actor/checkpoint-1000
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import logging
import yaml
import argparse
import json
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ShardedBatchSampler(Sampler):
    """Shard a batch sampler by rank (rank gets batches i where i % world_size == rank)."""

    def __init__(self, batch_sampler, num_replicas: int, rank: int):
        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = getattr(batch_sampler, "batch_size", None)
        self.drop_last = getattr(batch_sampler, "drop_last", False)

    def __iter__(self):
        for i, batch in enumerate(self.batch_sampler):
            if i % self.num_replicas == self.rank:
                yield batch

    def __len__(self):
        total = len(self.batch_sampler)
        if total <= self.rank:
            return 0
        return (total - 1 - self.rank) // self.num_replicas + 1

    def set_epoch(self, epoch: int):
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)


def init_distributed(device_arg: str = None):
    """Initialize distributed evaluation from torchrun env if available."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
    else:
        device = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "device": device,
        "is_main_process": rank == 0,
    }


def load_model(model_path: str, base_model_path: str = None, device: str = "cuda"):
    """
    Load ThinkVLNActor model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (LoRA adapter or full model)
        base_model_path: Path to base Qwen3VL model (required for LoRA checkpoints)
        device: Device to load model on
    
    Returns:
        Tuple of (model, processor)
    """
    from transformers import AutoProcessor
    from thinkvln.models.thinkvln_actor import ThinkVLNActor
    from thinkvln.models.actor_config import ThinkVLNActorConfig
    
    logger.info(f"Loading model from {model_path}")
    use_cuda = str(device).startswith("cuda")
    model_dtype = torch.bfloat16 if use_cuda else torch.float32
    
    # Check if it's a LoRA checkpoint
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)
    
    if is_lora:
        logger.info("Detected LoRA checkpoint")
        
        # Read adapter config to get base model path
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        # Use provided base_model_path or fall back to adapter config
        actual_base_model = base_model_path or adapter_config.get("base_model_name_or_path")
        
        if not actual_base_model:
            raise ValueError(
                "Base model path is required for LoRA checkpoint. "
                "Provide it via config (model.base_model_path) or ensure it's in adapter_config.json"
            )
        
        logger.info(f"Loading base model from {actual_base_model}")
        logger.info(f"Loading LoRA adapter from {model_path}")
        
        # Load processor from base model
        processor = AutoProcessor.from_pretrained(actual_base_model, trust_remote_code=True)
        
        # Load actor config from checkpoint if exists
        actor_config_path = os.path.join(model_path, "actor_config.json")
        if os.path.exists(actor_config_path):
            with open(actor_config_path, 'r') as f:
                actor_cfg_dict = json.load(f)
            actor_config = ThinkVLNActorConfig(**actor_cfg_dict)
        else:
            # Use default actor config
            actor_config = ThinkVLNActorConfig()
            logger.warning("No actor_config.json found, using default configuration")
        
        # Load base model first (on CPU first to avoid OOM)
        logger.info("Loading base ThinkVLNActor model...")
        model = ThinkVLNActor.from_pretrained(
            actual_base_model,
            actor_config=actor_config,
            device_map="cpu",  # Load on CPU first
            torch_dtype=model_dtype,
        )
        
        # Load LoRA adapter
        from peft import PeftModel
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        
        # Convert all parameters to target dtype (including actor heads from modules_to_save)
        logger.info(f"Converting model to {model_dtype}...")
        model = model.to(dtype=model_dtype)
        
        # Move entire model to target device
        logger.info(f"Moving model to {device}...")
        model = model.to(device)
        
        logger.info("LoRA adapter loaded successfully")
    else:
        logger.info("Loading full model")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = ThinkVLNActor.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=model_dtype,
        )
        model = model.to(device)
    
    model.eval()
    logger.info("Model loaded successfully")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Verify all parameters are on the same device
    devices = {p.device for p in model.parameters()}
    logger.info(f"Model parameters on devices: {devices}")
    
    return model, processor


def create_eval_dataset(data_config: Dict[str, Any], processor, sample_ratio: float = 1.0):
    """
    Create evaluation dataset with optional sampling.
    
    Args:
        data_config: Data configuration dictionary
        processor: Qwen3VL processor
        sample_ratio: Ratio of data to use (1.0 = all data, 0.1 = 10%)
    
    Returns:
        Tuple of (dataset, collator)
    """
    from thinkvln.dataset.dataset import ThinkVLNDataset, ThinkVLNDataCollator
    from torch.utils.data import Subset
    
    data_cfg = data_config.get('data', {})
    model_cfg = data_config.get('model', {})
    
    image_root = data_cfg.get('image_root')
    action_data_path = data_cfg.get('action_data_path')
    cot_data_path = data_cfg.get('cot_data_path')
    num_query_tokens = model_cfg.get('num_query_tokens', 4)
    
    logger.info("Creating evaluation dataset")
    logger.info(f"  Image root: {image_root}")
    logger.info(f"  Action data: {action_data_path}")
    logger.info(f"  CoT data: {cot_data_path}")
    logger.info(f"  Sample ratio: {sample_ratio}")
    
    full_dataset = ThinkVLNDataset(
        action_data_path=action_data_path,
        cot_data_path=cot_data_path,
        image_root=image_root,
        skip_missing_images=True,
    )
    
    logger.info(f"Full dataset created with {len(full_dataset)} samples")
    
    # Apply sampling if sample_ratio < 1.0
    if sample_ratio < 1.0:
        n_total = len(full_dataset)
        n_use = max(1, int(n_total * sample_ratio))
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))[:n_use].tolist()
        dataset = Subset(full_dataset, indices)
        logger.info(f"Subsampled to {n_use} samples ({sample_ratio:.2%} of {n_total})")
    else:
        dataset = full_dataset
        logger.info(f"Using all {len(dataset)} samples")
    
    # Get query token IDs from model config (or use defaults)
    action_query_token_id = model_cfg.get('action_query_token_id', 151700)
    progress_query_token_id = model_cfg.get('progress_query_token_id', 151701)
    memory_num_history_images = data_cfg.get('memory_num_history_images', 8)
    done_threshold = data_cfg.get('done_threshold', 0.85)

    collator = ThinkVLNDataCollator(
        processor=processor,
        num_query_tokens=num_query_tokens,
        action_query_token_id=action_query_token_id,
        progress_query_token_id=progress_query_token_id,
        image_root=image_root,
        memory_num_history_images=memory_num_history_images,
        done_threshold=done_threshold,
    )
    
    return dataset, collator


def create_dataloader(
    dataset,
    collator,
    batch_size: int = 16,
    num_workers: int = 0,
    world_size: int = 1,
    rank: int = 0,
):
    """Create evaluation dataloader with HomogeneousBatchSampler."""
    from thinkvln.dataset.dataset import HomogeneousBatchSampler
    
    base_sampler = HomogeneousBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        seed=42,
    )

    sampler = base_sampler
    if world_size > 1:
        sampler = ShardedBatchSampler(base_sampler, num_replicas=world_size, rank=rank)
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


def evaluate_model(
    model,
    dataloader,
    device: str = "cuda",
    progress_metric: str = "l1",
    action_metric: str = "accuracy",
    done_pred_threshold: float = 0.5,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, float]:
    """
    Evaluate model on dataset.
    
    Args:
        model: ThinkVLNActor model
        dataloader: Evaluation dataloader
        device: Device to run evaluation on
        progress_metric: Metric for progress evaluation ("l1", "mse", "huber")
        action_metric: Metric for action evaluation ("accuracy")
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    is_distributed = world_size > 1 and torch.distributed.is_initialized()
    metric_device = torch.device(device if str(device).startswith("cuda") else "cpu")

    # Scalar accumulators (better for distributed all_reduce than gathering full predictions)
    loss_sum = 0.0
    loss_count = 0
    action_correct = 0
    action_total = 0
    action_first_correct = 0
    action_first_total = 0
    progress_sum = 0.0
    progress_count = 0
    done_correct = 0
    done_total = 0
    cot_loss_sum = 0.0
    cot_loss_count = 0

    logger.info("Running evaluation...")
    with torch.no_grad():
        iterator = dataloader if rank != 0 else tqdm(dataloader, desc="Evaluating")
        for batch in iterator:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.get("loss")

            if loss is not None:
                loss_sum += float(loss.item())
                loss_count += 1

            action_labels = inputs.get("action_labels")
            if action_labels is not None:
                action_logits = outputs.get("action_logits")
                progress_preds = outputs.get("progress_preds")
                done_preds = outputs.get("done_preds")
                progress_labels = inputs.get("progress_labels")
                done_labels = inputs.get("done_labels")

                if action_logits is not None and action_metric == "accuracy":
                    action_preds = torch.argmax(action_logits, dim=-1)
                    valid_mask = action_labels != -100
                    action_correct += int(((action_preds == action_labels) & valid_mask).sum().item())
                    action_total += int(valid_mask.sum().item())

                    if action_labels.shape[1] > 0:
                        first_labels = action_labels[:, 0]
                        first_preds = action_preds[:, 0]
                        first_valid = first_labels != -100
                        action_first_correct += int(((first_preds == first_labels) & first_valid).sum().item())
                        action_first_total += int(first_valid.sum().item())

                if progress_preds is not None and progress_labels is not None:
                    if progress_preds.ndim > 1:
                        progress_preds = progress_preds[:, 0]
                    valid_mask = torch.isfinite(progress_labels) & (progress_labels != -100)
                    if valid_mask.any():
                        diff = progress_preds - progress_labels
                        if progress_metric == "l1":
                            progress_sum += float(diff.abs()[valid_mask].sum().item())
                        elif progress_metric == "mse":
                            progress_sum += float((diff[valid_mask] ** 2).sum().item())
                        elif progress_metric == "huber":
                            delta = 1.0
                            abs_diff = diff.abs()[valid_mask]
                            huber = torch.where(
                                abs_diff <= delta,
                                0.5 * abs_diff ** 2,
                                delta * (abs_diff - 0.5 * delta),
                            )
                            progress_sum += float(huber.sum().item())
                        progress_count += int(valid_mask.sum().item())

                if done_preds is not None and done_labels is not None:
                    valid_mask = torch.isfinite(done_labels) & (done_labels != -100)
                    if valid_mask.any():
                        pred_binary = (done_preds[valid_mask] > done_pred_threshold).to(done_labels.dtype)
                        done_correct += int((pred_binary == done_labels[valid_mask]).sum().item())
                        done_total += int(valid_mask.sum().item())
            else:
                cot_loss = outputs.get("lm_loss")
                if cot_loss is not None:
                    cot_loss_sum += float(cot_loss.item())
                    cot_loss_count += 1

    stats = torch.tensor(
        [
            loss_sum,
            float(loss_count),
            float(action_correct),
            float(action_total),
            float(action_first_correct),
            float(action_first_total),
            progress_sum,
            float(progress_count),
            float(done_correct),
            float(done_total),
            cot_loss_sum,
            float(cot_loss_count),
        ],
        device=metric_device,
        dtype=torch.float64,
    )

    if is_distributed:
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    (
        loss_sum,
        loss_count,
        action_correct,
        action_total,
        action_first_correct,
        action_first_total,
        progress_sum,
        progress_count,
        done_correct,
        done_total,
        cot_loss_sum,
        cot_loss_count,
    ) = stats.tolist()

    metrics = {}
    if loss_count > 0:
        metrics["eval_loss"] = float(loss_sum / loss_count)
    if action_total > 0:
        metrics["eval_action_accuracy"] = float(action_correct / action_total)
    if action_first_total > 0:
        metrics["eval_action_first_step_accuracy"] = float(action_first_correct / action_first_total)
    if progress_count > 0:
        metric_key = {
            "l1": "eval_progress_l1",
            "mse": "eval_progress_mse",
            "huber": "eval_progress_huber",
        }[progress_metric]
        metrics[metric_key] = float(progress_sum / progress_count)
    if done_total > 0:
        metrics["eval_done_accuracy"] = float(done_correct / done_total)
    if cot_loss_count > 0:
        mean_cot_loss = float(cot_loss_sum / cot_loss_count)
        metrics["eval_lm_loss"] = mean_cot_loss
        metrics["eval_perplexity"] = float(np.exp(mean_cot_loss))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ThinkVLN Actor Model")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to evaluation configuration YAML file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to trained model checkpoint (overrides config)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        help="Path to base Qwen3VL model (required for LoRA checkpoints, overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for evaluation (overrides config)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="Number of dataloader workers (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run evaluation on (overrides config)"
    )
    parser.add_argument(
        "--progress_metric",
        type=str,
        choices=["l1", "mse", "huber"],
        help="Metric for progress evaluation (overrides config)"
    )
    parser.add_argument(
        "--action_metric",
        type=str,
        choices=["accuracy"],
        help="Metric for action evaluation (overrides config)"
    )
    parser.add_argument(
        "--done_pred_threshold",
        type=float,
        help="Threshold for binarizing done prediction probability (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save evaluation results (overrides config)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    
    args = parser.parse_args()
    
    # Load configuration from YAML if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract configuration values
        model_cfg = config.get('model', {})
        data_cfg = config.get('data', {})
        eval_cfg = config.get('evaluation', {})
        metrics_cfg = eval_cfg.get('metrics', {})
        
        # Set defaults from config (can be overridden by command line)
        model_path = args.model_path or model_cfg.get('model_path')
        base_model_path = args.base_model_path or model_cfg.get('base_model_path')
        device = args.device or model_cfg.get('device', 'cuda')
        batch_size = args.batch_size or eval_cfg.get('batch_size', 16)
        num_workers = args.num_workers if args.num_workers is not None else eval_cfg.get('num_workers', 0)
        progress_metric = args.progress_metric or metrics_cfg.get('progress_metric', 'l1')
        action_metric = args.action_metric or metrics_cfg.get('action_metric', 'accuracy')
        done_pred_threshold = (
            args.done_pred_threshold
            if args.done_pred_threshold is not None
            else metrics_cfg.get('done_pred_threshold', 0.5)
        )
        output_dir = args.output_dir or eval_cfg.get('output_dir', 'eval_results')
        sample_ratio = data_cfg.get('sample_ratio', 1.0)
        
        # Build data configuration
        data_config = {
            'data': data_cfg,
            'model': model_cfg,
        }
    else:
        # Fallback to command-line arguments only (backward compatibility)
        if not args.model_path:
            raise ValueError("Either --config or --model_path must be provided")
        
        model_path = args.model_path
        base_model_path = None
        device = args.device or "cuda"
        batch_size = args.batch_size or 16
        num_workers = args.num_workers or 0
        progress_metric = args.progress_metric or "l1"
        action_metric = args.action_metric or "accuracy"
        done_pred_threshold = args.done_pred_threshold if args.done_pred_threshold is not None else 0.5
        output_dir = args.output_dir or "eval_results"
        sample_ratio = 1.0
        
        # For backward compatibility, try to infer data config
        logger.warning("No config file provided. Using default data paths from training config.")
        data_config = None
    
    if not model_path:
        raise ValueError("model_path is required")
    if not 0.0 <= done_pred_threshold <= 1.0:
        raise ValueError(f"done_pred_threshold must be in [0, 1], got {done_pred_threshold}")
    
    dist = init_distributed(device)
    device = dist["device"]
    rank = dist["rank"]
    world_size = dist["world_size"]
    is_main = dist["is_main_process"]

    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)

    if is_main:
        logger.info("=" * 80)
        logger.info("ThinkVLN Model Evaluation")
        logger.info("=" * 80)
        logger.info(f"Model path: {model_path}")
        if base_model_path:
            logger.info(f"Base model path: {base_model_path}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Device: {device}")
        logger.info(f"Distributed: {world_size} processes")
        logger.info(f"Progress metric: {progress_metric}")
        logger.info(f"Action metric: {action_metric}")
        logger.info(f"Done pred threshold: {done_pred_threshold}")
        logger.info(f"Sample ratio: {sample_ratio}")
        logger.info("=" * 80)
    
    # Load model and processor
    model, processor = load_model(model_path, base_model_path, device)
    
    # Create dataset and dataloader
    if data_config is None:
        raise ValueError("Data configuration is required. Please provide --config or set data paths.")
    
    dataset, collator = create_eval_dataset(data_config, processor, sample_ratio)
    dataloader = create_dataloader(
        dataset,
        collator,
        batch_size=batch_size,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
    )
    
    # Run evaluation
    metrics = evaluate_model(
        model,
        dataloader,
        device=device,
        progress_metric=progress_metric,
        action_metric=action_metric,
        done_pred_threshold=done_pred_threshold,
        rank=rank,
        world_size=world_size,
    )

    if is_main:
        # Print results
        logger.info("=" * 80)
        logger.info("Evaluation Results:")
        logger.info("=" * 80)
        for key, value in sorted(metrics.items()):
            logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 80)

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / "eval_results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    if dist["distributed"] and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
