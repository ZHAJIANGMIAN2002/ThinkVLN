#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ThinkVLN Model Evaluation Script

Standalone evaluation script for trained ThinkVLN actor models.
Supports both full fine-tuned models and LoRA adapters.

Usage:
    python thinkvln/engine/evaluate.py \
        --model_path outputs/actor/checkpoint-1000 \
        --data_config config/sft_training.yaml \
        --batch_size 16 \
        --output_dir eval_results
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
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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
            torch_dtype=torch.bfloat16,
        )
        
        # Load LoRA adapter
        from peft import PeftModel
        logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        
        # Convert all parameters to bfloat16 (including actor heads from modules_to_save)
        logger.info("Converting model to bfloat16...")
        model = model.to(dtype=torch.bfloat16)
        
        # Move entire model to target device
        logger.info(f"Moving model to {device}...")
        model = model.to(device)
        
        logger.info("LoRA adapter loaded successfully")
    else:
        logger.info("Loading full model")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = ThinkVLNActor.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
    
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
    
    collator = ThinkVLNDataCollator(
        processor=processor,
        num_query_tokens=num_query_tokens,
        action_query_token_id=action_query_token_id,
        progress_query_token_id=progress_query_token_id,
        image_root=image_root,
    )
    
    return dataset, collator


def create_dataloader(dataset, collator, batch_size: int = 16, num_workers: int = 0):
    """Create evaluation dataloader with HomogeneousBatchSampler."""
    from thinkvln.dataset.dataset import HomogeneousBatchSampler
    
    sampler = HomogeneousBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        seed=42,
    )
    
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
    
    # Initialize metric accumulators
    all_losses = []
    action_metrics = {
        "preds": [],
        "labels": [],
        "progress_preds": [],
        "progress_labels": []
    }
    cot_metrics = {"losses": []}
    
    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.get("loss")
            
            if loss is not None:
                all_losses.append(loss.item())
            
            # Collect predictions based on sample type
            if inputs.get("action_labels") is not None:
                # Action mode
                action_logits = outputs.get("action_logits")
                progress_preds = outputs.get("progress_preds")
                action_labels = inputs.get("action_labels")
                progress_labels = inputs.get("progress_labels")
                
                if action_logits is not None:
                    action_preds = torch.argmax(action_logits, dim=-1)
                    action_metrics["preds"].append(action_preds.cpu())
                    action_metrics["labels"].append(action_labels.cpu())
                
                if progress_preds is not None and progress_labels is not None:
                    action_metrics["progress_preds"].append(progress_preds.cpu())
                    action_metrics["progress_labels"].append(progress_labels.cpu())
            else:
                # CoT mode
                cot_loss = outputs.get("lm_loss")
                if cot_loss is not None:
                    cot_metrics["losses"].append(cot_loss.item())
    
    # Compute metrics
    metrics = {}
    
    # Overall loss
    if all_losses:
        metrics["eval_loss"] = np.mean(all_losses)
    
    # Action metrics
    if action_metrics["preds"]:
        action_preds = torch.cat(action_metrics["preds"], dim=0).numpy()   # [num_samples, 4]
        action_labels = torch.cat(action_metrics["labels"], dim=0).numpy() # [num_samples, 4]
        
        if action_metric == "accuracy":
            # Token-level accuracy across all 4 predicted steps
            valid_mask = action_labels != -100
            if valid_mask.sum() > 0:
                correct = (action_preds == action_labels) & valid_mask
                accuracy = correct.sum() / valid_mask.sum()
                metrics["eval_action_accuracy"] = float(accuracy)

            # First-step accuracy: only check the first of the 4 actions
            if action_labels.shape[1] > 0:
                first_preds = action_preds[:, 0]
                first_labels = action_labels[:, 0]
                first_valid = first_labels != -100
                if first_valid.sum() > 0:
                    first_correct = (first_preds == first_labels) & first_valid
                    first_acc = first_correct.sum() / first_valid.sum()
                    metrics["eval_action_first_step_accuracy"] = float(first_acc)
    
    # Progress metrics
    if action_metrics["progress_preds"]:
        progress_preds = torch.cat(action_metrics["progress_preds"], dim=0).float().numpy()
        progress_labels = torch.cat(action_metrics["progress_labels"], dim=0).float().numpy()
        
        valid_mask = ~np.isnan(progress_labels) & ~np.isinf(progress_labels) & (progress_labels != -100)
        if valid_mask.sum() > 0:
            valid_preds = progress_preds[valid_mask]
            valid_labels = progress_labels[valid_mask]
            
            if progress_metric == "l1":
                progress_error = np.abs(valid_preds - valid_labels).mean()
                metrics["eval_progress_l1"] = float(progress_error)
            elif progress_metric == "mse":
                progress_error = ((valid_preds - valid_labels) ** 2).mean()
                metrics["eval_progress_mse"] = float(progress_error)
            elif progress_metric == "huber":
                delta = 1.0
                abs_error = np.abs(valid_preds - valid_labels)
                huber = np.where(
                    abs_error <= delta,
                    0.5 * abs_error ** 2,
                    delta * (abs_error - 0.5 * delta)
                )
                metrics["eval_progress_huber"] = float(huber.mean())
    
    # CoT metrics
    if cot_metrics["losses"]:
        metrics["eval_lm_loss"] = np.mean(cot_metrics["losses"])
        metrics["eval_perplexity"] = np.exp(np.mean(cot_metrics["losses"]))
    
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
        "--output_dir",
        type=str,
        help="Directory to save evaluation results (overrides config)"
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
        output_dir = args.output_dir or eval_cfg.get('output_dir', 'eval_results')
        sample_ratio = data_cfg.get('sample_ratio', 1.0)
        
        # Build data configuration
        data_config = {
            'data': data_cfg,
            'model': {'num_query_tokens': data_cfg.get('num_query_tokens', 4)}
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
        output_dir = args.output_dir or "eval_results"
        sample_ratio = 1.0
        
        # For backward compatibility, try to infer data config
        logger.warning("No config file provided. Using default data paths from training config.")
        data_config = None
    
    if not model_path:
        raise ValueError("model_path is required")
    
    logger.info("=" * 80)
    logger.info("ThinkVLN Model Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    if base_model_path:
        logger.info(f"Base model path: {base_model_path}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Progress metric: {progress_metric}")
    logger.info(f"Action metric: {action_metric}")
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
        num_workers=num_workers
    )
    
    # Run evaluation
    metrics = evaluate_model(
        model,
        dataloader,
        device=device,
        progress_metric=progress_metric,
        action_metric=action_metric,
    )
    
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


if __name__ == "__main__":
    main()
