#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ThinkVLN Open-loop Evaluation Script

This script evaluates the model's CoT (Chain of Thought) reasoning chain generation
by comparing predicted outputs with ground truth answers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import random
import argparse
import re
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from thinkvln.engine.inference import load_model_and_processor, run_batch_inference, load_image
from thinkvln.dataset.dataset import ThinkVLNDataset, collate_fn


def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
    else:
        rank = 0
        world_size = 1
        gpu = 0
    return rank, world_size, gpu


# Regular expression patterns
LOCALIZATION_PATTERN = r'\[localization\](.*?)(?=\[(?:subtask determination|causal observation|reason|action)\]|$)'
SUBTASK_DETERMINATION_PATTERN = r'\[subtask determination\](.*?)(?=\[(?:causal observation|reason|action)\]|$)'
CAUSAL_OBSERVATION_PATTERN = r'\[causal observation\](.*?)(?=\[(?:reason|action)\]|$)'
REASON_PATTERN = r'\[reason\](.*?)(?=\[action\]|$)'
ACTION_PATTERN = r'\[action\](.*?)$'

PREV_SUBTASK_PATTERN = r'\[prev subtask\]\s*(\d+)'
CUR_SUBTASK_PATTERN = r'\[cur subtask\]\s*(\d+)'


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def sample_validation_set(dataset: List[Dict[str, Any]], eval_ratio: float, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Sample a validation set from the dataset.
    
    Args:
        dataset: Full dataset
        eval_ratio: Ratio of samples to use for evaluation (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Validation set (list of samples)
    """
    if seed is not None:
        random.seed(seed)
    
    total_samples = len(dataset)
    num_eval_samples = max(1, int(total_samples * eval_ratio))
    
    validation_set = random.sample(dataset, num_eval_samples)
    return validation_set


def parse_subtask(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse subtask format: number (description)
    
    Args:
        text: Text containing subtask information
    
    Returns:
        Dict with "number" and "description" keys, or None if parsing fails
    """
    pattern = r'(\d+)\s*\(([^)]+)\)'
    match = re.search(pattern, text)
    if match:
        return {
            "number": int(match.group(1)),
            "description": match.group(2).strip()
        }
    return None


def parse_reasoning_chain(output: str) -> Dict[str, Any]:
    """
    Parse reasoning chain from model output.
    
    Args:
        output: Model output text
    
    Returns:
        Structured dictionary with parsed sections
    """
    result = {
        "localization": None,
        "subtask_determination": None,
        "causal_observation": None,
        "reason": None,
        "action": None,
        "prev_subtask": None,
        "cur_subtask": None
    }
    
    # Extract sections
    localization_match = re.search(LOCALIZATION_PATTERN, output, re.DOTALL | re.IGNORECASE)
    if localization_match:
        result["localization"] = localization_match.group(1).strip()
    
    subtask_match = re.search(SUBTASK_DETERMINATION_PATTERN, output, re.DOTALL | re.IGNORECASE)
    if subtask_match:
        result["subtask_determination"] = subtask_match.group(1).strip()
    
    causal_match = re.search(CAUSAL_OBSERVATION_PATTERN, output, re.DOTALL | re.IGNORECASE)
    if causal_match:
        result["causal_observation"] = causal_match.group(1).strip()
    
    reason_match = re.search(REASON_PATTERN, output, re.DOTALL | re.IGNORECASE)
    if reason_match:
        result["reason"] = reason_match.group(1).strip()
    
    action_match = re.search(ACTION_PATTERN, output, re.DOTALL | re.IGNORECASE)
    if action_match:
        action_text = action_match.group(1).strip()
        result["action"] = action_text.lower().strip()
        # Extract action value (first non-empty line)
        for line in action_text.split('\n'):
            line = line.strip().lower()
            if line in ['forward', 'turn_left', 'turn_right', 'stop']:
                result["action"] = line
                break
    
    # Extract prev and cur subtask from subtask_determination
    if result["subtask_determination"]:
        prev_match = re.search(PREV_SUBTASK_PATTERN, result["subtask_determination"], re.IGNORECASE)
        if prev_match:
            result["prev_subtask"] = {
                "number": int(prev_match.group(1))
            }
        
        cur_match = re.search(CUR_SUBTASK_PATTERN, result["subtask_determination"], re.IGNORECASE)
        if cur_match:
            result["cur_subtask"] = {
                "number": int(cur_match.group(1))
            }
    
    return result


def evaluate_format(output: str) -> bool:
    """
    Check if the output has all required sections.
    
    Args:
        output: Model output text
    
    Returns:
        True if all sections exist, False otherwise
    """
    required_sections = [
        LOCALIZATION_PATTERN,
        SUBTASK_DETERMINATION_PATTERN,
        CAUSAL_OBSERVATION_PATTERN,
        REASON_PATTERN,
        ACTION_PATTERN
    ]
    
    for pattern in required_sections:
        if not re.search(pattern, output, re.DOTALL | re.IGNORECASE):
            return False
    
    return True


def evaluate_prev_subtask(predicted: Optional[Dict[str, Any]], ground_truth: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate prev subtask matching.
    
    Args:
        predicted: Predicted subtask dict with "number"
        ground_truth: Ground truth subtask dict with "number"
    
    Returns:
        Dict with "number_match" boolean and "predicted_number", "gt_number" values
    """
    if predicted is None or ground_truth is None:
        return {
            "number_match": False,
            "predicted_number": None,
            "gt_number": None
        }
    
    pred_num = predicted.get("number")
    gt_num = ground_truth.get("number")
    number_match = pred_num == gt_num
    
    return {
        "number_match": number_match,
        "predicted_number": pred_num,
        "gt_number": gt_num
    }


def evaluate_cur_subtask(predicted: Optional[Dict[str, Any]], ground_truth: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate cur subtask matching.
    
    Args:
        predicted: Predicted subtask dict with "number"
        ground_truth: Ground truth subtask dict with "number"
    
    Returns:
        Dict with "number_match" boolean and "predicted_number", "gt_number" values
    """
    return evaluate_prev_subtask(predicted, ground_truth)


def evaluate_action(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """
    Evaluate action matching.
    
    Args:
        predicted: Predicted action string
        ground_truth: Ground truth action string
    
    Returns:
        True if actions match, False otherwise
    """
    if predicted is None or ground_truth is None:
        return False
    
    pred_action = predicted.lower().strip()
    gt_action = ground_truth.lower().strip()
    
    return pred_action == gt_action


def evaluate_dataset(model, processor, dataloader, device, max_new_tokens, rank=0, world_size=1) -> Optional[Dict[str, Any]]:
    """
    Evaluate the model on the validation set using batch inference.
    """
    all_results = []
    
    if rank == 0:
        print(f"\nEvaluating {len(dataloader.dataset)} samples...")
        print("=" * 80)

    # Use tqdm on rank 0
    pbar = tqdm(dataloader, disable=rank != 0)
    
    for batch_idx, batch in enumerate(pbar):
        user_messages = [item["user_message"] for item in batch]
        images = [item["image"] for item in batch]
        assistant_messages = [item["assistant_message"] for item in batch]
        
        try:
            predicted_outputs = run_batch_inference(
                model, processor, user_messages, images, device, max_new_tokens
            )
            
            for i, (pred, gt) in enumerate(zip(predicted_outputs, assistant_messages)):
                # Print expected and actual for each sample (only rank 0 to avoid clutter)
                if rank == 0:
                    print(f"\n--- Sample {batch_idx * len(batch) + i + 1} ---")
                    print(f"Expected Output:\n{gt}")
                    print(f"\nActual Output:\n{pred}")
                    print("-" * 80)
                
                # Parse outputs
                predicted_parsed = parse_reasoning_chain(pred)
                ground_truth_parsed = parse_reasoning_chain(gt)
                
                # Evaluate
                format_ok = evaluate_format(pred)
                prev_subtask_eval = evaluate_prev_subtask(
                    predicted_parsed.get("prev_subtask"),
                    ground_truth_parsed.get("prev_subtask")
                )
                cur_subtask_eval = evaluate_cur_subtask(
                    predicted_parsed.get("cur_subtask"),
                    ground_truth_parsed.get("cur_subtask")
                )
                action_ok = evaluate_action(
                    predicted_parsed.get("action"),
                    ground_truth_parsed.get("action")
                )
                
                # Store result
                result = {
                    "format_correct": int(format_ok),
                    "prev_subtask": {
                        "match": int(prev_subtask_eval["number_match"]),
                        "predicted": prev_subtask_eval["predicted_number"],
                        "gt": prev_subtask_eval["gt_number"]
                    },
                    "cur_subtask": {
                        "match": int(cur_subtask_eval["number_match"]),
                        "predicted": cur_subtask_eval["predicted_number"],
                        "gt": cur_subtask_eval["gt_number"]
                    },
                    "action_match": int(action_ok),
                    "ground_truth": gt,
                    "predicted": pred
                }
                all_results.append(result)
        
        except Exception as e:
            if rank == 0:
                print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # Gather results from all ranks
    if world_size > 1:
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, all_results)
        if rank == 0:
            all_results = [item for sublist in gathered_results for item in sublist]
    
    if rank == 0:
        total = len(all_results)
        if total == 0:
            return {"summary": {}, "results": []}
            
        summary = {
            "total_samples": total,
            "format_accuracy": round(sum(r["format_correct"] for r in all_results) / total * 100, 2),
            "prev_subtask_number_accuracy": round(sum(r["prev_subtask"]["match"] for r in all_results) / total * 100, 2),
            "cur_subtask_number_accuracy": round(sum(r["cur_subtask"]["match"] for r in all_results) / total * 100, 2),
            "action_accuracy": round(sum(r["action_match"] for r in all_results) / total * 100, 2)
        }
        
        return {
            "summary": summary,
            "results": all_results
        }
    
    return None


def main():
    parser = argparse.ArgumentParser(description="ThinkVLN Open-loop Evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged model directory")
    parser.add_argument("--dataset_path", type=str,
                        default="data/cot_dataset/sft_dataset.json",
                        help="Path to dataset JSON file")
    parser.add_argument("--eval_ratio", type=float, default=0.01,
                        help="Ratio of samples to use for evaluation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu, default: cuda)")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Maximum number of new tokens to generate (default: 2048)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation (default: 1)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save evaluation results (optional)")
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        results_dir = "/mnt/swx/ThinkVLN/results"
        model_name = os.path.basename(args.model_path.rstrip("/"))
        args.output_path = os.path.join(results_dir, f"{model_name}_eval.json")
    
    # Initialize distributed mode
    rank, world_size, gpu = init_distributed_mode()
    device = f"cuda:{gpu}" if world_size > 1 else args.device
    
    # Set random seed
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    
    if rank == 0:
        print("=" * 80)
        print("ThinkVLN Open-loop Evaluation (Distributed)")
        print("=" * 80)
        print(f"Model path: {args.model_path}")
        print(f"Dataset path: {args.dataset_path}")
        print(f"Evaluation ratio: {args.eval_ratio}")
        print(f"Random seed: {args.seed}")
        print(f"World size: {world_size}")
        print(f"Batch size: {args.batch_size}")
        print("=" * 80)
    
    # Load dataset
    if rank == 0:
        print("\n[1/4] Loading dataset...")
    dataset_data = load_dataset(args.dataset_path)
    
    # Sample validation set
    if rank == 0:
        print(f"\n[2/4] Sampling validation set (ratio: {args.eval_ratio})...")
    validation_samples = sample_validation_set(dataset_data, args.eval_ratio, args.seed)
    
    eval_dataset = ThinkVLNDataset(validation_samples)
    
    sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4
    )
    
    if rank == 0:
        print(f"Selected {len(validation_samples)} samples for evaluation")
    
    # Load model and processor
    if rank == 0:
        print("\n[3/4] Loading model and processor...")
    try:
        model, processor = load_model_and_processor(args.model_path, device)
        if rank == 0:
            print("Model and processor loaded successfully")
    except Exception as e:
        print(f"Error loading model on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        if world_size > 1:
            dist.destroy_process_group()
        return
    
    # Run evaluation
    if rank == 0:
        print("\n[4/4] Running evaluation...")
    
    eval_results = evaluate_dataset(
        model, processor, dataloader, device, args.max_new_tokens, rank, world_size
    )
    
    # Only rank 0 handles results processing and saving
    if rank == 0 and eval_results:
        # Print summary
        print("\n" + "=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        summary = eval_results["summary"]
        print(f"Total samples evaluated: {summary['total_samples']}")
        print(f"\nAccuracy Metrics:")
        print(f"  Format correctness: {summary['format_accuracy']:.2f}%")
        print(f"  Prev subtask number match: {summary['prev_subtask_number_accuracy']:.2f}%")
        print(f"  Cur subtask number match: {summary['cur_subtask_number_accuracy']:.2f}%")
        print(f"  Action match: {summary['action_accuracy']:.2f}%")
        print("=" * 80)
        
        # Save results if output path is specified
        if args.output_path:
            print(f"\nSaving results to {args.output_path}...")
            output_data = {
                "metadata": {
                    "model_path": args.model_path,
                    "dataset_path": args.dataset_path,
                    "eval_ratio": args.eval_ratio,
                    "seed": args.seed,
                    "world_size": world_size,
                    "batch_size": args.batch_size,
                    "max_new_tokens": args.max_new_tokens
                },
                "summary": summary,
                "results": eval_results["results"]
            }
            
            os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print("Results saved successfully")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

