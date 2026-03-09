#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate StreamVLN on the ARVLNDataset action accuracy.

For each frame sample, resets model state and runs a single forward pass
(run_model=True with faked depth/pose), then compares the first 4 predicted
actions against ground-truth from extract_action_chunk.

Usage:
    python thinkvln/engine/eval_streamvln_ar.py \
        --model_path /path/to/streamvln_checkpoint \
        --action_data_path data/trajectory_data/R2R_back/summary_full.jsonl \
        --image_root /mnt/nvme/swx/dataset/R2R \
        --num_future_steps 4 \
        --max_samples 500 \
        --output_path results/streamvln_ar_eval.json
"""

import sys
import os

# Ensure streamvln and project root are importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "streamvln"))

import re
import copy
import json
import argparse
import logging
import numpy as np
import torch
import transformers
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm

from thinkvln.dataset.ar_dataset import ARVLNDataset
from thinkvln.tools.dataset_utils import extract_action_chunk

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# StreamVLN constants
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
MEMORY_TOKEN_INDEX = -300
DEFAULT_MEMORY_TOKEN = "<memory>"

ACTIONS2IDX = OrderedDict({
    "STOP": 0,
    "↑": 1,
    "←": 2,
    "→": 3,
})
ACTION_PATTERN = re.compile("|".join(re.escape(a) for a in ACTIONS2IDX))

# Default camera intrinsic (matches streamvln_agent.py default)
DEFAULT_INTRINSIC = np.array([
    [192.0,   0.0, 191.42857143, 0.0],
    [  0.0, 192.0, 191.42857143, 0.0],
    [  0.0,   0.0,           1.0, 0.0],
    [  0.0,   0.0,           0.0, 1.0],
])


def load_streamvln(model_path: str, device: str):
    from streamvln.model.stream_video_vln import StreamVLNForCausalLM

    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, model_max_length=4096, padding_side="right"
    )
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.add_tokens(["<memory>"], special_tokens=True)

    logger.info(f"Loading StreamVLN model from {model_path}")
    config = transformers.AutoConfig.from_pretrained(model_path)
    model = StreamVLNForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    model.reset(1)
    logger.info("StreamVLN model loaded")
    return model, tokenizer


def build_prompt(instruction: str, step_id: int) -> str:
    prompt = (
        f"You are an autonomous navigation assistant. Your task is to {instruction}. "
        "Devise an action sequence to follow the instruction using the four actions: "
        "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
        "MOVE FORWARD (↑) by 25 centimeters, or STOP."
        " Please devise an action sequence to follow the instruction which may include "
        "turning left or right by a certain degree, moving forward by a certain distance "
        "or stopping once the task is complete."
    )
    if step_id != 0:
        prompt += f" You have visited these areas {DEFAULT_MEMORY_TOKEN}."
    prompt += f" you can see {DEFAULT_IMAGE_TOKEN}."
    return prompt


def preprocess_qwen(prompt: str, tokenizer, image_token_index: int, memory_token_index: int, add_system: bool):
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.chat_template = chat_template

    source = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]
    input_id = []
    if add_system:
        input_id += tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}])
    for conv in source:
        encode_id = tokenizer.apply_chat_template([conv])
        input_id += encode_id

    # Replace image/memory token IDs
    img_tok_id = tokenizer.convert_tokens_to_ids("<image>")
    mem_tok_id = tokenizer.convert_tokens_to_ids("<memory>")
    for idx, eid in enumerate(input_id):
        if eid == img_tok_id:
            input_id[idx] = image_token_index
        if eid == mem_tok_id:
            input_id[idx] = memory_token_index

    return torch.tensor([input_id], dtype=torch.long)


def parse_actions(output: str) -> list:
    matches = ACTION_PATTERN.findall(output)
    return [ACTIONS2IDX[m] for m in matches]


def run_single_step(model, tokenizer, image_processor, rgb_np: np.ndarray,
                    instruction: str, device: str) -> list:
    """Run one StreamVLN forward pass from scratch (stateless)."""
    model.reset_for_env(0)

    # Preprocess image
    image = Image.fromarray(rgb_np).convert("RGB")
    image_tensor = image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    # Fake depth/pose (model ignores them in practice when no voxel pooling)
    depth = torch.zeros(rgb_np.shape[0], rgb_np.shape[1], 1, dtype=torch.float32)
    pose = torch.from_numpy(np.eye(4)).float()
    intrinsic = torch.from_numpy(DEFAULT_INTRINSIC).float()

    # Build input_ids
    prompt = build_prompt(instruction, step_id=0)
    input_ids = preprocess_qwen(
        prompt, tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        memory_token_index=MEMORY_TOKEN_INDEX,
        add_system=True,
    )

    input_dict = {
        "images": image_tensor.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16),
        "depths": depth.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16),
        "poses": pose.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16),
        "intrinsics": intrinsic.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.bfloat16),
        "inputs": input_ids.to(device),
        "env_id": 0,
        "time_ids": [[0]],
        "task_type": [0],
    }

    with torch.no_grad():
        outputs = model.generate(
            **input_dict,
            do_sample=False,
            num_beams=1,
            max_new_tokens=64,
            use_cache=True,
            return_dict_in_generate=True,
        )

    decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0].strip()
    actions = parse_actions(decoded)
    return actions, decoded


def episode_to_dir_key(episode_key: str) -> str:
    parts = episode_key.split("_")
    if len(parts) == 2:
        return f"{parts[0]}_r2r_{int(parts[1]):06d}"
    return episode_key


def evaluate(args):
    device = args.device

    # Load model
    model, tokenizer = load_streamvln(args.model_path, device)
    image_processor = model.get_vision_tower().image_processor

    # Load dataset
    dataset = ARVLNDataset(
        action_data_path=args.action_data_path,
        image_root=args.image_root,
        num_future_steps=args.num_future_steps,
    )

    samples = dataset.samples
    if args.max_samples and args.max_samples < len(samples):
        import random
        random.seed(args.seed)
        samples = random.sample(samples, args.max_samples)
        logger.info(f"Subsampled to {len(samples)} samples")

    logger.info(f"Evaluating {len(samples)} samples")

    # Per-step accuracy accumulators (4 steps)
    num_steps = args.num_future_steps
    step_correct = [0] * num_steps
    step_total = [0] * num_steps
    total_correct = 0
    total_preds = 0

    results = []

    for sample in tqdm(samples, desc="Evaluating"):
        episode_key = sample["episode_key"]
        frame_idx = sample["frame_idx"]
        instruction = sample["instruction"]
        dir_key = episode_to_dir_key(episode_key)
        image_path = os.path.join(args.image_root, dir_key, f"{frame_idx:06d}_rgb.jpg")

        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue

        rgb_np = np.array(Image.open(image_path).convert("RGB"))

        # Ground truth
        gt_actions, _ = extract_action_chunk(
            frame_idx, sample["actions"], sample["subtask_sequence"], num_steps
        )

        # Predict
        try:
            pred_actions, raw_output = run_single_step(
                model, tokenizer, image_processor, rgb_np, instruction, device
            )
        except Exception as e:
            logger.warning(f"Error on {episode_key} frame {frame_idx}: {e}")
            pred_actions = []
            raw_output = ""

        # Pad predictions to num_steps (default STOP=0 for missing predictions)
        pred_actions = (pred_actions + [0] * num_steps)[:num_steps]

        # Accumulate per-step metrics
        for k in range(num_steps):
            step_correct[k] += int(pred_actions[k] == gt_actions[k])
            step_total[k] += 1
            total_correct += int(pred_actions[k] == gt_actions[k])
            total_preds += 1

        results.append({
            "episode_key": episode_key,
            "frame_idx": frame_idx,
            "gt_actions": gt_actions,
            "pred_actions": pred_actions,
            "raw_output": raw_output,
        })

    # Compute metrics
    metrics = {
        "overall_accuracy": total_correct / total_preds if total_preds > 0 else 0.0,
        "num_samples": len(results),
        "total_predictions": total_preds,
    }
    for k in range(num_steps):
        acc = step_correct[k] / step_total[k] if step_total[k] > 0 else 0.0
        metrics[f"step_{k+1}_accuracy"] = acc

    logger.info("=" * 60)
    logger.info("StreamVLN AR Evaluation Results")
    logger.info("=" * 60)
    for key, val in metrics.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.4f}")
        else:
            logger.info(f"  {key}: {val}")
    logger.info("=" * 60)

    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)
        logger.info(f"Results saved to {args.output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate StreamVLN on AR action dataset")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to StreamVLN model checkpoint")
    parser.add_argument("--action_data_path", type=str, required=True,
                        help="Path to action JSONL file (ARVLNDataset format)")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root directory for RGB images")
    parser.add_argument("--num_future_steps", type=int, default=4,
                        help="Number of future actions to predict and evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (None = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str, default="results/streamvln_ar_eval.json",
                        help="Path to save evaluation results")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
