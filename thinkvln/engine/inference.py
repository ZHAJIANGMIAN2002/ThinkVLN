#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ThinkVLN Model Inference Script

This script loads a merged LoRA model and performs inference on random samples
from the dataset to check the model's output quality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import random
import argparse
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import List, Dict, Any, Optional


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_instruction_and_plan(content: str) -> tuple:
    """Extract instruction and plan from prompt content."""
    instruction = ""
    plan = ""
    
    # Find instruction
    if "**Instruction**: " in content:
        instruction_start = content.find("**Instruction**: ") + len("**Instruction**: ")
        instruction_end = content.find("\n\n**Plan**:", instruction_start)
        if instruction_end == -1:
            instruction_end = content.find("\n\nAnalyze", instruction_start)
        instruction = content[instruction_start:instruction_end].strip()
    
    # Find plan
    if "**Plan**: " in content:
        plan_start = content.find("**Plan**: ") + len("**Plan**: ")
        plan_end = content.find("\n\nAnalyze", plan_start)
        if plan_end == -1:
            plan_end = len(content)
        plan = content[plan_start:plan_end].strip()
    
    return instruction, plan


def load_image(image_path: str) -> Image.Image:
    """Load image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert('RGB')


def load_model_and_processor(model_path: str, device: str = "cuda"):
    """
    Load model and processor from the given path.
    
    Args:
        model_path: Path to the model directory
        device: Device to use (cuda or cpu)
    
    Returns:
        Tuple of (model, processor)
    """
    # Use auto device map only if generic "cuda" is requested and not in distributed mode
    # Otherwise, we'll manually move the model to the specific device
    use_device_map = device == "cuda"
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if use_device_map else None,
        trust_remote_code=True
    )
    
    if not use_device_map:
        model = model.to(device)
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    return model, processor


def run_inference(model, processor, user_message: str, image=None, device: str = "cuda", max_new_tokens: int = 2048) -> str:
    """
    Run inference on a single sample.
    """
    return run_batch_inference(model, processor, [user_message], [image], device, max_new_tokens)[0]


def run_batch_inference(model, processor, user_messages: List[str], images: List[Optional[Image.Image]] = None, device: str = "cuda", max_new_tokens: int = 2048) -> List[str]:
    """
    Run inference on a batch of samples.
    
    Args:
        model: The loaded model
        processor: The loaded processor
        user_messages: List of user message texts
        images: List of PIL Image objects (optional)
        device: Device to use (cuda or cpu)
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        List of generated text strings
    """
    if images is None:
        images = [None] * len(user_messages)
    
    batch_messages = []
    for user_msg, image in zip(user_messages, images):
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_msg},
                    ],
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": user_msg,
                },
            ]
        batch_messages.append(messages)
    
    # Apply chat template to each message in the batch
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]
    
    # Extract images for the processor
    batch_images = []
    for img in images:
        if img is not None:
            batch_images.append([img])
        else:
            batch_images.append(None)

    # Prepare inputs using processor
    # Note: Qwen2VL/Qwen3VL processor takes a list of texts and optionally a list of lists of images
    inputs = processor(
        text=texts,
        images=batch_images,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    # Extract only the new generated parts
    input_length = inputs['input_ids'].shape[1]
    generated_ids_trimmed = generated_ids[:, input_length:]
    
    # Decode outputs
    generated_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return generated_texts


def parse_action_from_output(output: str) -> str:
    """Parse action from model output."""
    # Look for [action] section
    if "[action]" in output.lower():
        action_start = output.lower().find("[action]")
        action_text = output[action_start:].split('\n')[0]
        
        # Extract action value
        action_lines = output[action_start:].split('\n')
        for line in action_lines[1:]:
            line = line.strip().lower()
            if line in ['forward', 'turn_left', 'turn_right', 'stop']:
                return line
            elif 'forward' in line:
                return 'forward'
            elif 'turn_left' in line or 'left' in line:
                return 'turn_left'
            elif 'turn_right' in line or 'right' in line:
                return 'turn_right'
            elif 'stop' in line:
                return 'stop'
    
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="ThinkVLN Model Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged model directory")
    parser.add_argument("--dataset_path", type=str, 
                        default="data/cot_dataset/sft_dataset_100.json",
                        help="Path to dataset JSON file")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of random samples to test")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save inference results (optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 80)
    print("ThinkVLN Model Inference")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} samples from dataset")
    
    # Randomly select samples
    selected_samples = random.sample(dataset, min(args.num_samples, len(dataset)))
    print(f"Selected {len(selected_samples)} samples for inference")
    
    # Load model and processor
    print("\n[2/4] Loading model and processor...")
    try:
        model, processor = load_model_and_processor(args.model_path, args.device)
        print("Model and processor loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare results
    results = []
    
    # Run inference on each sample
    print("\n[3/4] Running inference...")
    for idx, sample in enumerate(selected_samples, 1):
        print(f"\n{'='*80}")
        print(f"Sample {idx}/{len(selected_samples)}")
        print(f"{'='*80}")
        
        try:
            # Extract messages and images
            messages = sample.get("messages", [])
            images = sample.get("images", [])
            
            if not messages:
                print("Warning: No messages found in sample, skipping...")
                continue
            
            # Get user message (first message)
            user_message = None
            for msg in messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message:
                print("Warning: No user message found, skipping...")
                continue
            
            # Extract instruction and plan
            instruction, plan = extract_instruction_and_plan(user_message)
            
            print(f"\nInstruction: {instruction}")
            print(f"\nPlan:\n{plan}")
            
            # Load image
            if images:
                image_path = images[0]
                print(f"\nImage path: {image_path}")
                try:
                    image = load_image(image_path)
                    print(f"Image loaded: {image.size}")
                except Exception as e:
                    print(f"Error loading image: {e}")
                    image = None
            else:
                print("Warning: No image found in sample")
                image = None
            
            # Run inference
            print("\nGenerating response...")
            generated_text = run_inference(
                model, processor, user_message, image, args.device, args.max_new_tokens
            )
            
            print(f"\nGenerated output:\n{generated_text}")
            
            # Parse action
            action = parse_action_from_output(generated_text)
            print(f"\nParsed action: {action}")
            
            # Store result
            result = {
                "sample_idx": idx,
                "instruction": instruction,
                "plan": plan,
                "image_path": images[0] if images else None,
                "generated_output": generated_text,
                "parsed_action": action
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results if output path is specified
    if args.output_path:
        print(f"\n[4/4] Saving results to {args.output_path}...")
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Results saved successfully")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Inference Summary")
    print("=" * 80)
    print(f"Total samples processed: {len(results)}")
    actions = [r["parsed_action"] for r in results]
    action_counts = {action: actions.count(action) for action in set(actions)}
    print(f"Action distribution: {action_counts}")
    print("=" * 80)


if __name__ == "__main__":
    main()

