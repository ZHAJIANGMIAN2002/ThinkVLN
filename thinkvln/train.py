#!/usr/bin/env python3
"""
Training script for ThinkVLNModel with action classification.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoConfig
from thinkvln.model.thinkvln_model import ThinkVLNModel
from thinkvln.dataset.dataset import ThinkVLNDataset, load_image
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Any


def find_all_linear_names(model):
    """Find all linear layer names for LoRA target modules."""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    # Exclude action_head and lm_head from LoRA (they need full training)
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    if "action_head" in lora_module_names:
        lora_module_names.remove("action_head")
    return list(lora_module_names)


def collate_fn(batch, processor, max_length=2048):
    """Collate function for training."""
    user_messages = [item["user_message"] for item in batch]
    assistant_messages = [item["assistant_message"] for item in batch]
    images = [item["image"] for item in batch]
    action_labels = [item["action_label"] for item in batch]
    
    # Build messages
    batch_messages = []
    for user_msg, assistant_msg, img in zip(user_messages, assistant_messages, images):
        if img is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_msg},
                    ],
                },
                {
                    "role": "assistant",
                    "content": assistant_msg,
                },
            ]
        else:
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        batch_messages.append(messages)
    
    # Apply chat template and replace [action] with <action>
    texts = []
    for msg in batch_messages:
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        # Replace [action] with <action> so it's treated as a single token
        text = text.replace("[action]", "<action>").replace("[ACTION]", "<action>")
        texts.append(text)
    
    # Debug: check if <action> is in texts (only for first batch)
    if len(texts) > 0 and "<action>" in texts[0]:
        # Verify tokenization
        test_tokenized = processor.tokenizer(texts[0], add_special_tokens=False, return_tensors="pt")
        action_token_id_check = processor.tokenizer.convert_tokens_to_ids("<action>")
        has_token = (test_tokenized["input_ids"] == action_token_id_check).any().item()
        if not has_token:
            print(f"\n[WARNING] <action> in text but not found after tokenization!")
            print(f"  Text snippet: ...{texts[0][-200:]}...")
            print(f"  Action token ID: {action_token_id_check}")
            print(f"  Tokenized IDs (last 20): {test_tokenized['input_ids'][0][-20:].tolist()}")
    
    # Prepare images
    batch_images = []
    for img in images:
        if img is not None:
            batch_images.append([img])
        else:
            batch_images.append(None)
    
    # Tokenize
    # Important: processor may re-tokenize, so we need to ensure <action> stays as one token
    # First, tokenize texts to check if <action> is preserved
    action_token_id_check = processor.tokenizer.convert_tokens_to_ids("<action>")
    
    inputs = processor(
        text=texts,
        images=batch_images,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    
    # Verify action token is in input_ids after processor
    input_ids_check = inputs["input_ids"]
    has_action_after_processor = (input_ids_check == action_token_id_check).any().item()
    
    # If action token is missing, it might have been split or truncated
    # In that case, we need to handle it differently
    
    # Create labels (only for assistant part)
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    
    # Mask user part in labels
    # Use a simpler approach: tokenize user message separately to get length
    for i, (user_msg, assistant_msg, img) in enumerate(zip(user_messages, assistant_messages, images)):
        # Build user message in the same format as above
        if img is not None:
            user_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_msg},
                    ],
                }
            ]
        else:
            user_message = [{"role": "user", "content": user_msg}]
        
        # Apply chat template and tokenize
        user_text_str = processor.apply_chat_template(
            user_message,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Replace [action] with <action> to match main processing
        user_text_str = user_text_str.replace("[action]", "<action>").replace("[ACTION]", "<action>")
        
        # Tokenize to get length
        user_tokenized = processor.tokenizer(
            user_text_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        user_len = user_tokenized["input_ids"].shape[1]
        labels[i, :user_len] = -100  # Ignore user part
    
    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Process action_labels
    action_labels_tensor = torch.tensor([
        label if label is not None else -1 
        for label in action_labels
    ], dtype=torch.long)
    
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
        "action_labels": action_labels_tensor,
        "images": inputs.get("images", None),
    }


def train_epoch(model, dataloader, optimizer, device, action_token_id, action_loss_weight=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_lm_loss = 0
    total_action_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        action_labels = batch["action_labels"].to(device)
        
        # Prepare images if available
        images = None
        if batch.get("images") is not None:
            if isinstance(batch["images"], list):
                # Process images
                images = []
                for img_list in batch["images"]:
                    if img_list is not None and len(img_list) > 0:
                        images.append(img_list[0])
                    else:
                        images.append(None)
            else:
                images = batch["images"]
        
        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            action_labels=action_labels if (action_labels >= 0).any() else None,
            action_token_id=action_token_id,
            action_loss_weight=action_loss_weight,
            images=images,
            return_dict=True,
        )
        
        loss = outputs.loss
        # Get action_loss from output (it's added as an attribute)
        action_loss = getattr(outputs, 'action_loss', None)
        if action_loss is not None:
            lm_loss = outputs.loss - action_loss_weight * action_loss
        else:
            lm_loss = outputs.loss
            action_loss = torch.tensor(0.0, device=loss.device)
        
        # Debug: print first batch info
        if num_batches == 0:
            has_action_token = (input_ids == action_token_id).any().item()
            valid_labels = (action_labels >= 0).sum().item()
            action_token_positions = []
            for i in range(input_ids.shape[0]):
                positions = (input_ids[i] == action_token_id).nonzero(as_tuple=True)[0].tolist()
                action_token_positions.append(positions)
            
            print(f"\n[Debug] Batch 0:")
            print(f"  Input IDs shape: {input_ids.shape}")
            print(f"  Has action token in input_ids: {has_action_token}")
            print(f"  Action token ID: {action_token_id}")
            print(f"  Action token positions per sample: {action_token_positions}")
            print(f"  Valid action labels: {valid_labels}/{len(action_labels)}")
            print(f"  Action labels: {action_labels.tolist()}")
            print(f"  Action loss: {action_loss.item()}")
            if hasattr(outputs, 'action_logits') and outputs.action_logits is not None:
                print(f"  Action logits shape: {outputs.action_logits.shape}")
                print(f"  Action logits: {outputs.action_logits}")
            else:
                print(f"  Action logits: None")
            print(f"  Action loss from outputs: {getattr(outputs, 'action_loss', None)}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update stats
        total_loss += loss.item()
        total_lm_loss += lm_loss.item()
        total_action_loss += action_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lm_loss": f"{lm_loss.item():.4f}",
            "action_loss": f"{action_loss.item():.4f}",
        })
    
    return {
        "loss": total_loss / num_batches,
        "lm_loss": total_lm_loss / num_batches,
        "action_loss": total_action_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./models/qwen3vl-2", help="Path to model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--action_loss_weight", type=float, default=1.0, help="Action loss weight")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--action_token", type=str, default="<action>", help="Action token string")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for training")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated list of target modules for LoRA (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # Count samples with action labels
    num_with_action = sum(1 for item in data if "action" in item and item["action"] is not None)
    print(f"Samples with action labels: {num_with_action}/{len(data)}")
    
    # Create dataset
    dataset = ThinkVLNDataset(data)
    
    # Load processor
    print(f"Loading processor from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Find action token ID and add if needed
    action_token_str = "<action>"
    need_resize = False
    if action_token_str not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_tokens([action_token_str], special_tokens=True)
        print(f"Added action token: {action_token_str}")
        need_resize = True
    
    action_token_id = processor.tokenizer.convert_tokens_to_ids(action_token_str)
    print(f"Action token ID: {action_token_id}")
    
    # Load model first
    print(f"Loading model from {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = ThinkVLNModel.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    
    # Resize token embeddings if action token was added
    if need_resize:
        model.resize_token_embeddings(len(processor.tokenizer))
        print("Resized model token embeddings")
    
    # Apply LoRA if requested
    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            print("\nSetting up LoRA...")
            
            # Determine target modules
            if args.lora_target_modules:
                target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
            else:
                target_modules = find_all_linear_names(model)
                print(f"Auto-detected LoRA target modules: {target_modules[:10]}... (showing first 10)")
            
            # Create LoRA config
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            print(f"LoRA applied! Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except ImportError:
            print("Warning: PEFT library not found. Install it with: pip install peft")
            print("Continuing without LoRA...")
            args.use_lora = False
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor, args.max_length),
    )
    
    if args.device != "cuda" or model.device.type == "cpu":
        model = model.to(args.device)
    
    # Set action token ID
    model.set_action_token_id(action_token_id)
    
    # Note: Model embeddings already resized above if needed
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        stats = train_epoch(
            model, dataloader, optimizer, args.device, 
            action_token_id, args.action_loss_weight
        )
        print(f"Epoch {epoch + 1} stats:")
        print(f"  Total Loss: {stats['loss']:.4f}")
        print(f"  LM Loss: {stats['lm_loss']:.4f}")
        print(f"  Action Loss: {stats['action_loss']:.4f}")
        
        # Save checkpoint
        checkpoint_dir = f"{args.output_dir}/checkpoint-epoch-{epoch + 1}"
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        processor.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

