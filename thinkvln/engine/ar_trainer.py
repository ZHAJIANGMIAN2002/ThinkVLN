#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto-regressive VLN trainer with FlashAttention-2, LoRA and vision freezing.

This script trains a Qwen3VL model to predict the next 4 actions and
discretized progress bins as plain LM tokens, following the StreamVLN style.
All configuration is controlled by a YAML file under config/.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

import torch
from torch.utils.data import random_split
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from transformers import Qwen3VLForConditionalGeneration
import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")

from thinkvln.dataset.ar_dataset import ARVLNDataset, ARVLNDataCollator


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ARTrainingArguments(TrainingArguments):
    """Extended TrainingArguments for AR VLN training."""

    # Model
    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-2B",
        metadata={"help": "Pretrained Qwen3VL path or HF ID"},
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "Use FlashAttention-2 attention implementation"},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA for parameter-efficient tuning"},
    )
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated module names to apply LoRA"},
    )

    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Freeze visual encoder"},
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Freeze language model"},
    )

    # Data
    image_root: str = field(
        default="/mnt/nvme/swx/dataset/R2R",
        metadata={"help": "Root directory for RGB images"},
    )
    action_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to action JSONL file"},
    )
    val_split_ratio: float = field(
        default=0.1,
        metadata={"help": "Validation split ratio"},
    )
    sample_ratio: float = field(
        default=1.0,
        metadata={"help": "Subsample ratio for training data"},
    )
    num_future_steps: int = field(
        default=4,
        metadata={"help": "Number of future (action, progress) steps"},
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_data_path is None:
            raise ValueError("action_data_path must be provided for AR training")
        if not 0.0 <= self.val_split_ratio < 1.0:
            raise ValueError(f"val_split_ratio must be in [0,1), got {self.val_split_ratio}")
        if not 0.0 < self.sample_ratio <= 1.0:
            raise ValueError(f"sample_ratio must be in (0,1], got {self.sample_ratio}")


def create_datasets(
    args: ARTrainingArguments,
) -> Tuple[ARVLNDataset, Optional[ARVLNDataset]]:
    dataset = ARVLNDataset(
        action_data_path=args.action_data_path,
        image_root=args.image_root,
        num_future_steps=args.num_future_steps,
    )

    # Optional subsampling
    if args.sample_ratio < 1.0:
        n_total = len(dataset)
        n_use = max(1, int(n_total * args.sample_ratio))
        perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
        indices = perm[:n_use].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        logger.info(f"Subsampled AR dataset to {n_use} / {n_total} ({args.sample_ratio:.2%})")

    if args.val_split_ratio <= 0.0:
        return dataset, None

    n_total = len(dataset)
    val_size = int(n_total * args.val_split_ratio)
    train_size = n_total - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"Split AR dataset into {len(train_dataset)} train and {len(val_dataset)} val samples")
    return train_dataset, val_dataset


def add_progress_tokens(processor, progress_bin_step: int = 5) -> None:
    """Add progress bin tokens <p_0> ... <p_100> to tokenizer as special tokens.
    
    Using special_tokens=True ensures these tokens are treated as atomic units
    and won't be split by the BPE tokenizer.
    """
    tokens = [f"<p_{i * progress_bin_step}>" for i in range(0, 101 // progress_bin_step + 1)]
    added = processor.tokenizer.add_tokens(tokens, special_tokens=True)
    if added > 0:
        logger.info(f"Added {added} progress tokens to tokenizer as special tokens")
        logger.info(f"Sample progress tokens: {tokens[:3]} ... {tokens[-1]}")


def load_model(args: ARTrainingArguments):
    """Load Qwen3VL Causal LM with optional FlashAttention-2, LoRA and freezing."""
    from peft import LoraConfig, get_peft_model

    logger.info(f"Loading base model from {args.model_name_or_path}")

    model_kwargs: Dict[str, Any] = {
        "dtype": torch.bfloat16 if args.bf16 else torch.float32,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    # LoRA
    if args.use_lora:
        if args.lora_target_modules:
            target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA: r={args.lora_r}, alpha={args.lora_alpha}, target={target_modules}")
        model.print_trainable_parameters()

    # Freeze vision tower if available
    if args.freeze_vision_tower:
        logger.info("Freezing vision tower")
        base = getattr(model, "base_model", model)
        visual_root = getattr(base, "model", None)
        if visual_root is not None and hasattr(visual_root, "visual"):
            for p in visual_root.visual.parameters():
                p.requires_grad = False
            logger.info("Vision tower frozen")
        else:
            logger.warning("Model has no .model.visual; skipping vision freezing")

    # Freeze language model if requested
    if args.freeze_llm:
        logger.info("Freezing language model")
        base = getattr(model, "base_model", model)
        language_root = getattr(base, "model", None)
        if language_root is not None and hasattr(language_root, "language_model"):
            for p in language_root.language_model.parameters():
                p.requires_grad = False
            logger.info("Language model frozen")
        else:
            logger.warning("Model has no .model.language_model; skipping LLM freezing")

    # Gradient checkpointing
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    logger.info(
        f"Total parameters: {sum(p.numel() for p in model.parameters()):,}, "
        f"trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    return model


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return cfg


def create_training_args_from_config(config: Dict[str, Any]) -> ARTrainingArguments:
    flat: Dict[str, Any] = {}

    # Model
    model_cfg = config.get("model", {})
    flat["model_name_or_path"] = model_cfg.get("model_name_or_path", "Qwen/Qwen3-VL-2B")
    flat["use_flash_attention_2"] = bool(model_cfg.get("use_flash_attention_2", False))
    flat["use_lora"] = bool(model_cfg.get("use_lora", False))
    flat["lora_r"] = int(model_cfg.get("lora_r", 8))
    flat["lora_alpha"] = int(model_cfg.get("lora_alpha", 16))
    flat["lora_dropout"] = float(model_cfg.get("lora_dropout", 0.05))
    if "lora_target_modules" in model_cfg:
        tm = model_cfg["lora_target_modules"]
        flat["lora_target_modules"] = ",".join(tm) if isinstance(tm, list) else tm
    flat["freeze_vision_tower"] = bool(model_cfg.get("freeze_vision_tower", False))
    flat["freeze_llm"] = bool(model_cfg.get("freeze_llm", False))

    # Data
    data_cfg = config.get("data", {})
    flat["image_root"] = data_cfg.get("image_root", "/mnt/nvme/swx/dataset/R2R")
    flat["action_data_path"] = data_cfg.get("action_data_path")
    flat["val_split_ratio"] = float(data_cfg.get("val_split_ratio", 0.1))
    flat["sample_ratio"] = float(data_cfg.get("sample_ratio", 1.0))
    flat["num_future_steps"] = int(data_cfg.get("num_future_steps", 4))

    # Training
    train_cfg = config.get("training", {})
    flat.update(
        dict(
            remove_unused_columns=False,
            output_dir=train_cfg.get("output_dir", "outputs/ar_vln"),
            num_train_epochs=int(train_cfg.get("num_train_epochs", 3)),
            max_steps=int(train_cfg.get("max_steps", -1)),
            per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
            per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 2)),
            learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
            warmup_steps=int(train_cfg.get("warmup_steps", 500)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            bf16=bool(train_cfg.get("bf16", True)),
            fp16=bool(train_cfg.get("fp16", False)),
            gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
            logging_steps=int(train_cfg.get("logging_steps", 10)),
            logging_first_step=bool(train_cfg.get("logging_first_step", True)),
            save_steps=int(train_cfg.get("save_steps", 1000)),
            save_total_limit=int(train_cfg.get("save_total_limit", 3)),
            eval_strategy=train_cfg.get("eval_strategy", "steps"),
            eval_steps=int(train_cfg.get("eval_steps", 1000)),
            deepspeed=train_cfg.get("deepspeed"),
            dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 4)),
            dataloader_pin_memory=bool(train_cfg.get("dataloader_pin_memory", True)),
            seed=int(train_cfg.get("seed", 42)),
        )
    )

    # Logging
    log_cfg = config.get("logging", {})
    flat["report_to"] = log_cfg.get("report_to", [])
    flat["run_name"] = log_cfg.get("run_name")
    flat["logging_dir"] = log_cfg.get("logging_dir")

    return ARTrainingArguments(**flat)


def main() -> None:
    parser = argparse.ArgumentParser(description="ThinkVLN Auto-Regressive Trainer")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., config/ar_training.yaml)",
    )
    cmd_args = parser.parse_args()

    config = load_config_from_yaml(cmd_args.config)
    args = create_training_args_from_config(config)

    logger.info("=" * 80)
    logger.info("ThinkVLN Auto-Regressive Trainer")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Future steps: {args.num_future_steps}")
    logger.info(f"Action data: {args.action_data_path}")
    logger.info(f"Image root: {args.image_root}")
    logger.info(f"Val split ratio: {args.val_split_ratio}")
    logger.info("=" * 80)

    # Initialize wandb if enabled
    if "wandb" in args.report_to:
        import wandb
        wandb.init(
            project=config.get("logging", {}).get("project", "thinkvln"),
            name=args.run_name,
            config={
                "model_name": args.model_name_or_path,
                "num_future_steps": args.num_future_steps,
                "batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "max_grad_norm": args.max_grad_norm,
                "lr_scheduler_type": args.lr_scheduler_type,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
                "use_lora": args.use_lora,
                "lora_r": args.lora_r if args.use_lora else None,
                "lora_alpha": args.lora_alpha if args.use_lora else None,
                "freeze_vision_tower": args.freeze_vision_tower,
                "freeze_llm": args.freeze_llm,
                "use_flash_attention_2": args.use_flash_attention_2,
                "sample_ratio": args.sample_ratio,
                "val_split_ratio": args.val_split_ratio,
            },
        )
        logger.info(f"Initialized wandb: project={wandb.run.project}, run={wandb.run.name}")

    # Processor and tokenizer
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    logger.info("Processor loaded successfully")

    # Add progress tokens to tokenizer
    add_progress_tokens(processor, progress_bin_step=5)

    # Model
    model = load_model(args)

    # Resize embeddings if tokenizer was extended
    if getattr(model, "resize_token_embeddings", None) is not None:
        model.resize_token_embeddings(len(processor.tokenizer))
        logger.info("Resized token embeddings to match tokenizer")

    # Datasets and collator
    train_dataset, eval_dataset = create_datasets(args)

    data_collator = ARVLNDataCollator(
        processor=processor,
        image_root=args.image_root,
        num_future_steps=args.num_future_steps,
        progress_bin_step=5,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving final model and processor...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # Finalize wandb if enabled
    if "wandb" in args.report_to:
        import wandb
        wandb.finish()
        logger.info("Wandb run finished")

    logger.info("All done.")


if __name__ == "__main__":
    main()


