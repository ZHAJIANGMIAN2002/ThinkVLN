#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from torch.utils.data import Subset, random_split
from transformers import AutoProcessor, Trainer, TrainingArguments

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from thinkvln.dataset.watcher_sft_dataset import WatcherSFTCollator, WatcherSFTDataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class WatcherSFTTrainingArguments(TrainingArguments):
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-8B")
    use_flash_attention_2: bool = field(default=True)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: Optional[str] = field(default=None)
    freeze_vision_tower: bool = field(default=True)
    freeze_llm: bool = field(default=False)
    manifest_file: str = field(default="")
    annotation_file: str = field(default="")
    bundle_root: str = field(default="")
    summary_full_path: str = field(default="")
    image_stride: int = field(default=3)
    val_split_ratio: float = field(default=0.0)
    sample_ratio: float = field(default=1.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.manifest_file:
            raise ValueError("manifest_file must be provided")
        if not self.annotation_file:
            raise ValueError("annotation_file must be provided")
        if not self.bundle_root:
            raise ValueError("bundle_root must be provided")
        if not self.summary_full_path:
            raise ValueError("summary_full_path must be provided")
        if self.image_stride < 1:
            raise ValueError("image_stride must be >= 1")
        if not 0.0 <= self.val_split_ratio < 1.0:
            raise ValueError("val_split_ratio must be in [0, 1)")
        if not 0.0 < self.sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in (0, 1]")


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def create_training_args_from_config(config: Dict[str, Any]) -> WatcherSFTTrainingArguments:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", True))
    gradient_checkpointing_kwargs = train_cfg.get("gradient_checkpointing_kwargs")
    if gradient_checkpointing_kwargs is None and gradient_checkpointing:
        gradient_checkpointing_kwargs = {"use_reentrant": False}
    flat: Dict[str, Any] = {
        "remove_unused_columns": False,
        "output_dir": train_cfg.get("output_dir", "outputs/watcher_sft"),
        "model_name_or_path": model_cfg.get("model_name_or_path", "Qwen/Qwen3-VL-8B"),
        "use_flash_attention_2": bool(model_cfg.get("use_flash_attention_2", True)),
        "use_lora": bool(model_cfg.get("use_lora", True)),
        "lora_r": int(model_cfg.get("lora_r", 32)),
        "lora_alpha": int(model_cfg.get("lora_alpha", 64)),
        "lora_dropout": float(model_cfg.get("lora_dropout", 0.05)),
        "freeze_vision_tower": bool(model_cfg.get("freeze_vision_tower", True)),
        "freeze_llm": bool(model_cfg.get("freeze_llm", False)),
        "manifest_file": data_cfg.get("manifest_file", ""),
        "annotation_file": data_cfg.get("annotation_file", ""),
        "bundle_root": data_cfg.get("bundle_root", ""),
        "summary_full_path": data_cfg.get("summary_full_path", ""),
        "image_stride": int(data_cfg.get("image_stride", 3)),
        "val_split_ratio": float(data_cfg.get("val_split_ratio", 0.0)),
        "sample_ratio": float(data_cfg.get("sample_ratio", 1.0)),
        "num_train_epochs": int(train_cfg.get("num_train_epochs", 1)),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 1)),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-5)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "warmup_steps": int(train_cfg.get("warmup_steps", 0)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "bf16": bool(train_cfg.get("bf16", torch.cuda.is_available())),
        "fp16": bool(train_cfg.get("fp16", False)),
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": gradient_checkpointing_kwargs,
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "logging_first_step": bool(train_cfg.get("logging_first_step", True)),
        "save_steps": int(train_cfg.get("save_steps", 1000)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 3)),
        "eval_strategy": train_cfg.get("eval_strategy", "no"),
        "eval_steps": int(train_cfg.get("eval_steps", 1000)),
        "deepspeed": train_cfg.get("deepspeed"),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "dataloader_pin_memory": bool(train_cfg.get("dataloader_pin_memory", True)),
        "seed": int(train_cfg.get("seed", 42)),
        "ddp_find_unused_parameters": train_cfg.get("ddp_find_unused_parameters", False),
        "ddp_backend": train_cfg.get("ddp_backend", "nccl"),
        "report_to": log_cfg.get("report_to", []),
        "run_name": log_cfg.get("run_name"),
        "logging_dir": log_cfg.get("logging_dir"),
    }
    target_modules = model_cfg.get("lora_target_modules")
    if target_modules is not None:
        flat["lora_target_modules"] = ",".join(target_modules) if isinstance(target_modules, list) else str(target_modules)
    return WatcherSFTTrainingArguments(**flat)


def create_datasets(args: WatcherSFTTrainingArguments) -> Tuple[Any, Optional[Any]]:
    dataset = WatcherSFTDataset(
        manifest_file=args.manifest_file,
        annotation_file=args.annotation_file,
        bundle_root=args.bundle_root,
        summary_full_path=args.summary_full_path,
        image_stride=args.image_stride,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
    )
    if args.val_split_ratio <= 0.0:
        return dataset, None
    val_size = int(len(dataset) * args.val_split_ratio)
    train_size = len(dataset) - val_size
    if val_size <= 0 or train_size <= 0:
        return dataset, None
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )


def load_model(args: WatcherSFTTrainingArguments):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model_kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )

    if args.use_lora:
        target_modules = (
            [item.strip() for item in args.lora_target_modules.split(",")]
            if args.lora_target_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()

    base = getattr(model, "base_model", model)
    root = getattr(base, "model", None)
    if args.freeze_vision_tower and root is not None and hasattr(root, "visual"):
        for param in root.visual.parameters():
            param.requires_grad = False
    if args.freeze_llm and root is not None and hasattr(root, "language_model"):
        for param in root.language_model.parameters():
            param.requires_grad = False
    if args.use_lora and args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
    return model


def build_watcher_trainer_components(
    config: Dict[str, Any],
    processor=None,
    model=None,
):
    args = create_training_args_from_config(config)
    if processor is None:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    train_dataset, eval_dataset = create_datasets(args)
    collator = WatcherSFTCollator(processor=processor)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=getattr(processor, "tokenizer", None),
    )
    return {
        "args": args,
        "processor": processor,
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "trainer": trainer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Watcher update SFT trainer")
    parser.add_argument("--config", type=str, required=True)
    cmd_args = parser.parse_args()

    config = load_config_from_yaml(cmd_args.config)
    args = create_training_args_from_config(config)
    project = str(config.get("logging", {}).get("project", "")).strip()
    if project:
        os.environ.setdefault("WANDB_PROJECT", project)

    logger.info("=" * 80)
    logger.info("Watcher Update SFT Trainer")
    logger.info("Model: %s", args.model_name_or_path)
    logger.info("Manifest: %s", args.manifest_file)
    logger.info("Annotations: %s", args.annotation_file)
    logger.info("Bundle root: %s", args.bundle_root)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("=" * 80)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = load_model(args)
    components = build_watcher_trainer_components(config, processor=processor, model=model)
    trainer = components["trainer"]
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
