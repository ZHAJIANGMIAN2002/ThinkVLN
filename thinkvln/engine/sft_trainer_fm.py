#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SFT trainer for flow-matching waypoint actor."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import yaml
from transformers import AutoProcessor, Trainer, TrainingArguments

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from thinkvln.dataset.fm_waypoint_dataset import ThinkVLNFMDataCollator, ThinkVLNFMWaypointDataset
from thinkvln.models.fm_actor_config import FlowMatchingActorConfig
from thinkvln.models.thinkvln_fm_actor import ThinkVLNFMActor


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ThinkVLNFMTrainingArguments(TrainingArguments):
    model_name_or_path: str = field(default='Qwen/Qwen3-VL-2B')
    waypoint_data_path: str = field(default='')
    image_root: str = field(default='data')
    num_query_tokens: int = field(default=4)
    action_horizon: int = field(default=5)
    action_dim: int = field(default=2)
    memory_num_history_images: int = field(default=0)
    freeze_backbone: bool = field(default=True)

    def __post_init__(self):
        super().__post_init__()
        if not self.waypoint_data_path:
            raise ValueError("waypoint_data_path must be provided")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be > 0")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if self.memory_num_history_images < 0:
            raise ValueError("memory_num_history_images must be >= 0")


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_training_args_from_config(config: Dict[str, Any]) -> ThinkVLNFMTrainingArguments:
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})
    train_cfg = config.get('training', {})
    log_cfg = config.get('logging', {})

    args = ThinkVLNFMTrainingArguments(
        output_dir=train_cfg.get('output_dir', 'checkpoints/thinkvln_fm_actor'),
        remove_unused_columns=False,
        model_name_or_path=model_cfg.get('model_name_or_path', 'Qwen/Qwen3-VL-2B'),
        waypoint_data_path=data_cfg.get('waypoint_data_path', ''),
        image_root=data_cfg.get('image_root', 'data'),
        num_query_tokens=model_cfg.get('num_query_tokens', 4),
        action_horizon=model_cfg.get('action_horizon', 5),
        action_dim=model_cfg.get('action_dim', 2),
        memory_num_history_images=data_cfg.get('memory_num_history_images', 0),
        freeze_backbone=model_cfg.get('freeze_backbone', True),
        num_train_epochs=train_cfg.get('num_train_epochs', 1),
        max_steps=train_cfg.get('max_steps', -1),
        per_device_train_batch_size=train_cfg.get('per_device_train_batch_size', 1),
        gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 1),
        learning_rate=train_cfg.get('learning_rate', 2e-5),
        weight_decay=train_cfg.get('weight_decay', 0.0),
        warmup_steps=train_cfg.get('warmup_steps', 0),
        logging_steps=train_cfg.get('logging_steps', 10),
        save_steps=train_cfg.get('save_steps', 1000),
        save_total_limit=train_cfg.get('save_total_limit', 2),
        bf16=train_cfg.get('bf16', True),
        fp16=train_cfg.get('fp16', False),
        dataloader_num_workers=train_cfg.get('dataloader_num_workers', 2),
        dataloader_pin_memory=train_cfg.get('dataloader_pin_memory', True),
        report_to=log_cfg.get('report_to', []),
        run_name=log_cfg.get('run_name'),
        deepspeed=train_cfg.get('deepspeed'),
    )
    return args


class ThinkVLNFMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logged_first_train_batch = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
        if hasattr(loss, "ndim") and loss.ndim > 0:
            loss = loss.mean()
        if isinstance(outputs, dict):
            flow_loss = outputs.get("flow_loss")
            if flow_loss is not None and self.state.global_step % max(1, self.args.logging_steps) == 0:
                if hasattr(flow_loss, "ndim") and flow_loss.ndim > 0:
                    flow_loss = flow_loss.mean()
                self.log({"train/flow_loss": float(flow_loss.detach().item())})
        if not self._logged_first_train_batch:
            input_shape = tuple(inputs["input_ids"].shape) if "input_ids" in inputs else None
            wp_shape = tuple(inputs["waypoint_labels"].shape) if "waypoint_labels" in inputs else None
            logger.info("First FM batch | input_ids=%s waypoint_labels=%s", input_shape, wp_shape)
            self._logged_first_train_batch = True
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description='ThinkVLN Flow Matching SFT Trainer')
    parser.add_argument('--config', type=str, required=True)
    cmd_args = parser.parse_args()

    cfg = load_config_from_yaml(cmd_args.config)
    args = create_training_args_from_config(cfg)
    logger.info("=" * 80)
    logger.info("ThinkVLN Flow Matching SFT Trainer")
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Model: %s", args.model_name_or_path)
    logger.info("Waypoint data: %s", args.waypoint_data_path)
    logger.info("Image root: %s", args.image_root)
    logger.info("Horizon=%d action_dim=%d", args.action_horizon, args.action_dim)
    logger.info("Batch/device=%d grad_accum=%d", args.per_device_train_batch_size, args.gradient_accumulation_steps)
    logger.info("=" * 80)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    fm_cfg = FlowMatchingActorConfig(
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
    )
    model = ThinkVLNFMActor.from_pretrained(
        args.model_name_or_path,
        fm_config=fm_cfg,
        trust_remote_code=True,
    )
    if args.freeze_backbone:
        # Keep only FM head/projector trainable to avoid OOM.
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)
        model.shared_projector.requires_grad_(True)
        model.flow_head.requires_grad_(True)
        logger.info("Backbone frozen: training only shared_projector + flow_head")

    dataset = ThinkVLNFMWaypointDataset(
        waypoint_data_path=args.waypoint_data_path,
        image_root=args.image_root,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
    )
    collator = ThinkVLNFMDataCollator(
        processor=processor,
        image_root=args.image_root,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        num_query_tokens=args.num_query_tokens,
        memory_num_history_images=args.memory_num_history_images,
    )

    trainer = ThinkVLNFMTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "fm_actor_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(fm_cfg), f, indent=2)
    logger.info('Saved FM model to %s', args.output_dir)


if __name__ == '__main__':
    main()
