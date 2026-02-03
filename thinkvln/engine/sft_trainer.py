#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ThinkVLN SFT Trainer

This module implements the supervised fine-tuning (SFT) trainer for ThinkVLN actor model.
It supports hybrid training with both action prediction and chain-of-thought (CoT) generation.

Training Modes:
    - Action Mode: Predicts next 4 actions and progress values
    - CoT Mode: Generates reasoning text for action determination
    - Mixed Batches: Combines both modes in a single batch for efficient training

Usage:
    # Single GPU
    python thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
    
    # Multi-GPU with DeepSpeed
    deepspeed --num_gpus=8 thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 thinkvln/engine/sft_trainer.py --config config/sft_training.yaml
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Disable tokenizers parallelism warning in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import logging
import yaml
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union, Tuple
from transformers import (
    Trainer,
    TrainingArguments,
    AutoProcessor,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.integrations import WandbCallback

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ThinkVLNTrainingArguments(TrainingArguments):
    """
    Extended training arguments for ThinkVLN actor training.
    
    Extends HuggingFace TrainingArguments with ThinkVLN-specific parameters
    for model configuration and data paths.
    
    Model Arguments:
        model_name_or_path: Pretrained Qwen3VL model path or identifier
        num_query_tokens: Number of learnable query tokens for action prediction
        num_action_classes: Number of discrete action classes (0=stop, 1=forward, 2=left, 3=right)
        action_loss_weight: Weight for action classification loss
        progress_loss_weight: Weight for progress regression loss
    
    Data Arguments:
        image_root: Root directory containing image files
        action_data_path: Path to action JSONL file (absolute path)
        cot_data_path: Path to CoT JSONL file (absolute path)
        val_split_ratio: Ratio of data to use for validation (0.1 = 10% validation)
    
    Training Arguments:
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
    """
    
    # Model arguments
    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-2B",
        metadata={"help": "Path to pretrained Qwen3VL model or model identifier from huggingface.co/models"}
    )
    num_query_tokens: int = field(
        default=4,
        metadata={"help": "Number of learnable query tokens for action/progress prediction"}
    )
    num_action_classes: int = field(
        default=4,
        metadata={"help": "Number of action classes (0=stop, 1=forward, 2=turn_left, 3=turn_right)"}
    )
    action_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for action classification loss"}
    )
    progress_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for progress regression loss"}
    )
    use_huber_loss_for_progress: bool = field(
        default=False,
        metadata={"help": "Use Huber (SmoothL1) instead of MSE for progress regression, more robust to outliers"}
    )

    # LoRA configuration
    use_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha (scaling factor)"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of module names to apply LoRA"}
    )
    
    # Vision tower configuration
    freeze_vision_tower: bool = field(
        default=False,
        metadata={"help": "Freeze vision encoder to save memory"}
    )
    
    # Attention optimization
    use_flash_attention_2: bool = field(
        default=False,
        metadata={"help": "Use FlashAttention-2 for faster and more memory-efficient attention"}
    )
    
    # Data arguments
    image_root: str = field(
        default="/mnt/nvme/swx/dataset/R2R",
        metadata={"help": "Root directory for image files"}
    )
    action_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to action data JSONL file (absolute path)"}
    )
    cot_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to CoT data JSONL file (absolute path)"}
    )

    val_split_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of data to use for validation (0.1 = 10% validation)"}
    )
    sample_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of full dataset to use for training (1.0 = all data, 0.1 = 10%). Val split applied after subsampling."}
    )
    
    # Training-specific
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to save memory"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to checkpoint directory to resume training from (e.g., outputs/actor/checkpoint-1000)"}
    )
    
    def __post_init__(self):
        """Validate arguments after initialization."""
        super().__post_init__()
        
        # Validate data paths
        if self.action_data_path is None and self.cot_data_path is None:
            raise ValueError("At least one of action_data_path or cot_data_path must be provided")
        
        # Validate validation split
        if not 0.0 <= self.val_split_ratio < 1.0:
            raise ValueError(f"val_split_ratio must be between 0 and 1, got {self.val_split_ratio}")
        if not 0.0 < self.sample_ratio <= 1.0:
            raise ValueError(f"sample_ratio must be in (0, 1], got {self.sample_ratio}")
        
        # Validate loss weights
        if self.action_loss_weight < 0 or self.progress_loss_weight < 0:
            raise ValueError("Loss weights must be non-negative")


class ThinkVLNSFTTrainer(Trainer):
    """
    Custom Trainer for ThinkVLN actor supervised fine-tuning.
    
    This trainer extends HuggingFace Trainer to handle homogeneous batches.
    Each batch contains only one type of sample (action OR CoT, not mixed).
    
    Key Features:
        - Homogeneous batch training using HomogeneousBatchSampler
        - Automatic loss routing based on sample type
        - Individual loss component logging
        - Support for DeepSpeed and distributed training
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize trainer with standard HF Trainer arguments."""
        super().__init__(*args, **kwargs)
        
        # Track loss components for logging
        self.loss_history = {
            "action_loss": [],
            "progress_loss": [],
            "lm_loss": [],
        }
    
    def get_train_dataloader(self):
        """
        Returns training dataloader with HomogeneousBatchSampler.
        Ensures each batch contains only one type of sample (action or CoT).
        """
        from torch.utils.data import DataLoader
        from thinkvln.dataset.dataset import HomogeneousBatchSampler
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        # Create homogeneous batch sampler
        train_sampler = HomogeneousBatchSampler(
            dataset=train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            shuffle=True,
            seed=self.args.seed,
        )
        
        return DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute loss for a batch of samples.
        
        The model automatically handles routing based on the presence of action_labels:
        - If action_labels is not None: Action mode (action + progress loss)
        - If action_labels is None: CoT mode (LM loss only)
        
        Args:
            model: ThinkVLNActor model
            inputs: Dictionary of input tensors from data collator
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - pixel_values: Image tensor or None
                - image_grid_thw: Image grid info or None
                - action_labels: [batch_size, 4] or None (for action samples)
                - progress_labels: [batch_size, 4] or None (for action samples)
                - labels: [batch_size, seq_len] or None (for CoT samples)
        
        Returns:
            loss: Combined loss tensor
            outputs (optional): Model outputs for logging
        """
        # Forward pass - model handles routing internally
        outputs = model(**inputs)
        
        # Extract combined loss (already weighted by model)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        
        # Log individual loss components for monitoring
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_loss_components(outputs)
        
        return (loss, outputs) if return_outputs else loss
    
    def _log_loss_components(self, outputs: Dict[str, Any]):
        """
        Log individual loss components for monitoring.
        
        Args:
            outputs: Model forward outputs containing loss components
        """
        metrics = {}
        
        # Extract loss components from model outputs
        if isinstance(outputs, dict):
            if "action_loss" in outputs and outputs["action_loss"] is not None:
                action_loss_value = outputs["action_loss"].item()
                metrics["train/action_loss"] = action_loss_value
                self.loss_history["action_loss"].append(action_loss_value)
            
            if "progress_loss" in outputs and outputs["progress_loss"] is not None:
                progress_loss_value = outputs["progress_loss"].item()
                metrics["train/progress_loss"] = progress_loss_value
                self.loss_history["progress_loss"].append(progress_loss_value)
            
            if "lm_loss" in outputs and outputs["lm_loss"] is not None:
                lm_loss_value = outputs["lm_loss"].item()
                metrics["train/lm_loss"] = lm_loss_value
                self.loss_history["lm_loss"].append(lm_loss_value)
        
        # Log metrics if any were collected
        if metrics:
            self.log(metrics)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint with loss history.
        
        Overrides parent method to also save loss component history.
        """
        # Call parent save (metrics not supported in parent signature)
        super()._save_checkpoint(model, trial)
        
        # Save loss history
        if self.args.should_save:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            
            loss_history_path = os.path.join(output_dir, "loss_history.pt")
            torch.save(self.loss_history, loss_history_path)
            logger.info(f"Saved loss history to {loss_history_path}")


def load_model(args: ThinkVLNTrainingArguments):
    """
    Load ThinkVLNActor model from pretrained Qwen3VL with optional LoRA.
    
    Supports loading from:
    1. Base Qwen3VL model (fresh training)
    2. Previously trained LoRA adapter (resume with pretrained weights)
    
    Args:
        args: Training arguments containing model configuration
    
    Returns:
        ThinkVLNActor model with initialized actor heads and optional LoRA adapters
    """
    from thinkvln.models.thinkvln_actor import ThinkVLNActor
    from thinkvln.models.actor_config import ThinkVLNActorConfig
    
    logger.info(f"Loading model from {args.model_name_or_path}")
    
    # Check if model_name_or_path is a LoRA adapter checkpoint
    adapter_config_path = os.path.join(args.model_name_or_path, "adapter_config.json")
    is_lora_checkpoint = os.path.exists(adapter_config_path)
    
    if is_lora_checkpoint:
        logger.info(f"Detected LoRA adapter checkpoint at {args.model_name_or_path}")
        # Load adapter config to get base model path
        import json
        with open(adapter_config_path, 'r') as f:
            adapter_config_dict = json.load(f)
        base_model_path = adapter_config_dict.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(f"adapter_config.json missing base_model_name_or_path")
        logger.info(f"Loading base model from {base_model_path}")
        actual_model_path = base_model_path
    else:
        actual_model_path = args.model_name_or_path
    
    # Create actor configuration
    actor_config = ThinkVLNActorConfig(
        num_query_tokens=args.num_query_tokens,
        num_action_classes=args.num_action_classes,
        action_loss_weight=args.action_loss_weight,
        progress_loss_weight=args.progress_loss_weight,
        use_huber_loss_for_progress=getattr(args, 'use_huber_loss_for_progress', False),
    )
    
    logger.info(f"Actor config: {actor_config}")
    
    # Prepare model loading kwargs
    model_kwargs = {
        "actor_config": actor_config,
        "dtype": torch.bfloat16 if args.bf16 else torch.float32,
        "trust_remote_code": True,
    }
    
    # Enable FlashAttention-2 if requested
    if args.use_flash_attention_2:
        logger.info("Enabling FlashAttention-2 for efficient attention computation")
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Load base model
    model = ThinkVLNActor.from_pretrained(
        actual_model_path,
        **model_kwargs
    )
    
    # If loading from LoRA checkpoint, load the adapter weights
    if is_lora_checkpoint:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter weights from {args.model_name_or_path}")
        model = PeftModel.from_pretrained(model, args.model_name_or_path)
        logger.info("LoRA adapter loaded successfully")
    
    # Freeze vision tower if requested (do this FIRST)
    if args.freeze_vision_tower:
        logger.info("Freezing vision tower")
        # Access visual encoder through base_model if it's a PEFT model
        visual_module = model.base_model.model.visual if is_lora_checkpoint else model.model.visual
        for param in visual_module.parameters():
            param.requires_grad = False
        logger.info("Vision tower frozen")
    
    # Apply LoRA if enabled (skip if already loaded from LoRA checkpoint)
    if args.use_lora and not is_lora_checkpoint:
        from peft import LoraConfig, get_peft_model
        
        logger.info("Applying LoRA configuration")
        
        # Parse target modules (only apply LoRA to language model projections)
        if args.lora_target_modules:
            target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # Keep actor-specific modules fully trainable
            modules_to_save=["query_embeddings", "shared_projector", "action_head", "progress_head"],
        )
        
        model = get_peft_model(model, lora_config)
        logger.info(f"LoRA applied with r={args.lora_r}, alpha={args.lora_alpha}")
        model.print_trainable_parameters()
    elif is_lora_checkpoint:
        logger.info("Continuing training with loaded LoRA adapter")
        model.print_trainable_parameters()
    
    # Enable gradient checkpointing AFTER applying LoRA
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        # Check if model is a PEFT model (either freshly applied or loaded from checkpoint)
        is_peft_model = args.use_lora or is_lora_checkpoint
        if is_peft_model:
            # For PEFT models, call on the base model
            model.base_model.gradient_checkpointing_enable()
            # Enable input gradients to allow backprop through frozen layers
            model.enable_input_require_grads()
            logger.info("Enabled gradient checkpointing and input gradients for PEFT model")
        else:
            model.gradient_checkpointing_enable()
            if hasattr(model, 'enable_input_require_grads'):
                model.enable_input_require_grads()
                logger.info("Enabled gradient checkpointing and input gradients")
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Debug: Print trainable parameter names
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    logger.info(f"Number of trainable parameter tensors: {len(trainable_params)}")
    if args.use_lora or is_lora_checkpoint:
        logger.info(f"First few trainable params: {trainable_params[:10]}")
    
    return model


def load_lora_checkpoint(
    base_model_path: str,
    adapter_path: str,
    actor_config=None,
    use_flash_attention_2: bool = False,
    **kwargs
):
    """
    Load LoRA checkpoint for inference. Base model + adapter (includes actor heads from modules_to_save).

    Args:
        base_model_path: Path to pretrained Qwen3VL (base model)
        adapter_path: Path to saved adapter (output_dir from training)
        actor_config: ThinkVLNActorConfig, or None to use defaults
        use_flash_attention_2: Enable FlashAttention-2
        **kwargs: Passed to from_pretrained

    Returns:
        PeftModel wrapping ThinkVLNActor with adapter loaded
    """
    from thinkvln.models.thinkvln_actor import ThinkVLNActor
    from thinkvln.models.actor_config import ThinkVLNActorConfig
    from peft import PeftModel

    actor_config = actor_config or ThinkVLNActorConfig()
    model_kwargs = {"actor_config": actor_config, "trust_remote_code": True, **kwargs}
    if use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    base_model = ThinkVLNActor.from_pretrained(base_model_path, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model


def verify_lora_save(output_dir: str, expected_modules: list = None):
    """Verify LoRA save contains adapter and modules_to_save weights."""
    import json
    import time
    
    expected_modules = expected_modules or ["query_embeddings", "shared_projector", "action_head", "progress_head"]
    adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(output_dir, "adapter_model.bin")
    config_file = os.path.join(output_dir, "adapter_config.json")

    if not os.path.exists(adapter_file):
        logger.warning(f"LoRA adapter file not found: {adapter_file}")
        return False
    if not os.path.exists(config_file):
        logger.warning(f"adapter_config.json not found in {output_dir}")
        return False
    
    # Check if config file is empty (may still be writing)
    file_size = os.path.getsize(config_file)
    if file_size == 0:
        logger.warning(f"adapter_config.json is empty, may still be writing")
        # Wait briefly and check again
        time.sleep(0.5)
        file_size = os.path.getsize(config_file)
        if file_size == 0:
            logger.warning("adapter_config.json still empty after waiting")
            return False

    try:
        with open(config_file, 'r') as f:
            content = f.read()
            if not content.strip():
                logger.warning("adapter_config.json contains no data")
                return False
            cfg = json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse adapter_config.json: {e}")
        logger.warning("LoRA adapter files exist but config may be incomplete")
        return False
    except Exception as e:
        logger.warning(f"Error reading adapter_config.json: {e}")
        return False
    
    if "base_model_name_or_path" not in cfg:
        logger.warning("adapter_config.json missing base_model_name_or_path")
    else:
        logger.info(f"Adapter base_model: {cfg['base_model_name_or_path']}")
    if "modules_to_save" in cfg:
        saved = set(cfg["modules_to_save"])
        for m in expected_modules:
            if m not in saved:
                logger.warning(f"modules_to_save missing expected: {m}")
    logger.info(f"LoRA checkpoint verified: {adapter_file}")
    return True


def create_datasets(args: ThinkVLNTrainingArguments, processor):
    """
    Create training and evaluation datasets with optional train/val split.
    
    This function creates the dataset and collator using the API defined in
    thinkvln/dataset/dataset.py. The actual data loading implementation will
    be completed in a separate task.
    
    Args:
        args: Training arguments containing data paths
        processor: Qwen3VL processor for image and text processing
    
    Returns:
        Tuple of (train_dataset, eval_dataset, data_collator)
        eval_dataset is None if val_split_ratio is 0
    """
    from thinkvln.dataset.dataset import ThinkVLNDataset, ThinkVLNDataCollator
    from torch.utils.data import random_split, Subset
    
    logger.info("Creating datasets")
    logger.info(f"Image root: {args.image_root}")
    logger.info(f"Action data: {args.action_data_path}")
    logger.info(f"CoT data: {args.cot_data_path}")
    logger.info(f"Sample ratio: {args.sample_ratio}, Val split ratio: {args.val_split_ratio}")
    
    full_dataset = ThinkVLNDataset(
        action_data_path=args.action_data_path,
        cot_data_path=args.cot_data_path,
        image_root=args.image_root,
    )
    logger.info(f"Full dataset created with {len(full_dataset)} samples")
    
    # Subsample by sample_ratio if < 1.0
    if args.sample_ratio < 1.0:
        n_total = len(full_dataset)
        n_use = max(1, int(n_total * args.sample_ratio))
        perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
        indices = perm[:n_use].tolist()
        dataset_for_split = Subset(full_dataset, indices)
        logger.info(f"Subsampled to {n_use} samples ({args.sample_ratio:.2%} of {n_total})")
    else:
        dataset_for_split = full_dataset
    
    # Split into train and validation if val_split_ratio > 0
    eval_dataset = None
    if args.val_split_ratio > 0:
        n_split = len(dataset_for_split)
        val_size = int(n_split * args.val_split_ratio)
        train_size = n_split - val_size
        train_dataset, eval_dataset = random_split(
            dataset_for_split,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        logger.info(f"Split into {len(train_dataset)} train and {len(eval_dataset)} validation samples")
    else:
        train_dataset = dataset_for_split
        logger.info("No validation split - using all data for training")
    
    # Create data collator
    data_collator = ThinkVLNDataCollator(
        processor=processor,
        num_query_tokens=args.num_query_tokens,
        image_root=args.image_root,
    )
    
    logger.info("Data collator created")
    
    return train_dataset, eval_dataset, data_collator


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def create_training_args_from_config(config: Dict[str, Any]) -> ThinkVLNTrainingArguments:
    """
    Create ThinkVLNTrainingArguments from configuration dictionary.
    
    Args:
        config: Configuration dictionary loaded from YAML
    
    Returns:
        ThinkVLNTrainingArguments instance
    """
    # Flatten nested config structure
    flat_config = {}
    
    # Model config
    if 'model' in config:
        model_cfg = config['model']
        flat_config['model_name_or_path'] = model_cfg.get('model_name_or_path', 'Qwen/Qwen3-VL-2B')
        flat_config['num_query_tokens'] = model_cfg.get('num_query_tokens', 4)
        flat_config['num_action_classes'] = model_cfg.get('num_action_classes', 4)
        flat_config['action_loss_weight'] = model_cfg.get('action_loss_weight', 1.0)
        flat_config['progress_loss_weight'] = model_cfg.get('progress_loss_weight', 1.0)
        flat_config['use_huber_loss_for_progress'] = model_cfg.get('use_huber_loss_for_progress', False)

        # LoRA config
        flat_config['use_lora'] = model_cfg.get('use_lora', False)
        flat_config['lora_r'] = model_cfg.get('lora_r', 8)
        flat_config['lora_alpha'] = model_cfg.get('lora_alpha', 16)
        flat_config['lora_dropout'] = model_cfg.get('lora_dropout', 0.05)
        if 'lora_target_modules' in model_cfg:
            target_modules = model_cfg['lora_target_modules']
            if isinstance(target_modules, list):
                flat_config['lora_target_modules'] = ','.join(target_modules)
            else:
                flat_config['lora_target_modules'] = target_modules
        
        # Vision tower config
        flat_config['freeze_vision_tower'] = model_cfg.get('freeze_vision_tower', False)
        
        # Attention optimization
        flat_config['use_flash_attention_2'] = model_cfg.get('use_flash_attention_2', False)
    
    # Data config
    if 'data' in config:
        data_cfg = config['data']
        flat_config['image_root'] = data_cfg.get('image_root', 'data')
        flat_config['action_data_path'] = data_cfg.get('action_data_path')
        flat_config['cot_data_path'] = data_cfg.get('cot_data_path')

        flat_config['val_split_ratio'] = data_cfg.get('val_split_ratio', 0.1)
        flat_config['sample_ratio'] = data_cfg.get('sample_ratio', 1.0)
    
    # Training config
    if 'training' in config:
        train_cfg = config['training']
        flat_config.update({
            'remove_unused_columns': False,
            'output_dir': train_cfg.get('output_dir', 'checkpoints/thinkvln_actor'),
            'num_train_epochs': train_cfg.get('num_train_epochs', 3),
            'max_steps': train_cfg.get('max_steps', -1),
            'per_device_train_batch_size': train_cfg.get('per_device_train_batch_size', 2),
            'gradient_accumulation_steps': train_cfg.get('gradient_accumulation_steps', 8),
            'learning_rate': train_cfg.get('learning_rate', 2e-5),
            'weight_decay': train_cfg.get('weight_decay', 0.01),
            'warmup_steps': train_cfg.get('warmup_steps', 500),
            'max_grad_norm': train_cfg.get('max_grad_norm', 1.0),
            'lr_scheduler_type': train_cfg.get('lr_scheduler_type', 'cosine'),
            'bf16': train_cfg.get('bf16', True),
            'fp16': train_cfg.get('fp16', False),
            'gradient_checkpointing': train_cfg.get('gradient_checkpointing', True),
            'resume_from_checkpoint': train_cfg.get('resume_from_checkpoint'),
            'logging_steps': train_cfg.get('logging_steps', 10),
            'logging_first_step': train_cfg.get('logging_first_step', True),
            'save_steps': train_cfg.get('save_steps', 1000),
            'save_total_limit': train_cfg.get('save_total_limit', 3),
            'eval_steps': train_cfg.get('eval_steps', 1000),
            'deepspeed': train_cfg.get('deepspeed'),
            'dataloader_num_workers': train_cfg.get('dataloader_num_workers', 4),
            'dataloader_pin_memory': train_cfg.get('dataloader_pin_memory', True),
            'seed': train_cfg.get('seed', 42),
            # Distributed training
            'ddp_find_unused_parameters': train_cfg.get('ddp_find_unused_parameters', True),  # Required for HomogeneousBatchSampler
            'ddp_backend': train_cfg.get('ddp_backend', 'nccl'),
        })
    
    # Logging config (WandB)
    if 'logging' in config:
        log_cfg = config['logging']
        flat_config.update({
            'report_to': log_cfg.get('report_to', ['wandb']),
            'run_name': log_cfg.get('run_name'),
            'logging_dir': log_cfg.get('logging_dir'),
        })
    
    # Create TrainingArguments instance
    args = ThinkVLNTrainingArguments(**flat_config)
    return args


def main():
    """
    Main training script entry point.
    
    Steps:
        1. Parse config file path from command line
        2. Load configuration from YAML
        3. Create training arguments
        4. Load Qwen3VL processor
        5. Load ThinkVLNActor model
        6. Create datasets and collator
        7. Initialize trainer
        8. Run training
        9. Save final model
    """
    # Parse config file path
    parser = argparse.ArgumentParser(description="ThinkVLN SFT Trainer")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file (e.g., config/sft_training.yaml)'
    )
    cmd_args = parser.parse_args()
    
    # Load configuration from YAML
    config = load_config_from_yaml(cmd_args.config)
    
    # Create training arguments
    args = create_training_args_from_config(config)
    
    logger.info("=" * 80)
    logger.info("ThinkVLN SFT Trainer")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Number of epochs: {args.num_train_epochs}")
    logger.info(f"Batch size per device: {args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Mixed precision: {'bf16' if args.bf16 else 'fp16' if args.fp16 else 'fp32'}")
    logger.info(f"DeepSpeed: {args.deepspeed if args.deepspeed else 'Disabled'}")
    logger.info(f"Distributed: {args.world_size} GPUs")
    logger.info(f"Logging: {args.report_to}")
    logger.info("=" * 80)
    
    # Load processor
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    logger.info("Processor loaded successfully")
    
    # Load model
    model = load_model(args)
    
    # Create datasets
    train_dataset, eval_dataset, data_collator = create_datasets(args, processor)
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = ThinkVLNSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    logger.info("Trainer initialized")
    
    # Train
    logger.info("Starting training...")
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    logger.info("=" * 80)
    
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    logger.info(f"Training steps: {train_result.global_step}")
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    if args.use_lora:
        logger.info("Saving LoRA adapter weights (includes actor heads from modules_to_save)")
        model.save_pretrained(args.output_dir)
        # Only verify on main process to avoid race conditions
        if args.should_save:
            verify_lora_save(args.output_dir)
    else:
        # For full fine-tuning, save the complete model
        trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # Save training metrics
    metrics_path = os.path.join(args.output_dir, "training_metrics.pt")
    torch.save({
        "train_loss": train_result.training_loss,
        "global_step": train_result.global_step,
        "loss_history": trainer.loss_history,
    }, metrics_path)
    logger.info(f"Saved training metrics to {metrics_path}")
    
    logger.info("=" * 80)
    logger.info("All done!")
    logger.info("=" * 80)
    
    # Clean up distributed training
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise
