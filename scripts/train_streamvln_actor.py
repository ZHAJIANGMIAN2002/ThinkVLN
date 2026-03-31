#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StreamVLN actor from a YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_runtime_env(config: dict) -> dict:
    env = {"TOKENIZERS_PARALLELISM": "false"}
    runtime_env = (config.get("runtime") or {}).get("env") or {}
    env.update({str(k): str(v) for k, v in runtime_env.items() if v is not None})

    log_cfg = config.get("logging") or {}
    wandb_cfg = log_cfg.get("wandb") or {}
    if wandb_cfg.get("project"):
        env["WANDB_PROJECT"] = str(wandb_cfg["project"])
    if wandb_cfg.get("entity"):
        env["WANDB_ENTITY"] = str(wandb_cfg["entity"])
    if wandb_cfg.get("mode"):
        env["WANDB_MODE"] = str(wandb_cfg["mode"])
    tags = wandb_cfg.get("tags") or []
    if tags:
        env["WANDB_TAGS"] = ",".join(str(tag) for tag in tags)
    if log_cfg.get("run_name"):
        env["WANDB_NAME"] = str(log_cfg["run_name"])
    return env


def apply_runtime_env(env: dict) -> None:
    for key, value in env.items():
        os.environ[str(key)] = str(value)


def build_subset_jsonl(src_path: Path, max_samples: int) -> Path:
    fd, temp_path = tempfile.mkstemp(prefix="streamvln_actor_train_", suffix=".jsonl")
    os.close(fd)
    out_path = Path(temp_path)
    with src_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx >= max_samples:
                break
            fout.write(line)
    return out_path


def materialize_summary_data_path(data_cfg: dict) -> Tuple[str, Optional[str]]:
    summary_path = str(data_cfg.get("summary_data_path") or data_cfg.get("data_path") or "").strip()
    if not summary_path:
        raise ValueError("Config must provide data.summary_data_path or data.data_path")
    max_samples = data_cfg.get("max_samples")
    if max_samples in (None, 0, False):
        return summary_path, None
    subset_path = build_subset_jsonl(Path(summary_path), max(1, int(max_samples)))
    return str(subset_path), str(subset_path)


def _append_arg(argv: List[str], flag: str, value) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        argv.extend([flag, "True" if value else "False"])
        return
    if isinstance(value, (list, tuple)):
        if not value:
            return
        argv.append(flag)
        argv.extend(str(item) for item in value)
        return
    argv.extend([flag, str(value)])


def _load_pretrained_model_defaults(model_name_or_path: Optional[str]) -> dict:
    model_path = Path(str(model_name_or_path or "").strip())
    if not model_path:
        return {}
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle) or {}
    except (OSError, json.JSONDecodeError):
        return {}
    defaults = {}
    vision_tower = config.get("vision_tower") or config.get("mm_vision_tower")
    if vision_tower:
        defaults["vision_tower"] = vision_tower
    return defaults


def build_streamvln_train_argv(config: dict, data_path_override: Optional[str] = None) -> List[str]:
    model_cfg = dict(config.get("model") or {})
    data_cfg = config.get("data") or {}
    training_cfg = config.get("training") or {}
    logging_cfg = config.get("logging") or {}
    pretrained_defaults = _load_pretrained_model_defaults(model_cfg.get("model_name_or_path"))
    for key, value in pretrained_defaults.items():
        model_cfg.setdefault(key, value)

    argv: List[str] = []

    model_key_map = {
        "model_name_or_path": "model_name_or_path",
        "model_type": "model_type",
        "progress_loss_weight": "progress_loss_weight",
        "done_loss_weight": "done_loss_weight",
        "rope_scaling_factor": "rope_scaling_factor",
        "rope_scaling_type": "rope_scaling_type",
        "use_pos_skipping": "use_pos_skipping",
        "pos_skipping_range": "pos_skipping_range",
        "mm_tunable_parts": "mm_tunable_parts",
        "mm_newline_position": "mm_newline_position",
        "mm_patch_merge_type": "mm_patch_merge_type",
        "vision_tower": "vision_tower",
        "mm_spatial_pool_stride": "mm_spatial_pool_stride",
        "mm_spatial_pool_out_channels": "mm_spatial_pool_out_channels",
        "mm_spatial_pool_mode": "mm_spatial_pool_mode",
        "mm_spatial_pool_size": "mm_spatial_pool_size",
        "mm_resampler_type": "mm_resampler_type",
    }
    model_alias_map = {
        "use_lora": "lora_enable",
        "lora_enable": "lora_enable",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "lora_target_modules": "lora_target_modules",
        "lora_bias": "lora_bias",
        "bits": "bits",
    }
    data_key_map = {
        "summary_data_path": "summary_data_path",
        "data_path": "data_path",
        "image_root": "image_root",
        "image_folder": "image_folder",
        "watcher_memory_path": "watcher_memory_path",
        "watcher_memory_ratio": "watcher_memory_ratio",
        "watcher_memory_seed": "watcher_memory_seed",
        "done_threshold": "done_threshold",
        "num_history": "num_history",
        "num_future_steps": "num_future_steps",
        "multi_task_training": "multi_task_training",
    }
    training_key_map = {
        "output_dir": "output_dir",
        "num_train_epochs": "num_train_epochs",
        "max_steps": "max_steps",
        "per_device_train_batch_size": "per_device_train_batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "warmup_steps": "warmup_steps",
        "warmup_ratio": "warmup_ratio",
        "max_grad_norm": "max_grad_norm",
        "lr_scheduler_type": "lr_scheduler_type",
        "bf16": "bf16",
        "fp16": "fp16",
        "tf32": "tf32",
        "gradient_checkpointing": "gradient_checkpointing",
        "resume_from_checkpoint": "resume_from_checkpoint",
        "logging_steps": "logging_steps",
        "logging_first_step": "logging_first_step",
        "save_steps": "save_steps",
        "save_total_limit": "save_total_limit",
        "dataloader_num_workers": "dataloader_num_workers",
        "dataloader_pin_memory": "dataloader_pin_memory",
        "seed": "seed",
        "deepspeed": "deepspeed",
        "ddp_find_unused_parameters": "ddp_find_unused_parameters",
        "ddp_backend": "ddp_backend",
        "remove_unused_columns": "remove_unused_columns",
        "attn_implementation": "attn_implementation",
    }

    for source_key, arg_key in model_key_map.items():
        _append_arg(argv, f"--{arg_key}", model_cfg.get(source_key))
    for source_key, arg_key in model_alias_map.items():
        value = model_cfg.get(source_key)
        if source_key == "lora_target_modules" and isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value if str(item).strip())
        _append_arg(argv, f"--{arg_key}", value)

    effective_summary_path = data_path_override or data_cfg.get("summary_data_path") or data_cfg.get("data_path")
    if effective_summary_path:
        _append_arg(argv, "--summary_data_path", effective_summary_path)
        _append_arg(argv, "--data_path", effective_summary_path)
    for source_key, arg_key in data_key_map.items():
        if source_key in {"summary_data_path", "data_path"}:
            continue
        _append_arg(argv, f"--{arg_key}", data_cfg.get(source_key))

    for source_key, arg_key in training_key_map.items():
        _append_arg(argv, f"--{arg_key}", training_cfg.get(source_key))

    report_to = logging_cfg.get("report_to")
    if report_to is not None:
        _append_arg(argv, "--report_to", report_to)
    _append_arg(argv, "--run_name", logging_cfg.get("run_name"))
    _append_arg(argv, "--logging_dir", logging_cfg.get("logging_dir"))
    return argv


def run_streamvln_training(train_argv: List[str]) -> None:
    old_argv = list(sys.argv)
    try:
        sys.argv = ["streamvln_train.py", *train_argv]
        from streamvln import streamvln_train

        streamvln_train.train()
    finally:
        sys.argv = old_argv


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env = build_runtime_env(config)
    apply_runtime_env(env)

    cleanup_path = None
    try:
        summary_path, cleanup_path = materialize_summary_data_path(config.get("data") or {})
        train_argv = build_streamvln_train_argv(config, data_path_override=summary_path)
        print(f"config: {args.config}")
        print(f"summary_data_path: {summary_path}")
        print(f"train_argv: {' '.join(train_argv)}")
        run_streamvln_training(train_argv)
    finally:
        if cleanup_path and Path(cleanup_path).exists():
            Path(cleanup_path).unlink()


if __name__ == "__main__":
    main()
