#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from streamvln.utils.hf_local import apply_local_only_env, prepare_local_only_pretrained_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-batch StreamVLN actor smoke check.")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--do_backward", action="store_true")
    return parser.parse_args()


def build_subset_jsonl(src_path: Path, max_samples: int) -> Path:
    fd, temp_path = tempfile.mkstemp(prefix="streamvln_actor_smoke_", suffix=".jsonl")
    os.close(fd)
    out_path = Path(temp_path)
    with src_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if idx >= max_samples:
                break
            fout.write(line)
    return out_path


def move_batch_to_device(batch: dict, device) -> dict:
    import torch

    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def format_optional_loss(value) -> str:
    if value is None:
        return "None"
    return f"{value.detach().item():.6f}"


def get_model_float_dtype(model):
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return None


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import transformers
    from streamvln.streamvln_train import get_model, make_supervised_data_module

    env_updates, local_pretrained_kwargs = prepare_local_only_pretrained_kwargs(str(args.model_path))
    apply_local_only_env(env_updates)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    subset_path = build_subset_jsonl(args.data_path, max(1, int(args.max_samples)))
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(args.model_path),
            model_max_length=4096,
            padding_side="right",
            **local_pretrained_kwargs,
        )

        data_args = SimpleNamespace(
            image_root="",
            image_folder="",
            summary_data_path=str(subset_path),
            data_path=str(subset_path),
            watcher_memory_path=None,
            watcher_memory_ratio=1.0,
            watcher_memory_seed=42,
            done_threshold=0.85,
            num_history=int(args.num_history),
            num_future_steps=int(args.num_future_steps),
            multi_task_training=False,
        )
        model_args = SimpleNamespace(
            model_name_or_path=str(args.model_path),
            model_type="streamvln_actor",
            rope_scaling_factor=None,
            rope_scaling_type=None,
            mm_spatial_pool_stride=None,
            mm_spatial_pool_out_channels=None,
            mm_spatial_pool_mode=None,
            mm_resampler_type=None,
            use_pos_skipping=False,
            pos_skipping_range=4096,
            mm_spatial_pool_size=None,
            progress_loss_weight=1.0,
            done_loss_weight=1.0,
            mm_tunable_parts=None,
            mm_newline_position="grid",
            mm_patch_merge_type="flat",
        )
        training_args = SimpleNamespace(
            attn_implementation=str(args.attn_implementation),
            cache_dir=None,
            bf16=args.device.startswith("cuda"),
        )

        module = make_supervised_data_module(tokenizer, None, data_args, model_args)
        dataset = module["train_dataset"]
        collator = module["data_collator"]
        take = min(max(1, int(args.batch_size)), len(dataset))
        batch = collator([dataset[i] for i in range(take)])

        device = torch.device(args.device)
        model = get_model(model_args, training_args, data_args, {})
        model.model.num_history = int(args.num_history)
        model.to(device)
        model.train()

        batch = move_batch_to_device(batch, device)
        model_dtype = get_model_float_dtype(model)
        if device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}:
            with torch.autocast(device_type="cuda", dtype=model_dtype):
                outputs = model(**batch, return_dict=True)
        else:
            outputs = model(**batch, return_dict=True)

        print(f"subset_path: {subset_path}")
        print(f"batch_input_ids: {tuple(batch['input_ids'].shape)}")
        print(f"batch_images: {tuple(batch['images'].shape)}")
        print(f"loss: {outputs.loss.detach().item():.6f}")
        print(f"progress_loss: {format_optional_loss(outputs.progress_loss)}")
        print(f"done_loss: {format_optional_loss(outputs.done_loss)}")

        if args.do_backward:
            outputs.loss.backward()
            print("backward: ok")
    finally:
        if subset_path.exists():
            subset_path.unlink()


if __name__ == "__main__":
    main()
