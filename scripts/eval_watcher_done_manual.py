#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from thinkvln.dataset.watcher_sft_dataset import WatcherSFTDataset, build_rollout_prompt
from thinkvln.datagen.generation.watcher_utils import extract_json_object
from thinkvln.tools.dataset_utils import load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate watcher done accuracy on manual labels.")
    parser.add_argument("--manifest_file", type=Path, required=True)
    parser.add_argument("--annotation_file", type=Path, required=True)
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--summary_full_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--base_model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--image_stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = WatcherSFTDataset(
        manifest_file=str(args.manifest_file),
        annotation_file=str(args.annotation_file),
        bundle_root=str(args.bundle_root),
        summary_full_path=str(args.summary_full_path),
        image_stride=max(1, int(args.image_stride)),
        sample_ratio=1.0,
        seed=int(args.seed),
    )

    pos_idx = [i for i, s in enumerate(dataset.samples) if bool(s.get("done", False))]
    neg_idx = [i for i, s in enumerate(dataset.samples) if not bool(s.get("done", False))]
    rng = random.Random(int(args.seed))
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    sample_size = max(1, int(args.sample_size))
    half = sample_size // 2
    pick_pos = pos_idx[: min(len(pos_idx), half)]
    pick_neg = neg_idx[: min(len(neg_idx), sample_size - len(pick_pos))]
    remaining = sample_size - len(pick_pos) - len(pick_neg)
    extra: list[int] = []
    if remaining > 0:
        pool = pos_idx[len(pick_pos) :] + neg_idx[len(pick_neg) :]
        rng.shuffle(pool)
        extra = pool[:remaining]
    indices = pick_pos + pick_neg + extra
    rng.shuffle(indices)

    sampled_file = args.output_dir / "manual_done_testset.sampled.jsonl"
    with sampled_file.open("w", encoding="utf-8") as handle:
        for i in indices:
            sample = dataset.samples[i]
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "episode_key": sample["episode_key"],
                        "done": bool(sample["done"]),
                        "memory_start": sample["memory_start"],
                        "active_step": sample["active_step"],
                        "pending_steps": sample["pending_steps"],
                        "rollout_actions": sample["rollout_actions"],
                        "rollout_image_paths": sample["rollout_image_paths"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(str(args.base_model_path), trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.base_model_path),
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.model_path))
    model = model.to(device)
    model.eval()

    pred_file = args.output_dir / "predictions.jsonl"
    correct = 0
    valid = 0
    parse_fail = 0

    with pred_file.open("w", encoding="utf-8") as fout:
        for n, idx in enumerate(indices, start=1):
            sample = dataset.samples[idx]
            rollout_images = [load_image(path) for path in sample.get("rollout_image_paths", [])]
            prompt = build_rollout_prompt(
                instruction=str(sample.get("instruction", "")),
                plan_steps=sample.get("plan_steps", []),
                done_steps=sample.get("done_steps", []),
                active_step=str(sample.get("active_step", "")),
                pending_steps=sample.get("pending_steps", []),
                memory_start=str(sample.get("memory_start", "")),
                rollout_actions=sample.get("rollout_actions", []),
                rollout_images=rollout_images,
            )
            content = [{"type": "text", "text": prompt.user_text}]
            content.extend({"type": "image", "image": image} for image in prompt.images)
            messages = [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": content},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[prompt.images], padding=True, return_tensors="pt")
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=int(args.max_new_tokens), do_sample=False)

            in_len = inputs["input_ids"].shape[1]
            raw = processor.batch_decode(
                gen_ids[:, in_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            gt = bool(sample.get("done", False))
            pred_done = None
            error = None
            try:
                payload = extract_json_object(raw)
                pred_done = bool(payload.get("done", False))
                valid += 1
                if pred_done == gt:
                    correct += 1
            except Exception as exc:
                payload = {}
                parse_fail += 1
                error = str(exc)

            fout.write(
                json.dumps(
                    {
                        "sample_id": sample["sample_id"],
                        "episode_key": sample["episode_key"],
                        "gt_done": gt,
                        "pred_done": pred_done,
                        "correct": (pred_done == gt) if pred_done is not None else False,
                        "parse_ok": pred_done is not None,
                        "error": error,
                        "raw_text": raw,
                        "json_obj": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if n % 10 == 0:
                print(
                    f"[watcher-eval] {n}/{len(indices)} "
                    f"valid={valid} acc={(correct / max(valid, 1)):.4f} parse_fail={parse_fail}"
                )

    summary = {
        "checkpoint": str(args.model_path),
        "base_model": str(args.base_model_path),
        "sample_size": len(indices),
        "positive_samples": sum(1 for i in indices if bool(dataset.samples[i].get("done", False))),
        "negative_samples": sum(1 for i in indices if not bool(dataset.samples[i].get("done", False))),
        "valid_predictions": valid,
        "parse_failures": parse_fail,
        "done_accuracy_on_valid": (correct / valid) if valid else 0.0,
        "done_accuracy_treat_parse_fail_as_wrong": (correct / len(indices)) if indices else 0.0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[watcher-eval] summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
