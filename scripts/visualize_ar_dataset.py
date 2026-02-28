#!/usr/bin/env python3
"""Visualize AR VLN dataset: image + prompt + target (labels) per sample, export to PNGs and 1 fps video."""

import argparse
import os
import random
import sys
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PIL import Image, ImageDraw, ImageFont

from thinkvln.dataset.ar_dataset import ARVLNDataset, ARVLNDataCollator
from thinkvln.tools.dataset_utils import load_image, extract_action_chunk


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_image_path(image_root: str, episode_key: str, frame_idx: int) -> str:
    parts = episode_key.split("_")
    if len(parts) == 2:
        scene_id, episode_id = parts[0], parts[1]
        dir_episode_key = f"{scene_id}_r2r_{int(episode_id):06d}"
    else:
        dir_episode_key = episode_key
    return os.path.join(image_root, dir_episode_key, f"{frame_idx:06d}_rgb.jpg")


def wrap_text(draw, text, font, max_width):
    lines = []
    for line in text.split("\n"):
        words = line.split()
        current = ""
        for w in words:
            test = (current + " " + w).strip() if current else w
            try:
                ww = draw.textlength(test, font=font)
            except Exception:
                ww = len(test) * 8
            if ww > max_width and current:
                lines.append(current)
                current = w
            else:
                current = test
        if current:
            lines.append(current)
    return lines


def draw_sample(image, prompt, target_text, out_path, text_width=420, font_size=14, max_image_h=480):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    iw, ih = image.size
    if ih > max_image_h:
        image = image.resize((int(iw * max_image_h / ih), max_image_h), Image.Resampling.LANCZOS)
        iw, ih = image.size
    pad = 10
    canvas_w = iw + text_width + pad * 2
    canvas_h = ih
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    x0 = iw + pad
    y = pad
    line_h = int(font_size * 1.3)
    max_tw = text_width - pad
    draw.text((x0, y), "Prompt:", fill=(0, 0, 0), font=font)
    y += line_h
    prompt_lines = wrap_text(draw, prompt, font, max_tw)
    for line in prompt_lines[:12]:
        draw.text((x0, y), (line[:120] + "..") if len(line) > 120 else line, fill=(0, 0, 0), font=font)
        y += line_h
    if len(prompt_lines) > 12:
        draw.text((x0, y), "...", fill=(0, 0, 0), font=font)
        y += line_h
    y += 4
    draw.text((x0, y), "Target (labels):", fill=(0, 80, 0), font=font)
    y += line_h
    target_lines = wrap_text(draw, target_text, font, max_tw)
    for line in target_lines[:8]:
        draw.text((x0, y), (line[:120] + "..") if len(line) > 120 else line, fill=(0, 80, 0), font=font)
        y += line_h
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize AR VLN dataset samples.")
    parser.add_argument("--config", type=str, default="config/ar_training.yaml", help="Path to ar_training.yaml")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--out_dir", type=str, default="ar_dataset_vis", help="Output dir for PNGs and video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--fps", type=float, default=1.0, help="Video FPS")
    args = parser.parse_args()
    if not os.path.isabs(args.config):
        candidate = os.path.join(REPO_ROOT, args.config)
        args.config = candidate if os.path.exists(candidate) else os.path.join(os.getcwd(), args.config)
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(os.getcwd(), args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    image_root = data_cfg.get("image_root", "/mnt/nvme/swx/dataset/R2R")
    action_data_path = data_cfg.get("action_data_path")
    num_future_steps = int(data_cfg.get("num_future_steps", 4))
    model_name_or_path = model_cfg.get("model_name_or_path", "Qwen/Qwen3-VL-2B")

    if not action_data_path or not os.path.exists(action_data_path):
        print(f"Action data not found: {action_data_path}")
        sys.exit(1)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    progress_tokens = [f"<p_{i * 5}>" for i in range(0, 101 // 5 + 1)]
    processor.tokenizer.add_tokens(progress_tokens, special_tokens=True)

    collator = ARVLNDataCollator(
        processor=processor,
        image_root=image_root,
        num_future_steps=num_future_steps,
        progress_bin_step=5,
    )
    dataset = ARVLNDataset(
        action_data_path=action_data_path,
        image_root=image_root,
        num_future_steps=num_future_steps,
    )

    n_total = len(dataset)
    n = min(args.num_samples, n_total)
    random.seed(args.seed)
    indices = random.sample(range(n_total), n)
    print(f"Visualizing {n} AR samples -> {args.out_dir}")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image_path = build_image_path(image_root, sample["episode_key"], sample["frame_idx"])
        if not os.path.exists(image_path):
            print(f"Skip {idx}: image not found {image_path}")
            continue
        image = load_image(image_path)
        prompt = collator.prompt_template.format(subgoal=sample["current_plan_step"])
        actions, progresses = extract_action_chunk(
            sample["frame_idx"],
            sample["actions"],
            sample["subtask_sequence"],
            num_steps=num_future_steps,
        )
        target_text = collator._build_target_text(actions, progresses)
        out_path = os.path.join(args.out_dir, f"ar_vis_{i:03d}.png")
        draw_sample(image, prompt, target_text, out_path)

    import cv2
    video_path = os.path.join(args.out_dir, "ar_dataset_preview.mp4")
    frames = [os.path.join(args.out_dir, f"ar_vis_{i:03d}.png") for i in range(n)]
    first = cv2.imread(frames[0])
    if first is not None:
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))
        for p in frames:
            if os.path.exists(p):
                frame = cv2.imread(p)
                if frame is not None:
                    writer.write(frame)
        writer.release()
        print(f"Video saved: {video_path}")
    else:
        print("No frames to write video.")


if __name__ == "__main__":
    main()
