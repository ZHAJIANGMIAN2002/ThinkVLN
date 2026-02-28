#!/usr/bin/env python3
"""
Randomly keep a fraction of trajectory data and update summary.json only.
Supports R2R_back and ScaleVLN. Structure:
  - R2R_back: summary.json + r2r/ (per-frame rgb/map) + images/ (trajectory video dirs)
  - ScaleVLN_back: summary.json + images/ (trajectory video dirs)  [no scalevln/ here]
  - scalevln_back: scalevln/ (per-frame rgb/map only; summary lives in ScaleVLN_back)
- Deletes per-episode dirs and rewrites summary.json. Does NOT write summary_full.jsonl.

Usage:
  python scripts/thin_r2r_trajectory_data.py [--base ...] [--dataset r2r|scalevln] [--keep-ratio 0.5] [--dry-run]
  python scripts/thin_r2r_trajectory_data.py --all [--keep-ratio 0.5] [--dry-run]
"""

import argparse
import json
import os
import random
import shutil
from typing import Optional

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "trajectory_data")
DATASET_CONFIG = {
    "r2r": {"frames_subdir": "r2r", "tag": "r2r"},
    "scalevln": {"frames_subdir": "scalevln", "tag": "scalevln"},
}


def thin_one(
    base: str,
    dataset: str,
    keep_ratio: float,
    seed: int,
    dry_run: bool,
    frames_base: Optional[str] = None,
) -> None:
    base = os.path.abspath(base)
    cfg = DATASET_CONFIG.get(dataset)
    if not cfg:
        raise SystemExit(f"Unknown dataset: {dataset}. Choose from: {list(DATASET_CONFIG)}")
    frames_subdir = cfg["frames_subdir"]
    tag = cfg["tag"]

    summary_path = os.path.join(base, "summary.json")
    # ScaleVLN: summary+images in ScaleVLN_back; frames in scalevln_back/scalevln. Pass frames_base for that.
    if frames_base:
        frames_dir = os.path.join(os.path.abspath(frames_base), frames_subdir)
    else:
        frames_dir = os.path.join(base, frames_subdir)
    images_dir = os.path.join(base, "images")

    if not os.path.isfile(summary_path):
        print(f"Skip {base}: no summary.json")
        return

    # Load all entries
    entries = []
    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    n_total = len(entries)
    random.seed(seed)
    indices = list(range(n_total))
    random.shuffle(indices)
    n_keep = max(1, int(n_total * keep_ratio))
    keep_indices = set(indices[:n_keep])
    kept_entries = [e for i, e in enumerate(entries) if i in keep_indices]
    to_remove = [e for i, e in enumerate(entries) if i not in keep_indices]

    print(f"[{dataset}] {base}: total={n_total}, keeping={len(kept_entries)}, removing={len(to_remove)}")

    for e in to_remove:
        scene_id = e["scene_id"]
        episode_id = e["id"]
        dir_name = f"{scene_id}_{tag}_{episode_id:06d}"
        frame_ep_dir = os.path.join(frames_dir, dir_name)
        image_ep_dir = os.path.join(images_dir, dir_name)
        if dry_run:
            if os.path.isdir(frame_ep_dir):
                print(f"  [dry-run] would remove: {frame_ep_dir}")
            if os.path.isdir(image_ep_dir):
                print(f"  [dry-run] would remove: {image_ep_dir}")
        else:
            if os.path.isdir(frame_ep_dir):
                shutil.rmtree(frame_ep_dir)
            if os.path.isdir(image_ep_dir):
                shutil.rmtree(image_ep_dir)

    if dry_run:
        print(f"  [dry-run] Would rewrite {summary_path} with {len(kept_entries)} entries.")
        return

    with open(summary_path, "w") as f:
        for e in kept_entries:
            f.write(json.dumps(e) + "\n")
    print(f"  Wrote {summary_path} with {len(kept_entries)} entries.")


def main():
    parser = argparse.ArgumentParser(description="Thin trajectory data (R2R and/or ScaleVLN) to a fraction.")
    parser.add_argument(
        "--base",
        type=str,
        default=os.path.join(DATA_ROOT, "R2R_back"),
        help="Base directory (summary.json, <frames_subdir>/, images/). Ignored if --all.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG),
        default="r2r",
        help="Dataset type: r2r (R2R_back) or scalevln (ScaleVLN_back). Ignored if --all.",
    )
    parser.add_argument(
        "--scalevln-frames-base",
        type=str,
        default=None,
        help="For dataset=scalevln: dir containing scalevln/ (per-frame rgb/map). Default: <DATA_ROOT>/scalevln_back. Ignored if --all.",
    )
    parser.add_argument("--keep-ratio", type=float, default=0.5, help="Fraction of episodes to keep (default 0.5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be deleted, do not delete or rewrite.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Thin both R2R_back and ScaleVLN_back (same keep_ratio and seed).",
    )
    args = parser.parse_args()

    if args.all:
        thin_one(os.path.join(DATA_ROOT, "R2R_back"), "r2r", args.keep_ratio, args.seed, args.dry_run)
        # ScaleVLN: summary+images in ScaleVLN_back; per-frame dirs in scalevln_back/scalevln
        thin_one(
            os.path.join(DATA_ROOT, "ScaleVLN_back"),
            "scalevln",
            args.keep_ratio,
            args.seed,
            args.dry_run,
            frames_base=os.path.join(DATA_ROOT, "scalevln_back"),
        )
    else:
        frames_base = None
        if args.dataset == "scalevln":
            frames_base = args.scalevln_frames_base or os.path.join(DATA_ROOT, "scalevln_back")
        thin_one(
            args.base,
            args.dataset,
            args.keep_ratio,
            args.seed,
            args.dry_run,
            frames_base=frames_base,
        )


if __name__ == "__main__":
    main()
