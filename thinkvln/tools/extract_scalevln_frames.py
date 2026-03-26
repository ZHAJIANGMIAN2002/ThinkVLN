#!/usr/bin/env python3
"""Extract frames from ScaleVLN trajectory mp4 files into individual RGB jpgs.

ScaleVLN stores trajectories as mp4 videos in:
  ScaleVLN_back/images/{scene}_scalevln_{episode:06d}/trajectory.mp4

This script extracts frames to match the R2R format expected by ThinkVLNDataset:
  ScaleVLN_back/images/{scene}_scalevln_{episode:06d}/{frame:06d}_rgb.jpg

Usage:
  python thinkvln/tools/extract_scalevln_frames.py
  python thinkvln/tools/extract_scalevln_frames.py --images_dir /path/to/ScaleVLN_back/images --workers 8
"""

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def extract_frames(video_path: str, out_dir: str, num_frames: int) -> bool:
    """Extract num_frames evenly-spaced frames from video_path into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    # Use ffmpeg to extract exactly num_frames frames
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select=not(mod(n\\,1)),setpts=N/FRAME_RATE/TB",
        "-frames:v", str(num_frames),
        "-q:v", "2",
        os.path.join(out_dir, "%06d_rgb.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def already_extracted(out_dir: str, num_frames: int) -> bool:
    if not os.path.isdir(out_dir):
        return False
    jpgs = [f for f in os.listdir(out_dir) if f.endswith("_rgb.jpg")]
    return len(jpgs) >= num_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        default="/mnt/swx/ThinkVLN/data/trajectory_data/ScaleVLN_back/images",
    )
    parser.add_argument(
        "--summary_path",
        default="/mnt/swx/ThinkVLN/data/trajectory_data/ScaleVLN_back/summary_full_fixed.jsonl",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Build episode -> num_frames map
    ep_frames = {}
    with open(args.summary_path) as f:
        for line in f:
            d = json.loads(line)
            subdir = d.get("video", "").split("/")[-1]
            if subdir:
                ep_frames[subdir] = d["num_frames"]

    tasks = []
    for ep_dir, num_frames in ep_frames.items():
        video_path = os.path.join(args.images_dir, ep_dir, "trajectory.mp4")
        out_dir = os.path.join(args.images_dir, ep_dir)
        if not os.path.isfile(video_path):
            continue
        if already_extracted(out_dir, num_frames):
            continue
        tasks.append((video_path, out_dir, num_frames))

    print(f"Episodes to extract: {len(tasks)} / {len(ep_frames)}")
    if args.dry_run:
        return

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extract_frames, v, o, n): (v, n) for v, o, n in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            v, n = futs[fut]
            ok = fut.result()
            if ok:
                done += 1
            else:
                failed += 1
                print(f"  FAILED: {v}")
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(tasks)} (done={done} failed={failed})")

    print(f"Done. Extracted={done}, Failed={failed}")


if __name__ == "__main__":
    main()
