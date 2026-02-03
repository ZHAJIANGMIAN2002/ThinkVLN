#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


def list_episode_dirs(source_dir: str) -> list:
    names = []
    for name in os.listdir(source_dir):
        path = os.path.join(source_dir, name)
        if os.path.isdir(path):
            names.append(name)
    return names


def build_mp4_from_dir(episode_dir: str, fps: int = 6) -> None:
    rgb_files = [f for f in os.listdir(episode_dir) if f.endswith("_rgb.jpg")]
    if not rgb_files:
        return

    indices = []
    for name in rgb_files:
        try:
            indices.append(int(name.split("_")[0]))
        except ValueError:
            continue
    indices = sorted(set(indices))
    if not indices:
        return

    first_rgb = cv2.imread(os.path.join(episode_dir, f"{indices[0]:06d}_rgb.jpg"))
    first_map = cv2.imread(os.path.join(episode_dir, f"{indices[0]:06d}_map.jpg"))
    if first_rgb is None or first_map is None:
        return

    rgb_h, rgb_w = first_rgb.shape[:2]
    map_h, map_w = first_map.shape[:2]
    if map_h != rgb_h:
        new_w = int(map_w * (rgb_h / map_h))
        first_map = cv2.resize(first_map, (new_w, rgb_h), interpolation=cv2.INTER_AREA)
        map_h, map_w = first_map.shape[:2]

    video_w = rgb_w + map_w
    video_h = rgb_h
    video_path = os.path.join(episode_dir, "trajectory.mp4")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_w, video_h),
    )

    for idx in indices:
        rgb_path = os.path.join(episode_dir, f"{idx:06d}_rgb.jpg")
        map_path = os.path.join(episode_dir, f"{idx:06d}_map.jpg")
        rgb = cv2.imread(rgb_path)
        map_img = cv2.imread(map_path)
        if rgb is None or map_img is None:
            continue
        if map_img.shape[0] != rgb_h:
            new_w = int(map_img.shape[1] * (rgb_h / map_img.shape[0]))
            map_img = cv2.resize(map_img, (new_w, rgb_h), interpolation=cv2.INTER_AREA)
        frame = np.concatenate([rgb, map_img], axis=1)
        if frame.shape[1] != video_w or frame.shape[0] != video_h:
            frame = cv2.resize(frame, (video_w, video_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()


def copy_and_build(
    source_dir: str,
    dest_dir: str,
    episode_name: str,
    fps: int,
) -> str:
    src = os.path.join(source_dir, episode_name)
    dst = os.path.join(dest_dir, episode_name)
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
    build_mp4_from_dir(dst, fps=fps)
    return episode_name


def zip_directory(source_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for name in files:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, source_dir)
                zf.write(path, rel)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter ScaleVLN episodes, build mp4, zip.")
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--count", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--zip_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.dest_dir, exist_ok=True)

    episodes = list_episode_dirs(args.source_dir)
    if len(episodes) < args.count:
        raise ValueError(f"Not enough episodes: found {len(episodes)}, need {args.count}")

    random.seed(args.seed)
    sampled = random.sample(episodes, args.count)

    manifest_path = os.path.join(args.dest_dir, "sampled_episodes.txt")
    with open(manifest_path, "w") as f:
        for name in sampled:
            f.write(name + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(copy_and_build, args.source_dir, args.dest_dir, name, args.fps)
            for name in sampled
        ]
        completed = 0
        total = len(futures)
        for fut in as_completed(futures):
            _ = fut.result()
            completed += 1
            if completed % 100 == 0:
                print(f"Processed {completed}/{total}")

    zip_directory(args.dest_dir, args.zip_path)


if __name__ == "__main__":
    main()
