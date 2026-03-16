#!/usr/bin/env python3
"""
Extract every frame from the R2R trajectory videos.

Walks through all subdirectories under a base path, looks for `trajectory.mp4`
files, and writes every frame found into a mirrored directory tree inside the
target output folder.

Frames are written as zero-padded JPEGs, and the extraction process can run
over multiple workers to parallelise the work.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

import cv2


def find_trajectory_videos(video_root: Path) -> List[Path]:
    """Return all `trajectory.mp4` files beneath the provided root."""
    return sorted(video_root.rglob("trajectory.mp4"))


def get_episode_state(destination_dir: Path) -> str:
    if not destination_dir.exists():
        return "missing"
    rgb_files = sorted(destination_dir.glob("*_rgb.jpg"))
    map_files = sorted(destination_dir.glob("*_map.jpg"))
    if rgb_files and len(rgb_files) == len(map_files):
        return "complete"
    return "partial"


def split_frame(frame, fixed_rgb_width: int):
    height, width = frame.shape[:2]
    if width <= fixed_rgb_width:
        raise ValueError(f"frame width {width} must be larger than rgb width {fixed_rgb_width}")
    return frame[:, :fixed_rgb_width], frame[:, fixed_rgb_width:]


def extract_frames(
    video_path: Path,
    video_root: Path,
    output_root: Path,
    overwrite: bool,
    fixed_rgb_width: int = 640,
) -> Dict[str, object]:
    """Extract every frame from `video_path` and store them under `output_root`."""
    rel_dir = video_path.parent.relative_to(video_root)
    destination_dir = output_root / rel_dir
    state = get_episode_state(destination_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video": str(rel_dir),
            "error": "cannot open video",
            "frames_extracted": 0,
            "frames_skipped": 0,
            "status": "error",
        }

    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if state == "complete" and not overwrite:
        cap.release()
        return {
            "video": str(rel_dir),
            "frames_extracted": 0,
            "frames_skipped": frames_total * 2,
            "frames_total": frames_total,
            "status": "skipped_complete",
        }

    if state == "partial" or overwrite:
        shutil.rmtree(destination_dir, ignore_errors=True)
    destination_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    skipped = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            rgb_frame, map_frame = split_frame(frame, fixed_rgb_width)
            rgb_path = destination_dir / f"{frame_idx:06d}_rgb.jpg"
            map_path = destination_dir / f"{frame_idx:06d}_map.jpg"
            if rgb_path.exists() and map_path.exists() and not overwrite:
                skipped += 2
            else:
                cv2.imwrite(str(rgb_path), rgb_frame)
                cv2.imwrite(str(map_path), map_frame)
                saved += 2

            frame_idx += 1
    finally:
        cap.release()

    status = "rebuilt_partial" if state == "partial" else "extracted"
    return {
        "video": str(rel_dir),
        "frames_extracted": saved,
        "frames_skipped": skipped,
        "frames_total": frame_idx,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract all R2R trajectory frames.")
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("/mnt/swx/ThinkVLN/data/trajectory_data/R2R_back/images"),
        help="Root directory that contains the trajectory videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/nvme/swx/dataset/R2R"),
        help="Directory where extracted frames and folder structure are written.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of videos to process in parallel.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild episode outputs even when rgb/map frames already exist.",
    )
    parser.add_argument(
        "--fixed-rgb-width",
        type=int,
        default=640,
        help="Width of the left RGB panel inside each composite video frame.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not args.video_root.exists():
        logging.error("video root %s does not exist", args.video_root)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_trajectory_videos(args.video_root)
    if not videos:
        logging.warning("No trajectory videos found under %s", args.video_root)
        return

    logging.info("Found %d videos. Extracting frames with %d workers...", len(videos), args.num_workers)

    stats: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                extract_frames,
                video,
                args.video_root,
                args.output_dir,
                args.overwrite,
                args.fixed_rgb_width,
            ): video
            for video in videos
        }

        for future in as_completed(futures):
            video_path = futures[future]
            try:
                result = future.result()
                stats.append(result)
                logging.info(
                    "Video %s → status=%s extracted=%s skipped=%s total=%s",
                    result["video"],
                    result.get("status"),
                    result.get("frames_extracted"),
                    result.get("frames_skipped"),
                    result.get("frames_total"),
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.error("Failed to process %s: %s", video_path, exc)

    total_frames = sum(s.get("frames_extracted", 0) for s in stats)
    total_skipped = sum(s.get("frames_skipped", 0) for s in stats)
    total_videos = len(videos)

    logging.info("Jobs finished: videos=%d frames=%d skipped=%d", total_videos, total_frames, total_skipped)


if __name__ == "__main__":
    main()
