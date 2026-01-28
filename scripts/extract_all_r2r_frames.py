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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

import cv2


def find_trajectory_videos(video_root: Path) -> List[Path]:
    """Return all `trajectory.mp4` files beneath the provided root."""
    return sorted(video_root.rglob("trajectory.mp4"))


def extract_frames(
    video_path: Path,
    video_root: Path,
    output_root: Path,
    overwrite: bool,
) -> Dict[str, object]:
    """Extract every frame from `video_path` and store them under `output_root`."""
    rel_dir = video_path.parent.relative_to(video_root)
    destination_dir = output_root / rel_dir
    destination_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video": str(rel_dir),
            "error": "cannot open video",
            "frames_extracted": 0,
            "frames_skipped": 0,
        }

    frame_idx = 0
    saved = 0
    skipped = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_path = destination_dir / f"{frame_idx:06d}.jpg"
            if frame_path.exists() and not overwrite:
                skipped += 1
            else:
                cv2.imwrite(str(frame_path), frame)
                saved += 1

            frame_idx += 1
    finally:
        cap.release()

    return {
        "video": str(rel_dir),
        "frames_extracted": saved,
        "frames_skipped": skipped,
        "frames_total": frame_idx,
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
        help="Overwrite existing frame files instead of skipping them.",
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
            executor.submit(extract_frames, video, args.video_root, args.output_dir, args.overwrite): video
            for video in videos
        }

        for future in as_completed(futures):
            video_path = futures[future]
            try:
                result = future.result()
                stats.append(result)
                logging.info(
                    "Video %s → extracted=%s skipped=%s total=%s",
                    result["video"],
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
