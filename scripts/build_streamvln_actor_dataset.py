#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from streamvln.dataset.streamvln_actor_dataset import iter_materialized_streamvln_actor_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a materialized StreamVLN actor training dataset.")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--r2r_summary_path", type=Path, default=None)
    parser.add_argument("--r2r_image_root", type=Path, default=None)
    parser.add_argument("--scalevln_summary_path", type=Path, default=None)
    parser.add_argument("--scalevln_image_root", type=Path, default=None)
    parser.add_argument("--watcher_hint_path", type=Path, default=None)
    parser.add_argument("--watcher_memory_ratio", type=float, default=1.0)
    parser.add_argument("--memory_num_history_images", type=int, default=8)
    parser.add_argument("--memory_pre_anchor_count", type=int, default=None)
    parser.add_argument("--memory_post_anchor_count", type=int, default=None)
    parser.add_argument("--done_threshold", type=float, default=0.85)
    parser.add_argument("--use_next_token", action="store_true")
    parser.add_argument("--use_sliding_window", action="store_true")
    parser.add_argument("--action_history_len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_specs = []
    if args.r2r_summary_path is not None:
        summary_specs.append(
            {
                "dataset_name": "r2r",
                "summary_path": str(args.r2r_summary_path),
                "image_root": str(args.r2r_image_root) if args.r2r_image_root is not None else "",
            }
        )
    if args.scalevln_summary_path is not None:
        summary_specs.append(
            {
                "dataset_name": "scalevln",
                "summary_path": str(args.scalevln_summary_path),
                "image_root": str(args.scalevln_image_root) if args.scalevln_image_root is not None else "",
            }
        )
    if not summary_specs:
        raise ValueError("At least one of --r2r_summary_path or --scalevln_summary_path must be provided.")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for row in iter_materialized_streamvln_actor_records(
            summary_specs=summary_specs,
            watcher_memory_path=str(args.watcher_hint_path) if args.watcher_hint_path is not None else None,
            watcher_memory_ratio=float(args.watcher_memory_ratio),
            done_threshold=float(args.done_threshold),
            memory_num_history_images=int(args.memory_num_history_images),
            memory_pre_anchor_count=args.memory_pre_anchor_count,
            memory_post_anchor_count=args.memory_post_anchor_count,
            use_next_token=bool(args.use_next_token),
            use_sliding_window=bool(args.use_sliding_window),
            action_history_len=int(args.action_history_len),
            seed=int(args.seed),
        ):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"streamvln actor dataset written: {count}")
    print(f"output_path: {args.output_path}")


if __name__ == "__main__":
    main()
