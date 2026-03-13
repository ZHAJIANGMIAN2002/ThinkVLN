from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

from thinkvln.datagen.generation.watcher_utils import (
    append_jsonl,
    format_traj,
    load_jsonl,
    load_jsonl_by_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge watcher manifest and annotations.")
    parser.add_argument("--manifest_file", type=Path, required=True)
    parser.add_argument("--annotation_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    return parser.parse_args()


def merge_dataset(
    manifest_file: Path,
    annotation_file: Path,
    output_file: Path,
) -> Tuple[int, int]:
    manifest_rows = load_jsonl(manifest_file)
    annotations = load_jsonl_by_key(annotation_file, "sample_id")

    written = 0
    missing = 0
    if output_file.exists():
        output_file.unlink()

    for row in manifest_rows:
        sample_id = str(row.get("sample_id", ""))
        annotation = annotations.get(sample_id)
        if not sample_id or annotation is None:
            missing += 1
            continue

        merged = {
            "sample_id": sample_id,
            "episode_key": row["episode_key"],
            "episode_id": row["episode_id"],
            "scene_id": row["scene_id"],
            "pivot_frame": row["pivot_frame"],
            "rollout_id": row["rollout_id"],
            "subtask_id": row["subtask_id"],
            "instruction": row["instruction"],
            "subtask_text": row["subtask_text"],
            "memory_start": annotation["memory_start"],
            "traj": format_traj(row.get("actions", [])),
            "label": annotation["label"],
            "memory_end": annotation["memory_end"],
            "base_image_path": row["base_image_path"],
            "pivot_image_relpath": row["pivot_image_relpath"],
            "rollout_image_relpaths": row["rollout_image_relpaths"],
        }
        append_jsonl(output_file, merged)
        written += 1

    return written, missing


def main() -> None:
    args = parse_args()
    written, missing = merge_dataset(
        manifest_file=args.manifest_file,
        annotation_file=args.annotation_file,
        output_file=args.output_file,
    )
    print(f"watcher dataset written: {written}")
    print(f"watcher dataset missing annotations: {missing}")


if __name__ == "__main__":
    main()
