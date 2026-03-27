from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_sample_id_to_episode_frame(sample_id: str) -> Tuple[str, int]:
    text = str(sample_id or "").strip()
    match = re.match(r"^(?P<episode_key>.+)_p(?P<frame_idx>\d+)_r\d+$", text)
    if not match:
        raise ValueError(f"Invalid sample_id format: {sample_id}")
    return match.group("episode_key"), int(match.group("frame_idx"))


def convert_watcher_dataset_to_actor_memory_records(watcher_dataset_path: str) -> List[Dict]:
    records: List[Dict] = []
    for row in _load_jsonl(watcher_dataset_path):
        episode_key = str(row.get("episode_key") or "").strip()
        hint = str(row.get("watcher_hint") or row.get("memory_start") or "").strip()
        frame_idx = row.get("frame_idx", row.get("pivot_frame"))
        if (not episode_key or frame_idx is None) and row.get("sample_id"):
            episode_key, frame_idx = parse_sample_id_to_episode_frame(str(row.get("sample_id")))
        if not episode_key or not hint or frame_idx is None:
            continue
        records.append(
            {
                "episode_key": episode_key,
                "frame_idx": int(frame_idx),
                "watcher_hint": hint,
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert watcher dataset rows into GT-aligned actor memory rows.")
    parser.add_argument("--watcher_dataset_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = convert_watcher_dataset_to_actor_memory_records(str(args.watcher_dataset_path))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"actor memory rows written: {len(rows)}")


if __name__ == "__main__":
    main()
