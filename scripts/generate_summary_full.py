import argparse
import json
import os
from typing import Any, Dict, List

from thinkvln.datagen.generation import subtask_determination as det_mod
from thinkvln.datagen.generation import subtask_split as split_mod


def get_episode_key(scene_id: str, episode_id: int) -> str:
    if scene_id:
        return f"{scene_id}_{episode_id}"
    return str(episode_id)


def load_summary(trajectory_dir: str) -> List[Dict[str, Any]]:
    summary_file = os.path.join(trajectory_dir, "summary.json")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"summary.json not found: {summary_file}")

    episodes: List[Dict[str, Any]] = []
    with open(summary_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            episodes.append(json.loads(line))
    return episodes


def load_jsonl_by_key(path: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = record.get("episode_key")
            if not key and "scene_id" in record and "episode_id" in record:
                key = get_episode_key(record.get("scene_id"), record.get("episode_id"))
            if key:
                data[key] = record
    return data


def merge_summary_with_subtasks(
    trajectory_dir: str,
    split_file: str,
    determination_file: str,
    summary_full_file: str,
) -> None:
    episodes = load_summary(trajectory_dir)
    split_map = load_jsonl_by_key(split_file)
    determination_map = load_jsonl_by_key(determination_file)

    os.makedirs(os.path.dirname(summary_full_file), exist_ok=True)
    merged = 0
    skipped = 0

    with open(summary_full_file, "w", encoding="utf-8") as out_f:
        for ep in episodes:
            episode_id = ep.get("id")
            scene_id = ep.get("scene_id")
            episode_key = get_episode_key(scene_id, episode_id)

            split_record = split_map.get(episode_key)
            determination_record = determination_map.get(episode_key)
            if not split_record or not determination_record:
                skipped += 1
                continue

            instruction = split_record.get("instruction")
            if not instruction:
                instructions = ep.get("instructions")
                if isinstance(instructions, list) and instructions:
                    instruction = instructions[0]
                else:
                    instruction = instructions

            merged_record = {
                "episode_key": episode_key,
                "episode_id": episode_id,
                "id": episode_id,
                "scene_id": scene_id,
                "trajectory_id": ep.get("trajectory_id"),
                "instruction": instruction,
                "plan": split_record.get("plan", []),
                "actions": ep.get("actions", []),
                "video": ep.get("video", ""),
                "num_subtasks": determination_record.get("num_subtasks"),
                "num_frames": determination_record.get("num_frames"),
                "keyframes": determination_record.get("keyframes"),
                "subtask_sequence": determination_record.get("subtask_sequence"),
                "subtask_counts": determination_record.get("subtask_counts"),
                "transition_points": determination_record.get("transition_points"),
            }

            out_f.write(json.dumps(merged_record, ensure_ascii=False) + "\n")
            merged += 1

    print(f"summary_full merged: {merged}")
    print(f"summary_full skipped (missing split/determination): {skipped}")
    print(f"summary_full path: {summary_full_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-file pipeline: subtask_split + subtask_determination + merge summary_full.jsonl"
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        required=True,
        help="Trajectory directory containing summary.json and videos",
    )
    parser.add_argument(
        "--split_output_file",
        type=str,
        required=True,
        help="Output JSONL path for subtask split",
    )
    parser.add_argument(
        "--determination_output_file",
        type=str,
        required=True,
        help="Output JSONL path for subtask determination",
    )
    parser.add_argument(
        "--summary_full_file",
        type=str,
        required=True,
        help="Output path for merged summary_full.jsonl",
    )
    parser.add_argument("--max_workers_split", type=int, default=8)
    parser.add_argument("--max_workers_det", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.split_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.determination_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_full_file), exist_ok=True)

    print("Step 1/3: subtask split")
    split_mod.process_episodes_for_subtask_split(
        trajectory_dir=args.trajectory_dir,
        output_file=args.split_output_file,
        max_workers=args.max_workers_split,
    )

    print("Step 2/3: subtask determination")
    det_mod.determine_subtasks_for_all_episodes(
        trajectory_dir=args.trajectory_dir,
        subtask_splits_file=args.split_output_file,
        output_file=args.determination_output_file,
        max_workers=args.max_workers_det,
    )

    print("Step 3/3: merge summary_full")
    merge_summary_with_subtasks(
        trajectory_dir=args.trajectory_dir,
        split_file=args.split_output_file,
        determination_file=args.determination_output_file,
        summary_full_file=args.summary_full_file,
    )


if __name__ == "__main__":
    main()

