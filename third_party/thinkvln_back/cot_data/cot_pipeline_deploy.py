import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

import subtask_split_deploy as split_mod
import subtask_determination_deploy as determine_mod
import cot_generation_deploy as cot_mod


def get_episode_key(scene_id: str, episode_id: int) -> str:
    if scene_id:
        return f"{scene_id}_{episode_id}"
    return str(episode_id)


def load_summary(trajectory_dir: str) -> List[Dict[str, Any]]:
    summary_file = os.path.join(trajectory_dir, "summary.json")
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    episodes = []
    with open(summary_file, "r") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def load_jsonl_by_key(path: str, key_field: str = "episode_key") -> Dict[str, Dict[str, Any]]:
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get(key_field)
            if not key and "scene_id" in record and "episode_id" in record:
                key = get_episode_key(record.get("scene_id"), record.get("episode_id"))
            if key:
                data[key] = record
    return data


def load_processed_episode_keys(path: str) -> set:
    processed = set()
    if not os.path.exists(path):
        return processed
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            scene_id = record.get("scene_id")
            episode_id = record.get("episode_id")
            episode_key = record.get("episode_key")
            key = resolve_episode_key(episode_key, scene_id, episode_id)
            if key:
                processed.add(key)
    return processed


def resolve_episode_key(episode_key: Optional[str], scene_id: Optional[str], episode_id: Optional[int]) -> Optional[str]:
    if episode_key:
        return episode_key
    if scene_id is not None and episode_id is not None:
        return get_episode_key(scene_id, episode_id)
    return None


def find_episode(trajectory_dir: str, target_key: str) -> Dict[str, Any]:
    episodes = load_summary(trajectory_dir)
    for ep in episodes:
        episode_key = get_episode_key(ep.get("scene_id"), ep.get("id"))
        if episode_key == target_key:
            return ep
    raise ValueError(f"Episode not found in summary.json: {target_key}")


def build_video_from_rgb_map(
    frame_dir: str,
    output_path: str,
    fps: int = 6,
) -> None:
    rgb_files = [f for f in os.listdir(frame_dir) if f.endswith("_rgb.jpg")]
    if not rgb_files:
        raise FileNotFoundError(f"No RGB frames found in {frame_dir}")

    indices = []
    for name in rgb_files:
        try:
            indices.append(int(name.split("_")[0]))
        except ValueError:
            continue
    indices = sorted(set(indices))
    if not indices:
        raise ValueError(f"No valid frame indices in {frame_dir}")

    first_rgb = cv2.imread(os.path.join(frame_dir, f"{indices[0]:06d}_rgb.jpg"))
    first_map = cv2.imread(os.path.join(frame_dir, f"{indices[0]:06d}_map.jpg"))
    if first_rgb is None or first_map is None:
        raise FileNotFoundError(f"Missing rgb/map for frame {indices[0]:06d} in {frame_dir}")

    rgb_h, rgb_w = first_rgb.shape[:2]
    map_h, map_w = first_map.shape[:2]
    if map_h != rgb_h:
        new_w = int(map_w * (rgb_h / map_h))
        first_map = cv2.resize(first_map, (new_w, rgb_h), interpolation=cv2.INTER_AREA)
        map_h, map_w = first_map.shape[:2]

    video_w = rgb_w + map_w
    video_h = rgb_h

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_w, video_h),
    )

    for idx in indices:
        rgb_path = os.path.join(frame_dir, f"{idx:06d}_rgb.jpg")
        map_path = os.path.join(frame_dir, f"{idx:06d}_map.jpg")
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


def merge_summary_with_subtasks(
    trajectory_dir: str,
    subtask_splits_file: str,
    determination_file: str,
    output_file: str,
) -> None:
    episodes = load_summary(trajectory_dir)
    split_map = load_jsonl_by_key(subtask_splits_file)
    determine_map = load_jsonl_by_key(determination_file)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    merged_count = 0
    skipped_count = 0

    with open(output_file, "w") as f:
        for ep in episodes:
            episode_id = ep.get("id")
            scene_id = ep.get("scene_id")
            episode_key = get_episode_key(scene_id, episode_id)

            split_record = split_map.get(episode_key)
            determine_record = determine_map.get(episode_key)
            if not split_record or not determine_record:
                skipped_count += 1
                continue

            instruction = split_record.get("instruction")
            if not instruction:
                instructions = ep.get("instructions")
                if isinstance(instructions, list) and instructions:
                    instruction = instructions[0]
                else:
                    instruction = instructions

            plan = split_record.get("plan", [])

            merged_record = {
                "episode_key": episode_key,
                "episode_id": episode_id,
                "id": episode_id,
                "scene_id": scene_id,
                "trajectory_id": ep.get("trajectory_id"),
                "instruction": instruction,
                "plan": plan,
                "actions": ep.get("actions", []),
                "video": ep.get("video", ""),
                "num_subtasks": determine_record.get("num_subtasks"),
                "num_frames": determine_record.get("num_frames"),
                "keyframes": determine_record.get("keyframes"),
                "subtask_sequence": determine_record.get("subtask_sequence"),
                "subtask_counts": determine_record.get("subtask_counts"),
                "transition_points": determine_record.get("transition_points"),
            }

            json.dump(merged_record, f)
            f.write("\n")
            merged_count += 1

    print(f"Merged summary_full.jsonl: {merged_count} episodes")
    print(f"Skipped (missing split/determination): {skipped_count} episodes")


def run_cot_generation(
    trajectory_dir: str,
    annotation_file: str,
    output_file: str,
    max_workers: int,
    max_episodes: int,
    frame_base_dir: str,
    use_frames_only: bool,
) -> None:
    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "cot_generation_deploy.py",
            "--trajectory_dir",
            trajectory_dir,
            "--annotation_file",
            annotation_file,
            "--output_file",
            output_file,
            "--max_workers",
            str(max_workers),
        ]
        if max_episodes is not None:
            sys.argv.extend(["--max_episodes", str(max_episodes)])
        if frame_base_dir:
            sys.argv.extend(["--frame_base_dir", frame_base_dir])
        if use_frames_only:
            sys.argv.append("--use_frames_only")
        cot_mod.main()
    finally:
        sys.argv = argv_backup


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot CoT pipeline (deploy)")
    parser.add_argument("--trajectory_dir", type=str, default="data/trajectory_data/R2R",
                        help="Path to trajectory data directory (contains summary.json)")
    parser.add_argument("--split_output_file", type=str,
                        default="data/subtask_splits/R2R/subtask_splits.jsonl",
                        help="Output file for subtask splits")
    parser.add_argument("--determination_output_file", type=str,
                        default="data/subtask_determination_results/subtask_determination.jsonl",
                        help="Output file for subtask determination results")
    parser.add_argument("--summary_full_file", type=str,
                        default=None,
                        help="Output summary_full.jsonl path (default: <trajectory_dir>/summary_full.jsonl)")
    parser.add_argument("--cot_output_file", type=str,
                        default="data/cot_dataset/R2R/cot_dataset.jsonl",
                        help="Output COT JSONL file")
    parser.add_argument("--max_workers_split", type=int, default=8)
    parser.add_argument("--max_workers_det", type=int, default=4)
    parser.add_argument("--max_workers_cot", type=int, default=4)
    parser.add_argument("--max_episodes_cot", type=int, default=None)
    parser.add_argument("--episode_key", type=str, default=None,
                        help="Process a single episode key (scene_id_episode_id)")
    parser.add_argument("--scene_id", type=str, default=None,
                        help="Process a single episode by scene_id + episode_id")
    parser.add_argument("--episode_id", type=int, default=None,
                        help="Episode id for single-episode mode")
    parser.add_argument("--frame_base_dir", type=str, default=None,
                        help="Base dir with *_rgb.jpg and *_map.jpg frames")
    parser.add_argument("--use_frames_only", action="store_true",
                        help="Use rgb/map frames only (no mp4 required)")
    parser.add_argument("--build_video_from_frames", action="store_true",
                        help="Build trajectory.mp4 from rgb/map frames for single-episode mode")
    args = parser.parse_args()

    summary_full_file = args.summary_full_file or os.path.join(args.trajectory_dir, "summary_full.jsonl")
    os.makedirs(os.path.dirname(args.split_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.determination_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(summary_full_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.cot_output_file), exist_ok=True)

    target_key = resolve_episode_key(args.episode_key, args.scene_id, args.episode_id)
    if target_key:
        episode_data = find_episode(args.trajectory_dir, target_key)
        print(f"Running single-episode mode: {target_key}")

        if args.build_video_from_frames:
            video_rel_path = episode_data.get("video", "")
            if not video_rel_path:
                raise ValueError("Episode has no video path in summary.json")
            frame_dir = None
            if args.frame_base_dir:
                frame_dir = os.path.join(args.frame_base_dir, os.path.basename(video_rel_path))
            if not frame_dir or not os.path.exists(frame_dir):
                raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

            video_path = os.path.join(args.trajectory_dir, video_rel_path, "trajectory.mp4")
            if not os.path.exists(video_path):
                print(f"Building trajectory.mp4 from frames: {frame_dir}")
                build_video_from_rgb_map(frame_dir, video_path, fps=6)

        split_mod.process_single_episode(
            episode_data=episode_data,
            episode_idx=0,
            total_episodes=1,
            processed_episode_keys=load_processed_episode_keys(args.split_output_file),
            output_file=args.split_output_file,
            write_lock=split_mod.Lock(),
        )

        subtask_splits = load_jsonl_by_key(args.split_output_file)
        determine_mod.process_single_episode(
            episode_data=episode_data,
            episode_idx=0,
            total_episodes=1,
            subtask_splits=subtask_splits,
            trajectory_dir=args.trajectory_dir,
            output_file=args.determination_output_file,
            write_lock=determine_mod.Lock(),
            processed_episode_keys=load_processed_episode_keys(args.determination_output_file),
            frame_base_dir=args.frame_base_dir,
            use_frames_only=args.use_frames_only,
        )
    else:
        split_mod.process_episodes_for_subtask_split(
            trajectory_dir=args.trajectory_dir,
            output_file=args.split_output_file,
            max_workers=args.max_workers_split,
        )

        determine_mod.determine_subtasks_for_all_episodes(
            trajectory_dir=args.trajectory_dir,
            subtask_splits_file=args.split_output_file,
            output_file=args.determination_output_file,
            max_workers=args.max_workers_det,
            frame_base_dir=args.frame_base_dir,
            use_frames_only=args.use_frames_only,
        )

    merge_summary_with_subtasks(
        trajectory_dir=args.trajectory_dir,
        subtask_splits_file=args.split_output_file,
        determination_file=args.determination_output_file,
        output_file=summary_full_file,
    )

    run_cot_generation(
        trajectory_dir=args.trajectory_dir,
        annotation_file=summary_full_file,
        output_file=args.cot_output_file,
        max_workers=args.max_workers_cot,
        max_episodes=args.max_episodes_cot,
        frame_base_dir=args.frame_base_dir,
        use_frames_only=args.use_frames_only,
    )


if __name__ == "__main__":
    main()
