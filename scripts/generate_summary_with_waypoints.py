#!/usr/bin/env python3
"""Generate merged summary jsonl with waypoint labels."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from scripts.generate_summary_full import get_episode_key, load_jsonl_by_key


def load_summary_waypoint(trajectory_dir: str) -> List[Dict[str, Any]]:
    cand = [
        os.path.join(trajectory_dir, 'summary_waypoint.json'),
        os.path.join(trajectory_dir, 'summary_with_waypoints.json'),
        os.path.join(trajectory_dir, 'summary.json'),
    ]
    summary_file = None
    for path in cand:
        if os.path.exists(path):
            summary_file = path
            break
    if summary_file is None:
        raise FileNotFoundError(f'No summary file found under {trajectory_dir}')

    rows: List[Dict[str, Any]] = []
    with open(summary_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def merge_summary_with_waypoints(
    trajectory_dir: str,
    split_file: str,
    determination_file: str,
    summary_full_waypoint_file: str,
) -> None:
    episodes = load_summary_waypoint(trajectory_dir)
    split_map = load_jsonl_by_key(split_file)
    determination_map = load_jsonl_by_key(determination_file)

    os.makedirs(os.path.dirname(summary_full_waypoint_file), exist_ok=True)
    merged = 0
    skipped = 0

    with open(summary_full_waypoint_file, 'w', encoding='utf-8') as out_f:
        for ep in episodes:
            episode_id = ep.get('id')
            scene_id = ep.get('scene_id')
            episode_key = get_episode_key(scene_id, episode_id)

            split_record = split_map.get(episode_key)
            determination_record = determination_map.get(episode_key)
            if not split_record or not determination_record:
                skipped += 1
                continue

            instruction = split_record.get('instruction')
            if not instruction:
                instructions = ep.get('instructions')
                instruction = instructions[0] if isinstance(instructions, list) and instructions else instructions

            actions = ep.get('actions', [])
            positions = ep.get('positions', [])
            num_frames = min(len(actions), len(positions)) if positions else len(actions)

            merged_record = {
                'episode_key': episode_key,
                'episode_id': episode_id,
                'id': episode_id,
                'scene_id': scene_id,
                'trajectory_id': ep.get('trajectory_id'),
                'instruction': instruction,
                'plan': split_record.get('plan', []),
                'actions': actions,
                'positions': positions,
                'num_frames': num_frames,
                'video': ep.get('video', ''),
                'num_subtasks': determination_record.get('num_subtasks'),
                'subtask_sequence': determination_record.get('subtask_sequence'),
                'subtask_counts': determination_record.get('subtask_counts'),
                'transition_points': determination_record.get('transition_points'),
            }
            out_f.write(json.dumps(merged_record, ensure_ascii=False) + '\n')
            merged += 1

    print(f'summary_full_waypoint merged: {merged}')
    print(f'summary_full_waypoint skipped: {skipped}')
    print(f'summary_full_waypoint path: {summary_full_waypoint_file}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge summary with waypoint metadata')
    parser.add_argument('--trajectory_dir', type=str, required=True)
    parser.add_argument('--split_file', type=str, required=True)
    parser.add_argument('--determination_file', type=str, required=True)
    parser.add_argument('--summary_full_waypoint_file', type=str, required=True)
    args = parser.parse_args()

    merge_summary_with_waypoints(
        trajectory_dir=args.trajectory_dir,
        split_file=args.split_file,
        determination_file=args.determination_file,
        summary_full_waypoint_file=args.summary_full_waypoint_file,
    )


if __name__ == '__main__':
    main()
