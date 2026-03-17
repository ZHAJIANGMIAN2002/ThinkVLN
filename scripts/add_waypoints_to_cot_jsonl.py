#!/usr/bin/env python3
"""Add synthetic waypoint fields to cot_data jsonl records.

Waypoint source:
- Use trajectory actions from summary.json (matched by episode_key).
- Integrate a simple 2D kinematic model to produce per-frame positions.
- Derive per-frame delta waypoints from positions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def get_episode_key(record: Dict[str, Any]) -> str:
    key = record.get("episode_key")
    if key:
        return str(key)
    scene_id = record.get("scene_id")
    episode_id = record.get("episode_id", record.get("id"))
    if scene_id is None or episode_id is None:
        return ""
    return f"{scene_id}_{int(episode_id)}"


def load_summary_actions(summary_path: Path) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    with summary_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = str(item.get("key", ""))
            actions = item.get("actions", [])
            if key and isinstance(actions, list):
                mapping[key] = actions
    return mapping


def integrate_positions_from_actions(
    actions: Iterable[int],
    forward_step_m: float,
    turn_deg: float,
) -> List[List[float]]:
    theta = 0.0
    x = 0.0
    z = 0.0
    turn_rad = math.radians(turn_deg)
    positions: List[List[float]] = []

    for act in actions:
        try:
            a = int(act)
        except (TypeError, ValueError):
            a = 0

        if a == 2:  # turn_left
            theta += turn_rad
        elif a == 3:  # turn_right
            theta -= turn_rad
        elif a == 1:  # forward
            x += forward_step_m * math.sin(theta)
            z += forward_step_m * math.cos(theta)
        # a in {-1, 0} or unknown => no motion
        positions.append([float(x), float(z)])
    return positions


def make_delta_waypoints(
    positions: List[List[float]],
    horizon: int,
) -> List[List[List[float]]]:
    if not positions:
        return []

    out: List[List[List[float]]] = []
    n = len(positions)
    for i in range(n):
        px, pz = positions[i]
        frame_wps: List[List[float]] = []
        for k in range(1, horizon + 1):
            j = min(i + k, n - 1)
            nx, nz = positions[j]
            frame_wps.append([float(nx - px), float(nz - pz)])
        out.append(frame_wps)
    return out


def resolve_num_frames(record: Dict[str, Any], actions: List[int], positions: List[List[float]]) -> int:
    nf = record.get("num_frames")
    if isinstance(nf, int) and nf > 0:
        return nf
    if actions:
        return len(actions)
    return len(positions)


def augment_file(
    input_path: Path,
    output_path: Path,
    summary_actions: Dict[str, List[int]],
    horizon: int,
    forward_step_m: float,
    turn_deg: float,
) -> Tuple[int, int]:
    total = 0
    augmented = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)

            episode_key = get_episode_key(record)
            actions = summary_actions.get(episode_key, [])
            positions = integrate_positions_from_actions(actions, forward_step_m=forward_step_m, turn_deg=turn_deg)
            delta_waypoints = make_delta_waypoints(positions, horizon=horizon)

            if positions:
                num_frames = resolve_num_frames(record, actions, positions)
                num_frames = max(1, min(num_frames, len(positions)))
                record["waypoint_source"] = "actions_integrated"
                record["positions"] = positions[:num_frames]
                record["actions"] = actions[:num_frames]
                record["delta_waypoints"] = delta_waypoints[:num_frames]
                record["num_frames"] = num_frames
                augmented += 1
            else:
                record["waypoint_source"] = "missing_actions"
                record.setdefault("positions", [])
                record.setdefault("delta_waypoints", [])

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    return total, augmented


def main() -> None:
    parser = argparse.ArgumentParser(description="Add waypoint fields to cot_data jsonl")
    parser.add_argument("--input", type=str, required=True, help="Input jsonl path")
    parser.add_argument("--output", type=str, required=True, help="Output jsonl path")
    parser.add_argument(
        "--summary",
        type=str,
        default="/mnt/swx/ThinkVLN/data/trajectory_data/R2R_back/summary.json",
        help="Trajectory summary.json with actions",
    )
    parser.add_argument("--horizon", type=int, default=5, help="Delta waypoint horizon")
    parser.add_argument("--forward_step_m", type=float, default=0.25, help="Forward step in meters")
    parser.add_argument("--turn_deg", type=float, default=15.0, help="Turn angle per step in degrees")
    args = parser.parse_args()

    summary_actions = load_summary_actions(Path(args.summary))
    total, augmented = augment_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        summary_actions=summary_actions,
        horizon=int(args.horizon),
        forward_step_m=float(args.forward_step_m),
        turn_deg=float(args.turn_deg),
    )
    print(f"done: total={total}, augmented={augmented}, missing={total - augmented}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
