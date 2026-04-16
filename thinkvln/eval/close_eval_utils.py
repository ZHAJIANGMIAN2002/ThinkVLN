import hashlib
import json
import math
import os
import re
from typing import Any, Dict, List, Sequence, TextIO, Tuple

import numpy as np


def extract_scene_id(scene_id_or_path: str) -> str:
    if not scene_id_or_path:
        return ""
    parts = scene_id_or_path.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def build_episode_key(scene_id: str, episode_id: Any) -> str:
    return f"{scene_id}_{episode_id}"


def _normalize_subtask_idx(value: Any) -> int:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return 1
    return idx if idx > 0 else 1


def parse_plan_steps(plan: Any) -> List[str]:
    if isinstance(plan, list):
        return [str(step).strip() for step in plan if str(step).strip()]
    if isinstance(plan, str):
        steps = []
        for line in plan.splitlines():
            text = line.strip()
            if not text:
                continue
            text = re.sub(r"^\d+\s*[\.\)]\s*", "", text).strip()
            if text:
                steps.append(text)
        return steps
    return []


def load_summary_full(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        raise ValueError("summary_full_path is required for ladder modes.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"summary_full_path not found: {path}")

    summary: Dict[str, Dict[str, Any]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_idx} in {path}: {exc}") from exc

                if not isinstance(record, dict):
                    continue

                key_candidates = []
                raw_episode_key = record.get("episode_key")
                if isinstance(raw_episode_key, str) and raw_episode_key:
                    key_candidates.append(raw_episode_key)

                scene_id = record.get("scene_id")
                episode_id = record.get("episode_id", record.get("id"))
                if scene_id is not None and episode_id is not None:
                    key_candidates.append(build_episode_key(extract_scene_id(str(scene_id)), episode_id))

                for key in key_candidates:
                    summary[key] = record
    except OSError as exc:
        raise RuntimeError(f"Failed reading summary_full_path={path}: {exc}") from exc

    if not summary:
        raise ValueError(f"No episodes loaded from summary_full_path={path}")
    return summary


def build_subtask_spans(subtask_sequence: List[Any]) -> List[Tuple[int, int, int]]:
    if not subtask_sequence:
        return []

    spans: List[Tuple[int, int, int]] = []
    current_idx = _normalize_subtask_idx(subtask_sequence[0])
    start = 0
    for i in range(1, len(subtask_sequence)):
        idx = _normalize_subtask_idx(subtask_sequence[i])
        if idx != current_idx:
            spans.append((current_idx, start, i - 1))
            current_idx = idx
            start = i
    spans.append((current_idx, start, len(subtask_sequence) - 1))
    return spans


def timeline_progress(step_idx: int, gt_subtask_steps: int) -> float:
    if gt_subtask_steps <= 0:
        return 1.0
    progress = (max(step_idx, 0) + 1) / float(gt_subtask_steps)
    return max(0.0, min(1.0, progress))


def distance_based_progress(
    start_to_current_distance: float,
    current_to_goal_distance: float,
) -> float:
    traveled = max(0.0, float(start_to_current_distance))
    remaining = max(0.0, float(current_to_goal_distance))
    denom = traveled + remaining
    if denom <= 1e-6:
        return 1.0
    progress = traveled / denom
    return max(0.0, min(1.0, progress))


def compute_step_budget(gt_subtask_steps: int, factor: float) -> int:
    return max(1, int(math.ceil(max(gt_subtask_steps, 0) * factor)))


def summarize_subtask_aggregation(stats: Dict[str, float]) -> Dict[str, float]:
    def _binary_metrics(prefix: str = "") -> Dict[str, float]:
        tp = float(stats.get(f"{prefix}tp", 0.0))
        tn = float(stats.get(f"{prefix}tn", 0.0))
        fp = float(stats.get(f"{prefix}fp", 0.0))
        fn = float(stats.get(f"{prefix}fn", 0.0))
        total = tp + tn + fp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "accuracy": (tp + tn) / total if total > 0 else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total_subtasks = float(stats.get("subtasks_total", 0.0))
    successful_subtasks = float(stats.get("subtasks_success", 0.0))
    success_steps_count = float(stats.get("steps_success_count", 0.0))
    success_steps_sum = float(stats.get("steps_success_sum", 0.0))
    progress_count = float(stats.get("progress_count", 0.0))
    progress_error_sum = float(stats.get("progress_abs_error_sum", 0.0))
    progress_smooth_count = float(stats.get("progress_smooth_count", 0.0))
    progress_smooth_error_sum = float(stats.get("progress_smooth_abs_error_sum", 0.0))
    done_metrics = _binary_metrics()
    done_smooth_metrics = _binary_metrics("done_smooth_")

    return {
        "subtask_success_rate": successful_subtasks / total_subtasks if total_subtasks > 0 else 0.0,
        "steps_to_subgoal": success_steps_sum / success_steps_count if success_steps_count > 0 else 0.0,
        "progress_mae": progress_error_sum / progress_count if progress_count > 0 else 0.0,
        "progress_smooth_mae": (
            progress_smooth_error_sum / progress_smooth_count if progress_smooth_count > 0 else 0.0
        ),
        "done_accuracy": done_metrics["accuracy"],
        "done_precision": done_metrics["precision"],
        "done_recall": done_metrics["recall"],
        "done_f1": done_metrics["f1"],
        "done_smooth_accuracy": done_smooth_metrics["accuracy"],
        "done_smooth_precision": done_smooth_metrics["precision"],
        "done_smooth_recall": done_smooth_metrics["recall"],
        "done_smooth_f1": done_smooth_metrics["f1"],
    }


def normalize_action(action: Any) -> int:
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, str):
        key = action.strip().lower()
        mapping = {
            "stop": 0,
            "forward": 1,
            "turn_left": 2,
            "turn left": 2,
            "turn_right": 3,
            "turn right": 3,
            "↑": 1,
            "←": 2,
            "→": 3,
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
        }
        return mapping.get(key, 0)
    return 0


def should_sample_episode(scene_id: str, episode_id: Any, sample_rate: float) -> bool:
    rate = float(sample_rate)
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False

    key = f"{scene_id}_{episode_id}".encode("utf-8")
    digest = hashlib.sha1(key).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)
    return value < rate


def shard_items_round_robin(items: Sequence[Any], rank: int, world_size: int) -> List[Any]:
    data = list(items)
    if world_size <= 1:
        return data
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got rank={rank}")
    return data[rank::world_size]


def write_jsonl_record(handle: TextIO, payload: Dict[str, Any], sync_to_disk: bool = True) -> None:
    handle.write(json.dumps(payload) + "\n")
    handle.flush()
    if sync_to_disk:
        os.fsync(handle.fileno())


__all__ = [
    "extract_scene_id",
    "build_episode_key",
    "_normalize_subtask_idx",
    "parse_plan_steps",
    "load_summary_full",
    "build_subtask_spans",
    "timeline_progress",
    "distance_based_progress",
    "compute_step_budget",
    "summarize_subtask_aggregation",
    "normalize_action",
    "should_sample_episode",
    "shard_items_round_robin",
    "write_jsonl_record",
]
