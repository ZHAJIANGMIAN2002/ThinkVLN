from __future__ import annotations

import hashlib
import json
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List, Optional, Tuple


ACTION_ID_TO_NAME = {
    0: "stop",
    1: "forward",
    2: "turn_left",
    3: "turn_right",
}


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def load_jsonl_by_key(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        value = row.get(key)
        if isinstance(value, str) and value:
            rows[value] = row
    return rows


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_sample_id(episode_key: str, pivot_frame: int, rollout_id: int) -> str:
    return f"{episode_key}_p{int(pivot_frame):06d}_r{int(rollout_id):02d}"


def build_pivot_image_relpath(episode_key: str, pivot_frame: int) -> str:
    return f"pivot/{episode_key}/pivot_{int(pivot_frame):06d}_rgb.jpg"


def build_rollout_image_relpath(
    episode_key: str,
    pivot_frame: int,
    rollout_id: int,
    step_idx: int,
) -> str:
    return (
        f"rollout/{episode_key}/pivot_{int(pivot_frame):06d}/"
        f"rollout_{int(rollout_id):02d}/{int(step_idx):06d}_rgb.jpg"
    )


def resolve_bundle_image_path(bundle_root: Path, base_image_path: str, relpath: str) -> Path:
    return bundle_root / base_image_path / relpath


def action_id_to_name(action: Any) -> str:
    try:
        action_id = int(action)
    except (TypeError, ValueError):
        return str(action)
    return ACTION_ID_TO_NAME.get(action_id, str(action_id))


def format_progress(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def format_traj(actions: List[Any], progresses: Optional[List[Any]] = None) -> str:
    action_text = ",".join(action_id_to_name(action) for action in actions)
    if not progresses:
        return f"a=[{action_text}]"
    progress_text = ",".join(format_progress(progress) for progress in progresses)
    return f"a=[{action_text}], p=[{progress_text}]"


def select_label_image_relpaths(
    pivot_image_relpath: str,
    rollout_image_relpaths: List[str],
    stride: int,
) -> List[str]:
    stride = max(1, int(stride))
    selected = [pivot_image_relpath]
    selected.extend(rollout_image_relpaths[::stride])
    if rollout_image_relpaths:
        last_relpath = rollout_image_relpaths[-1]
        if not selected or selected[-1] != last_relpath:
            selected.append(last_relpath)
    return selected


def derive_seed(base_seed: int, *parts: Any) -> int:
    digest = hashlib.sha1(
        ("::".join([str(int(base_seed))] + [str(part) for part in parts])).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty response")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise ValueError("response is not valid json")
        payload = json.loads(raw[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response json must be an object")
    return payload


def select_pivots(
    spans: List[Tuple[int, int, int]],
    num_pivots: int,
    rng: Random,
) -> List[Dict[str, Any]]:
    if num_pivots <= 0 or not spans:
        return []
    candidates = [
        {"pivot_frame": int(frame), "subtask_id": int(subtask_id)}
        for subtask_id, start, end in spans
        for frame in range(int(start), int(end) + 1)
    ]
    if len(candidates) <= num_pivots:
        selected = list(candidates)
    else:
        indices = sorted(rng.sample(range(len(candidates)), num_pivots))
        selected = [candidates[index] for index in indices]
    selected.sort(key=lambda item: int(item["pivot_frame"]))
    return selected


__all__ = [
    "ACTION_ID_TO_NAME",
    "action_id_to_name",
    "append_jsonl",
    "build_pivot_image_relpath",
    "build_rollout_image_relpath",
    "build_sample_id",
    "derive_seed",
    "ensure_parent",
    "extract_json_object",
    "format_traj",
    "iter_jsonl",
    "load_jsonl",
    "load_jsonl_by_key",
    "resolve_bundle_image_path",
    "select_label_image_relpaths",
    "select_pivots",
]
