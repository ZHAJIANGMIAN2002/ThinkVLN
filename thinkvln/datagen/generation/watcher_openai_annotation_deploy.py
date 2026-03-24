from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from openai import OpenAI


DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-397B-A17B"
# DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REASONING_EFFORT = "high"
# DEFAULT_API_KEY = "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"

DEFAULT_BASE_URL = "http://localhost:11451/v1"
DEFAULT_API_KEY = "EMPTY"

MEMORY_START_SYSTEM_PROMPT = """You write watcher memory for navigation rollouts.

Return JSON only:
{"memory_start":"..."}

Rules:
- memory_start must be a single line with exactly three short semicolon-separated fragments.
- Use this exact order: traj summary; current state; neutral status
- The first fragment must summarize the path already traveled before the pivot state.
- The second fragment must state where the robot is now at the pivot.
- The third fragment must stay neutral, such as active step in progress, still on current step, or approach still ongoing.
- Sentence fragments are allowed.
- Do not mark ready for next step or task complete.
- Keep only useful progress that still matters.
- Make the current pivot state explicit.
- Write it so memory_end can directly update it in the same format.
- Do not restate instructions, plan steps, or guesses about what to do next.

Good examples:
- {"memory_start":"Left the dining area and entered the hall; at the bathroom entrance facing inward; active step in progress"}
- {"memory_start":"Moved along the wall from the bedroom; near the doorway to the sink area; still on current step"}

Bad examples:
- {"memory_start":"Left the dining area and entered the hall; at the bathroom entrance facing inward; ready for next step"}
- {"memory_start":"First the agent moved forward, then turned left, then moved again, and now sees a hallway."}
- {"memory_start":"The instruction says to go to the bathroom, so the agent should keep going there next."}
"""

ROLLOUT_SYSTEM_PROMPT = """You are a navigation evaluator updating watcher memory and deciding if the current subtask is complete.

You MUST return JSON only, and strictly in this EXACT order:
{"memory_end":"...","done":true/false,"next_subtask":"..."}

# Thinking Guidelines (For your internal reasoning before generating JSON)
1. Identify the robot's current 'Active' step from the plan.
2. Look at the FINAL rollout frame: Has the specific visual or physical goal of this active step been achieved? (e.g., if the step is "enter kitchen", is it clearly inside the kitchen?)
3. Is the robot fully aligned and ready to start the "Pending" step, or is it still adjusting?

# Output Rules

Step 1: memory_end
- Must be exactly three short semicolon-separated fragments: [traj summary]; [current physical state]; [neutral status].
- Update the memory_start with the new progress.
- Keep it concise and state exactly where the robot is in the final frame.

Step 2: done
- Set to true ONLY IF your internal reasoning confirms the active step's goal is fully reached AND the robot is in a stable position to begin the next step.
- Set to false if the robot is still moving toward the goal, still turning, halfway through a door, or recovering from a mistake.
- NEVER set to true just because the robot is "close" to the goal.

Step 3: next_subtask
- If done=false: Write an imperative command to continue or finish the current active step (e.g., "finish turning left").
- If done=true: Write the imperative command for the NEW active step (promoted from pending), or "stop" if no steps remain.

Examples:
{"memory_end":"Reached the hallway entrance; mid-turn facing the wall; step ongoing","done":false,"next_subtask":"finish turning right to face down the hallway"}

{"memory_end":"Cleared the dining area and entered bathroom; standing inside facing the sink; ready for next step","done":true,"next_subtask":"approach the sink"}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watcher annotation deploy tool (standalone + parallel).")
    parser.add_argument("--gt_video_root", type=Path, required=True)
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--manifest_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--summary_full_path", type=Path, default=None)
    parser.add_argument("--image_stride", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--request_timeout", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip_missing_gt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_sanity_check", action="store_true")
    parser.add_argument("--deploy_mode", action="store_true")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api_base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY", DEFAULT_API_KEY))
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=DEFAULT_REASONING_EFFORT,
        choices=["none", "low", "medium", "high"],
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path) or [])


def load_jsonl_by_key(path: Path, key: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path) or []:
        value = row.get(key)
        if isinstance(value, str) and value:
            rows[value] = row
    return rows


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_bundle_image_path(bundle_root: Path, base_image_path: str, relpath: str) -> Path:
    return bundle_root / base_image_path / relpath


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
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response json must be object")
    return payload


def extract_scene_id(scene_id_or_path: str) -> str:
    if not scene_id_or_path:
        return ""
    parts = scene_id_or_path.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def build_episode_key(scene_id: str, episode_id: Any) -> str:
    return f"{scene_id}_{episode_id}"


def normalize_plan_steps(plan: Any) -> List[str]:
    if isinstance(plan, list):
        return [str(step).strip() for step in plan if str(step).strip()]
    if isinstance(plan, str):
        return [line.strip() for line in plan.splitlines() if line.strip()]
    return []


def load_summary_full(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None or not str(path).strip():
        return {}
    if not path.exists():
        raise FileNotFoundError(f"summary_full_path not found: {path}")
    summary: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            episode_key = str(record.get("episode_key", "")).strip()
            if episode_key:
                summary[episode_key] = record
            scene_id = record.get("scene_id")
            episode_id = record.get("episode_id", record.get("id"))
            if scene_id is not None and episode_id is not None:
                summary[build_episode_key(extract_scene_id(str(scene_id)), episode_id)] = record
    return summary


def resolve_summary_record(record: Dict[str, Any], summary_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    episode_key = str(record.get("episode_key", "")).strip()
    if episode_key and episode_key in summary_lookup:
        return summary_lookup[episode_key]
    scene_id = record.get("scene_id")
    episode_id = record.get("episode_id")
    if scene_id is None or episode_id is None:
        return None
    derived_key = build_episode_key(extract_scene_id(str(scene_id)), episode_id)
    return summary_lookup.get(derived_key)


def enrich_record_from_summary(record: Dict[str, Any], summary_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not summary_lookup:
        return record
    summary_record = resolve_summary_record(record, summary_lookup)
    if summary_record is None:
        return record
    enriched = dict(record)
    plan_steps = normalize_plan_steps(summary_record.get("plan", []))
    if plan_steps:
        enriched["plan"] = plan_steps
    if not str(enriched.get("instruction", "")).strip():
        enriched["instruction"] = str(summary_record.get("instruction", ""))
    return enriched


def build_reasoning_config(reasoning_effort: str) -> Dict[str, Any]:
    effort = str(reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    if effort == "none":
        return {"reasoning_effort": "none"}
    return {"reasoning_effort": effort} # fixed by qianli


def encode_image_content(image_path: Path) -> Dict[str, Any]:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


def select_stride_paths(paths: List[str], stride: int) -> List[str]:
    if not paths:
        return []
    stride = max(1, int(stride))
    selected = [paths[idx] for idx in range(0, len(paths), stride)]
    if selected[-1] != paths[-1]:
        selected.append(paths[-1])
    return selected


def build_gt_rgb_dir_path(gt_image_root: Path, record: Dict[str, Any]) -> Path:
    scene_id = str(record.get("scene_id", "")).strip()
    episode_id = int(record["episode_id"])
    return gt_image_root / f"{scene_id}_r2r_{episode_id:06d}"


def build_gt_video_path(gt_video_root: Path, record: Dict[str, Any]) -> Path:
    return build_gt_rgb_dir_path(gt_video_root, record) / "trajectory.mp4"


def build_tmp_rgb_cache_dir(record: Dict[str, Any]) -> Path:
    return Path(tempfile.gettempdir()) / "thinkvln_watcher_frames" / f"{str(record.get('scene_id', '')).strip()}_r2r_{int(record['episode_id']):06d}"


def split_r2r_frame(frame, fixed_rgb_width: int = 640, map_left_crop: int = 0) -> Tuple[Any, Any]:
    _, width = frame.shape[:2]
    if width < fixed_rgb_width + map_left_crop:
        raise ValueError(
            f"frame width {width} is less than fixed RGB width {fixed_rgb_width} + map left crop {map_left_crop}"
        )
    rgb_frame = frame[:, :fixed_rgb_width]
    map_frame = frame[:, fixed_rgb_width + map_left_crop :]
    return rgb_frame, map_frame


def ensure_episode_frames_from_video(gt_video_root: Path, record: Dict[str, Any]) -> Path:
    rgb_dir = build_tmp_rgb_cache_dir(record)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    video_path = build_gt_video_path(gt_video_root, record)
    if not video_path.exists():
        raise FileNotFoundError(f"GT video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open GT video: {video_path}")
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frames_total > 0:
        complete = True
        for frame_idx in range(frames_total):
            if not (rgb_dir / f"{frame_idx:06d}_rgb.jpg").exists() or not (rgb_dir / f"{frame_idx:06d}_map.jpg").exists():
                complete = False
                break
        if complete:
            cap.release()
            return rgb_dir

    try:
        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break
            rgb_path = rgb_dir / f"{frame_idx:06d}_rgb.jpg"
            map_path = rgb_dir / f"{frame_idx:06d}_map.jpg"
            rgb_frame, map_frame = split_r2r_frame(frame)
            if not cv2.imwrite(str(rgb_path), rgb_frame):
                raise RuntimeError(f"Failed to write RGB frame: {rgb_path}")
            if not cv2.imwrite(str(map_path), map_frame):
                raise RuntimeError(f"Failed to write map frame: {map_path}")
            frame_idx += 1
    finally:
        cap.release()
    return rgb_dir


def extract_gt_history_rgb_paths(gt_video_root: Path, record: Dict[str, Any], stride: int) -> List[Path]:
    pivot_frame = max(0, int(record.get("pivot_frame", 0)))
    frame_indices = select_stride_paths([str(i) for i in range(pivot_frame + 1)], stride)
    rgb_dir = ensure_episode_frames_from_video(gt_video_root, record)
    rgb_paths: List[Path] = []
    for frame_idx in (int(x) for x in frame_indices):
        rgb_path = rgb_dir / f"{frame_idx:06d}_rgb.jpg"
        if not rgb_path.exists():
            raise FileNotFoundError(f"GT RGB frame not found after extraction: {rgb_path}")
        rgb_paths.append(rgb_path)
    if not rgb_paths:
        raise ValueError(f"No GT history RGB frames extracted from video: {build_gt_video_path(gt_video_root, record)}")
    return rgb_paths


def extract_gt_history_image_contents(gt_video_root: Path, record: Dict[str, Any], stride: int) -> List[Dict[str, Any]]:
    return [encode_image_content(path) for path in extract_gt_history_rgb_paths(gt_video_root, record, stride)]


def request_json_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_retries: int,
    request_timeout: float,
    log_prefix: str,
    reasoning_effort: str,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=float(request_timeout),
                extra_body=build_reasoning_config(reasoning_effort), # fixed by qianli
            )
            content = response.choices[0].message.content or ""
            return extract_json_object(content)
        except Exception as exc:
            last_error = exc
            prefix = f"{log_prefix} " if log_prefix else ""
            print(f"{prefix}request failed attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def split_plan_state(record: Dict[str, Any], plan_steps: List[str]) -> Tuple[List[str], List[str], List[str]]:
    if not plan_steps:
        active_step = str(record.get("subtask_text", "")).strip()
        return [], [active_step] if active_step else [], []
    subtask_id = max(1, int(record.get("subtask_id", 1)))
    index = min(subtask_id - 1, len(plan_steps) - 1)
    return plan_steps[:index], [plan_steps[index]], plan_steps[index + 1 :]


def format_plan_section(steps: List[str]) -> str:
    if not steps:
        return "- None."
    return "\n".join(f"- {step}" for step in steps)


def build_memory_start_messages(record: Dict[str, Any], history_images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    user_text = (
        "History\n"
        f"- Pivot frame: {int(record.get('pivot_frame', 0))}\n"
        "- Images are sampled from episode start to the pivot in time order.\n"
        "- First image = episode start. Last image = pivot.\n"
        "- Write memory_start as exactly three short semicolon-separated fragments.\n"
        "- Use this exact order: traj summary; current state; neutral status.\n"
        "- The first fragment must summarize the path already traveled before the pivot state.\n"
        "- The third fragment must stay neutral, such as active step in progress, still on current step, or approach still ongoing.\n"
        "- Do not use ready for next step or task complete.\n"
        "- Make it the same format that memory_end will update later.\n"
        "- Keep only useful progress that still matters.\n"
        "- Make the current pivot state explicit.\n"
        "- Do not narrate frame by frame."
    )
    return [
        {"role": "system", "content": MEMORY_START_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": user_text}] + history_images},
    ]


def build_rollout_messages(record: Dict[str, Any], memory_start: str, image_paths: List[Path]) -> List[Dict[str, Any]]:
    plan_steps = normalize_plan_steps(record.get("plan", []))
    done_steps, active_steps, pending_steps = split_plan_state(record, plan_steps)
    user_text = (
        "Plan state\n"
        "Done\n"
        f"{format_plan_section(done_steps)}\n"
        "Active\n"
        f"{format_plan_section(active_steps)}\n"
        "Pending\n"
        f"{format_plan_section(pending_steps)}\n"
        "State\n"
        f"- Memory start: {memory_start}\n"
        f"- Rollout actions: {record['actions']}\n"
        "Decision\n"
        "- Set done=true only if the active step reaches a natural handoff by rollout end.\n"
        "- The rollout end must also be a good starting point for the next subtask.\n"
        "- Do not hand off early just because the active step looks mostly complete.\n"
        "- If the next pending step is a turn, judge whether the robot has actually reached the turning point.\n"
        "- If the active step is a turn, judge it together with the next pending step and only hand off once the robot is aligned for that next movement.\n"
        "- If the active step is still the right step, set done=false.\n"
        "- With done=false, next_subtask should continue the active step or give a short recovery step.\n"
        "- With done=true, next_subtask should describe the promoted pending step, or stop if pending is empty.\n"
        "Memory\n"
        "- Rewrite memory_start into a new cumulative watcher memory.\n"
        "- Keep only the still-relevant part of memory_start.\n"
        "- Write memory_end as a direct update of memory_start.\n"
        "- Prefer extending memory_start forward with the new observation and then compressing if needed.\n"
        "- Do not reduce memory_end to only the final frame.\n"
        "- Write memory_end as exactly three short semicolon-separated fragments.\n"
        "- Use this exact order: traj summary; current state; task status.\n"
        "- The first fragment must summarize the path already traveled before the final state.\n"
        "- The third fragment must say whether the step is ongoing, ready for next step, or task complete.\n"
        "- Sentence fragments are allowed."
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    content.extend(encode_image_content(path) for path in image_paths)
    return [
        {"role": "system", "content": ROLLOUT_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def annotate_sample(
    client: OpenAI,
    model: str,
    gt_video_root: Path,
    bundle_root: Path,
    record: Dict[str, Any],
    image_stride: int,
    request_timeout: float,
    max_retries: int,
    reasoning_effort: str,
) -> Dict[str, Any]:
    sample_id = str(record.get("sample_id", ""))
    print(f"[annotate] sample={sample_id} stage=prepare_history")
    history_rgb_paths = extract_gt_history_rgb_paths(gt_video_root, record, image_stride)
    history_images = [encode_image_content(path) for path in history_rgb_paths]
    print(f"[annotate] sample={sample_id} stage=memory_start images={len(history_images)} timeout={request_timeout}s")
    memory_payload = request_json_completion(
        client=client,
        model=model,
        messages=build_memory_start_messages(record, history_images),
        max_retries=max_retries,
        request_timeout=request_timeout,
        log_prefix=f"[annotate] sample={sample_id} stage=memory_start",
        reasoning_effort=reasoning_effort,
    )
    memory_start = str(memory_payload["memory_start"]).strip()
    if not memory_start:
        raise ValueError("memory_start is empty")

    base_image_path = str(record["base_image_path"])
    selected_relpaths = select_stride_paths([str(p) for p in record.get("rollout_image_relpaths", [])], image_stride)
    selected_paths = [resolve_bundle_image_path(bundle_root, base_image_path, relpath) for relpath in selected_relpaths]
    print(f"[annotate] sample={sample_id} stage=memory_end images={len(selected_paths)} timeout={request_timeout}s")
    rollout_payload = request_json_completion(
        client=client,
        model=model,
        messages=build_rollout_messages(record, memory_start, selected_paths),
        max_retries=max_retries,
        request_timeout=request_timeout,
        log_prefix=f"[annotate] sample={sample_id} stage=memory_end",
        reasoning_effort=reasoning_effort,
    )

    done = rollout_payload.get("done")
    next_subtask = str(rollout_payload["next_subtask"]).strip()
    memory_end = str(rollout_payload["memory_end"]).strip()
    if not isinstance(done, bool):
        raise ValueError(f"done must be boolean: {done!r}")
    if not next_subtask:
        raise ValueError("next_subtask is empty")
    if not memory_end:
        raise ValueError("memory_end is empty")
    return {
        "sample_id": str(record["sample_id"]),
        "memory_start": memory_start,
        "done": done,
        "next_subtask": next_subtask,
        "memory_end": memory_end,
        "gt_history_images": history_images,
        "rollout_images": [encode_image_content(path) for path in selected_paths],
    }


def build_persisted_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": str(annotation["sample_id"]),
        "memory_start": str(annotation["memory_start"]),
        "done": bool(annotation["done"]),
        "next_subtask": str(annotation["next_subtask"]),
        "memory_end": str(annotation["memory_end"]),
    }


def run_text_ping(client: OpenAI, model: str) -> None:
    payload = request_json_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return {\"ok\": true}."},
        ],
        max_retries=3,
        request_timeout=60.0,
        log_prefix="[sanity] text",
        reasoning_effort="none",
    )
    if payload.get("ok") is not True:
        raise ValueError("text-only sanity check failed")


def run_multimodal_ping(client: OpenAI, model: str, pivot_image_path: Path) -> None:
    payload = request_json_completion(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return {\"ok\": true} after reading this image."},
                    encode_image_content(pivot_image_path),
                ],
            },
        ],
        max_retries=3,
        request_timeout=60.0,
        log_prefix="[sanity] multimodal",
        reasoning_effort="none",
    )
    if payload.get("ok") is not True:
        raise ValueError("multimodal sanity check failed")


def annotate_manifest(args: argparse.Namespace) -> int:
    if not args.api_key:
        raise ValueError("api key is empty, set OPENROUTER_API_KEY or pass --api_key")
    client = OpenAI(base_url=args.api_base_url, api_key=args.api_key)

    manifest_rows = load_jsonl(args.manifest_file)
    summary_lookup: Dict[str, Dict[str, Any]] = {}
    if args.summary_full_path is not None:
        try:
            summary_lookup = load_summary_full(args.summary_full_path)
        except FileNotFoundError:
            if not args.deploy_mode:
                raise
            print(f"[annotate] summary_full missing in deploy_mode, continue: {args.summary_full_path}")

    existing = load_jsonl_by_key(args.output_file, "sample_id") if args.resume else {}
    pending_rows = [row for row in manifest_rows if str(row.get("sample_id", "")) and str(row["sample_id"]) not in existing]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(pending_rows)
    if args.max_samples is not None:
        pending_rows = pending_rows[: max(0, int(args.max_samples))]
    if not pending_rows:
        return 0

    print(
        f"[annotate] pending_samples={len(pending_rows)} image_stride={args.image_stride} model={args.model_name} "
        f"workers={args.max_workers} reasoning_effort={args.reasoning_effort}"
    )

    if not args.skip_sanity_check and not args.deploy_mode:
        first_row = pending_rows[0]
        first_pivot = resolve_bundle_image_path(
            args.bundle_root,
            str(first_row["base_image_path"]),
            str(first_row["pivot_image_relpath"]),
        )
        print("[annotate] sanity_check=text")
        run_text_ping(client, args.model_name)
        print("[annotate] sanity_check=multimodal")
        run_multimodal_ping(client, args.model_name, first_pivot)

    results: List[Tuple[int, Dict[str, Any]]] = []
    skipped_missing_gt = 0

    max_workers = max(1, int(args.max_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map: Dict[concurrent.futures.Future, Tuple[int, Dict[str, Any], Dict[str, Any]]] = {}
        for idx, row in enumerate(pending_rows, start=1):
            enriched_row = enrich_record_from_summary(row, summary_lookup)
            future = pool.submit(
                annotate_sample,
                client,
                args.model_name,
                args.gt_video_root,
                args.bundle_root,
                enriched_row,
                max(1, int(args.image_stride)),
                float(args.request_timeout),
                max(1, int(args.max_retries)),
                str(args.reasoning_effort),
            )
            future_map[future] = (idx, row, enriched_row)

        for future in concurrent.futures.as_completed(future_map):
            idx, row, enriched_row = future_map[future]
            sample_id = str(row.get("sample_id", ""))
            print(f"[annotate] progress={idx}/{len(pending_rows)} sample={sample_id}")
            try:
                annotation = future.result()
            except FileNotFoundError:
                if not args.skip_missing_gt:
                    raise
                skipped_missing_gt += 1
                print(f"[annotate] sample={sample_id} skipped=missing_gt")
                continue
            except Exception as exc:
                print(f"[annotate] sample={sample_id} status=failed error={type(exc).__name__}: {exc}")
                continue

            persisted = build_persisted_annotation(annotation)
            append_jsonl(args.output_file, persisted)
            written = len(results) + 1
            print(f"[annotate] sample={row.get('sample_id', '')} status=written total_written={written}")
            results.append((idx, row))

    results.sort(key=lambda item: item[0])
    if skipped_missing_gt:
        print(f"watcher annotations skipped_missing_gt: {skipped_missing_gt}")
    return len(results)


def main() -> None:
    args = parse_args()
    written = annotate_manifest(args)
    print(f"watcher annotations written: {written}")


if __name__ == "__main__":
    main()
