from __future__ import annotations

import argparse
import base64
import html
import mimetypes
from pathlib import Path
from typing import Any, Dict, List
import os

from openai import OpenAI

from thinkvln.datagen.generation.watcher_utils import (
    append_jsonl,
    extract_json_object,
    load_jsonl,
    load_jsonl_by_key,
    resolve_bundle_image_path,
)
from thinkvln.eval.close_eval_utils import (
    build_episode_key,
    extract_scene_id,
    load_summary_full,
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "EMPTY"),
)

# client = OpenAI(
#     base_url="http://localhost:11451/v1",
#     api_key="EMPTY",
# )
MODEL_NAME = 'qwen/qwen3.5-397b-a17b'
DEFAULT_REASONING_EFFORT = "none"


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


# ROLLOUT_SYSTEM_PROMPT = """You update watcher memory and choose the robot's next subtask.

# Return JSON only:
# {"done":true,"next_subtask":"...","memory_end":"..."}

# There is no map. Judge progress only from memory_start, rollout frames, rollout actions, and plan state.

# Done flag rules:
# - Set done=true only if both conditions hold by rollout end:
#   1. The current active step reaches a natural handoff.
#   2. The rollout end is already a good starting point for the next subtask.
# - Set done=false if the current active step should remain active, including recovery cases.
# - Do not hand off early just because the active step looks mostly complete.
# - If no pending step remains after done=true, the next_subtask should be stop only when the final stopping condition is met.
# - done=true means the current active step moves to done and the first pending step becomes active.

# Transition logic table:
# | Meta-Action | Current Behavior | Switch when | Stay on current step when |
# | :--- | :--- | :--- | :--- |
# | Turn | Rotating into a new heading. | Orientation is stably aligned with the new path and the next movement can start now. | Still rotating, still correcting angle, or not yet aligned for the next move. |
# | Region Transition | Passing through a doorway or room boundary. | The robot has clearly crossed into the next region. | The doorway or boundary is still ahead, straddled, or only partially crossed. |
# | Visual Approach | Closing in on a visible target object or stop point. | The target or stopping point is immediate and ready for the final settle. | The target is still a short approach away. |
# | General Cruise | Moving along a hall, room edge, or open route toward a future event. | The robot has reached the structural trigger for the next step, such as an intersection, doorway, corner, or hall end. | The trigger point is still ahead, even if the current route looks mostly complete. |
# | Stop | Settling into the intended stopping position. | The robot is already in the intended stopping position. | The robot is still adjusting position or orientation. |

# Next subtask rules:
# - next_subtask must be short and actionable.
# - Write it as an imperative instruction.
# - If done=false, continue or refine the current active step, or write a short recovery step.
# - If done=true, describe the new active step after transition, or use "stop" if nothing is pending.

# Memory_end rules:
# - memory_end must be a single line with exactly three short semicolon-separated fragments.
# - Use this exact order: traj summary; current state; task status
# - The first fragment must summarize the path already traveled before the final state.
# - The second fragment must state where the robot is now.
# - The third fragment must say whether the step is ongoing, ready for next step, or task complete.
# - Sentence fragments are allowed.
# - Start from past progress, not the final frame.
# - Write memory_end as a direct update of memory_start.
# - Keep only still-relevant past progress.
# - Prefer stable, task-relevant landmarks over incidental details.
# - Do not explain why next_subtask was chosen.
# - Do not mention image order, uncertainty, or formatting.

# Good examples:
# - {"done":false,"next_subtask":"continue toward the intersection before turning right","memory_end":"Left the bedroom and followed the hall; approaching the dining-room opening, not at the turn yet; step ongoing"}
# - {"done":false,"next_subtask":"finish turning right toward the hallway","memory_end":"Reached the hallway entrance from the room; mid-turn toward the hallway; step ongoing"}
# - {"done":true,"next_subtask":"enter the bathroom","memory_end":"Cleared the dining area and reached the hall entrance; aligned with the bathroom approach; ready for next step"}
# - {"done":true,"next_subtask":"stop","memory_end":"Entered the bathroom and approached the sink; beside the sink in the stopping spot; task complete"}

# Bad examples:
# - {"done":true,"next_subtask":"turn right","memory_end":"At the intersection; ready for next step; turned down the hall"}
# - {"done":"RESUME","next_subtask":"continue","memory_end":"The agent is near the doorway."}
# - {"done":false,"next_subtask":"The rollout failed","memory_end":"This rollout failed because the agent is off-route."}
# """

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
    parser = argparse.ArgumentParser(description="Annotate watcher rollout manifests with OpenAI.")
    parser.add_argument("--gt_image_root", type=Path, required=True)
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--manifest_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--summary_full_path", type=Path, default=None)
    parser.add_argument("--image_stride", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--debug_html_dir", type=Path, default=None)
    parser.add_argument("--request_timeout", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=DEFAULT_REASONING_EFFORT,
        choices=["none", "low", "medium", "high"],
    )
    parser.add_argument("--skip_missing_gt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def encode_image_content(image_path: Path) -> Dict[str, Any]:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def encode_jpeg_bytes(jpeg_bytes: bytes) -> Dict[str, Any]:
    encoded = base64.b64encode(jpeg_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


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


def extract_gt_history_image_contents(
    gt_image_root: Path,
    record: Dict[str, Any],
    stride: int,
) -> List[Dict[str, Any]]:
    pivot_frame = max(0, int(record.get("pivot_frame", 0)))
    frame_indices = select_stride_paths(
        [str(frame_idx) for frame_idx in range(pivot_frame + 1)],
        stride,
    )
    rgb_dir = build_gt_rgb_dir_path(gt_image_root, record)
    if not rgb_dir.exists():
        raise FileNotFoundError(f"GT RGB directory not found: {rgb_dir}")
    encoded_images: List[Dict[str, Any]] = []
    for frame_idx in (int(frame_idx) for frame_idx in frame_indices):
        rgb_path = rgb_dir / f"{frame_idx:06d}_rgb.jpg"
        if not rgb_path.exists():
            raise FileNotFoundError(f"GT RGB frame not found: {rgb_path}")
        encoded_images.append(encode_image_content(rgb_path))

    if not encoded_images:
        raise ValueError(f"No GT history RGB frames extracted from: {rgb_dir}")
    return encoded_images


def request_json_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_retries: int = 3,
    request_timeout: float = 180.0,
    log_prefix: str = "",
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=float(request_timeout),
                extra_body={"reasoning": build_reasoning_config(reasoning_effort)},
            )
            content = response.choices[0].message.content or ""
            return extract_json_object(content)
        except Exception as exc:  # pragma: no cover - retried in tests via fake client
            last_error = exc
            prefix = f"{log_prefix} " if log_prefix else ""
            print(
                f"{prefix}request failed attempt {attempt}/{max_retries}: {type(exc).__name__}: {exc}"
            )
    raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def build_reasoning_config(reasoning_effort: str) -> Dict[str, Any]:
    effort = str(reasoning_effort or DEFAULT_REASONING_EFFORT).strip().lower()
    if effort == "none":
        return {"effort": "none"}
    return {"effort": effort, "exclude": True}


def normalize_plan_steps(plan: Any) -> List[str]:
    if isinstance(plan, list):
        return [str(step).strip() for step in plan if str(step).strip()]
    if isinstance(plan, str):
        return [line.strip() for line in plan.splitlines() if line.strip()]
    return []


def resolve_current_plan_step(record: Dict[str, Any], plan_steps: List[str]) -> str:
    if plan_steps:
        subtask_id = max(1, int(record.get("subtask_id", 1)))
        index = min(subtask_id - 1, len(plan_steps) - 1)
        return plan_steps[index]
    return str(record.get("subtask_text", "")).strip()


def split_plan_state(record: Dict[str, Any], plan_steps: List[str]) -> tuple[List[str], List[str], List[str]]:
    if not plan_steps:
        active_step = str(record.get("subtask_text", "")).strip()
        return [], [active_step] if active_step else [], []
    subtask_id = max(1, int(record.get("subtask_id", 1)))
    index = min(subtask_id - 1, len(plan_steps) - 1)
    return plan_steps[:index], [plan_steps[index]], plan_steps[index + 1:]


def format_plan_section(steps: List[str]) -> str:
    if not steps:
        return "- None."
    return "\n".join(f"- {step}" for step in steps)


def resolve_summary_record(
    record: Dict[str, Any],
    summary_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    episode_key = str(record.get("episode_key", "")).strip()
    if episode_key and episode_key in summary_lookup:
        return summary_lookup[episode_key]
    scene_id = record.get("scene_id")
    episode_id = record.get("episode_id")
    if scene_id is None or episode_id is None:
        return None
    derived_key = build_episode_key(extract_scene_id(str(scene_id)), episode_id)
    return summary_lookup.get(derived_key)


def enrich_record_from_summary(
    record: Dict[str, Any],
    summary_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
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


def build_memory_start_messages(
    record: Dict[str, Any],
    history_images: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
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
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}] + history_images,
        },
    ]


def build_rollout_messages(
    record: Dict[str, Any],
    image_paths: List[Path],
    memory_start: str,
) -> List[Dict[str, Any]]:
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
    gt_image_root: Path,
    bundle_root: Path,
    record: Dict[str, Any],
    image_stride: int,
    request_timeout: float = 180.0,
    max_retries: int = 4,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> Dict[str, Any]:
    sample_id = str(record.get("sample_id", ""))
    print(f"[annotate] sample={sample_id} stage=prepare_history")
    history_images = extract_gt_history_image_contents(gt_image_root, record, image_stride)
    base_image_path = str(record["base_image_path"])
    print(
        f"[annotate] sample={sample_id} stage=memory_start images={len(history_images)} timeout={request_timeout}s"
    )
    memory_payload = request_json_completion(
        client,
        model,
        build_memory_start_messages(record, history_images),
        max_retries=max_retries,
        request_timeout=request_timeout,
        log_prefix=f"[annotate] sample={sample_id} stage=memory_start",
        reasoning_effort=reasoning_effort,
    )
    memory_start = str(memory_payload["memory_start"]).strip()
    if not memory_start:
        raise ValueError("memory_start is empty")

    selected_relpaths = select_stride_paths(
        [str(relpath) for relpath in record.get("rollout_image_relpaths", [])],
        image_stride,
    )
    selected_paths = [
        resolve_bundle_image_path(bundle_root, base_image_path, relpath)
        for relpath in selected_relpaths
    ]
    rollout_images = [encode_image_content(path) for path in selected_paths]
    print(
        f"[annotate] sample={sample_id} stage=memory_end images={len(rollout_images)} timeout={request_timeout}s"
    )
    rollout_payload = request_json_completion(
        client,
        model,
        build_rollout_messages(record, selected_paths, memory_start),
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
        "rollout_images": rollout_images,
    }


def build_persisted_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": str(annotation["sample_id"]),
        "memory_start": str(annotation["memory_start"]),
        "done": bool(annotation["done"]),
        "next_subtask": str(annotation["next_subtask"]),
        "memory_end": str(annotation["memory_end"]),
    }


def write_debug_html(debug_html_dir: Path, records: List[Dict[str, Any]]) -> None:
    debug_html_dir.mkdir(parents=True, exist_ok=True)
    sections: List[str] = []
    for record in records:
        plan_steps = normalize_plan_steps(record.get("plan", []))
        done_steps, active_steps, pending_steps = split_plan_state(record, plan_steps)
        gt_images = "".join(
            f'<img src="{html.escape(item["image_url"]["url"])}" alt="gt" loading="lazy" />'
            for item in record.get("gt_history_images", [])
        )
        rollout_images = "".join(
            f'<img src="{html.escape(item["image_url"]["url"])}" alt="rollout" loading="lazy" />'
            for item in record.get("rollout_images", [])
        )
        sections.append(
            f"""
<section class="sample">
  <h2>{html.escape(str(record.get("sample_id", "")))}</h2>
  <p><strong>Instruction:</strong> {html.escape(str(record.get("instruction", "")))}</p>
  <p><strong>Actions:</strong> {html.escape(str(record.get("actions", [])))}</p>
  <p><strong>Plan State:</strong></p>
  <p><strong>Done Plan:</strong> {html.escape(str(done_steps))}</p>
  <p><strong>Active Plan:</strong> {html.escape(str(active_steps))}</p>
  <p><strong>Pending Plan:</strong> {html.escape(str(pending_steps))}</p>
  <p><strong>Memory Start:</strong> {html.escape(str(record.get("memory_start", "")))}</p>
  <p><strong>Done:</strong> {html.escape(str(record.get("done", "")))}</p>
  <p><strong>Next Subtask:</strong> {html.escape(str(record.get("next_subtask", "")))}</p>
  <p><strong>Memory End:</strong> {html.escape(str(record.get("memory_end", "")))}</p>
  <div class="strip">{gt_images}</div>
  <div class="strip">{rollout_images}</div>
</section>
""".strip()
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Watcher Annotation Debug</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; line-height: 1.4; }}
    .sample {{ border-top: 1px solid #ccc; padding: 16px 0; }}
    .strip {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 16px; }}
    img {{ width: 180px; height: auto; border: 1px solid #ddd; }}
    p {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <h1>Watcher Annotation Debug</h1>
  {''.join(sections)}
</body>
</html>
"""
    (debug_html_dir / "index.html").write_text(html_text, encoding="utf-8")


def run_text_ping(client: OpenAI, model: str) -> None:
    payload = request_json_completion(
        client,
        model,
        [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return {\"ok\": true}."},
        ],
    )
    if payload.get("ok") is not True:
        raise ValueError("text-only sanity check failed")


def run_multimodal_ping(client: OpenAI, model: str, pivot_image_path: Path) -> None:
    payload = request_json_completion(
        client,
        model,
        [
            {"role": "system", "content": "Return JSON only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Return {\"ok\": true} after reading this image."},
                    encode_image_content(pivot_image_path),
                ],
            },
        ],
    )
    if payload.get("ok") is not True:
        raise ValueError("multimodal sanity check failed")


def annotate_manifest(args: argparse.Namespace) -> int:
    manifest_rows = load_jsonl(args.manifest_file)
    summary_full_path = getattr(args, "summary_full_path", None)
    summary_lookup = load_summary_full(str(summary_full_path)) if summary_full_path is not None else {}
    existing = load_jsonl_by_key(args.output_file, "sample_id") if args.resume else {}
    pending_rows = [
        row for row in manifest_rows
        if str(row.get("sample_id", "")) and str(row["sample_id"]) not in existing
    ]
    max_samples = getattr(args, "max_samples", None)
    if max_samples is not None:
        pending_rows = pending_rows[:max(0, int(max_samples))]
    if not pending_rows:
        return 0
    print(
        f"[annotate] pending_samples={len(pending_rows)} image_stride={args.image_stride} model={MODEL_NAME} reasoning_effort={getattr(args, 'reasoning_effort', DEFAULT_REASONING_EFFORT)}"
    )

    first_row = pending_rows[0]
    first_pivot = resolve_bundle_image_path(
        args.bundle_root,
        str(first_row["base_image_path"]),
        str(first_row["pivot_image_relpath"]),
    )
    print("[annotate] sanity_check=text")
    run_text_ping(client, MODEL_NAME)
    print("[annotate] sanity_check=multimodal")
    run_multimodal_ping(client, MODEL_NAME, first_pivot)

    written = 0
    skipped_missing_gt = 0
    debug_records: List[Dict[str, Any]] = []
    for idx, row in enumerate(pending_rows, start=1):
        enriched_row = enrich_record_from_summary(row, summary_lookup)
        sample_id = str(row.get("sample_id", ""))
        print(f"[annotate] progress={idx}/{len(pending_rows)} sample={sample_id}")
        try:
            annotation = annotate_sample(
                client=client,
                model=MODEL_NAME,
                gt_image_root=args.gt_image_root,
                bundle_root=args.bundle_root,
                record=enriched_row,
                image_stride=max(1, int(args.image_stride)),
                request_timeout=float(getattr(args, "request_timeout", 180.0)),
                max_retries=max(1, int(getattr(args, "max_retries", 4))),
                reasoning_effort=str(getattr(args, "reasoning_effort", DEFAULT_REASONING_EFFORT)),
            )
        except FileNotFoundError:
            if not getattr(args, "skip_missing_gt", False):
                raise
            skipped_missing_gt += 1
            print(f"[annotate] sample={sample_id} skipped=missing_gt")
            continue
        append_jsonl(args.output_file, build_persisted_annotation(annotation))
        written += 1
        print(f"[annotate] sample={sample_id} status=written total_written={written}")
        if getattr(args, "debug_html_dir", None) is not None:
            debug_records.append(
                {
                    "sample_id": str(enriched_row.get("sample_id", "")),
                    "instruction": str(enriched_row.get("instruction", "")),
                    "plan": enriched_row.get("plan", []),
                    "subtask_id": enriched_row.get("subtask_id"),
                    "subtask_text": str(enriched_row.get("subtask_text", "")),
                    "actions": enriched_row.get("actions", []),
                    "memory_start": annotation["memory_start"],
                    "done": annotation["done"],
                    "next_subtask": annotation["next_subtask"],
                    "memory_end": annotation["memory_end"],
                    "gt_history_images": annotation.get("gt_history_images", []),
                    "rollout_images": annotation.get("rollout_images", []),
                }
            )
    if skipped_missing_gt:
        print(f"watcher annotations skipped_missing_gt: {skipped_missing_gt}")
    debug_html_dir = getattr(args, "debug_html_dir", None)
    if debug_html_dir is not None and debug_records:
        write_debug_html(debug_html_dir, debug_records)
        print(f"[annotate] debug_html={debug_html_dir / 'index.html'}")
    return written


def main() -> None:
    args = parse_args()
    written = annotate_manifest(args)
    print(f"watcher annotations written: {written}")


if __name__ == "__main__":
    main()
