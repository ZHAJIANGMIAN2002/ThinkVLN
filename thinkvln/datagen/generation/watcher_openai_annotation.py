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

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "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"),
)

# client = OpenAI(
#     base_url="http://localhost:11451/v1",
#     api_key="EMPTY",
# )
MODEL_NAME = 'qwen/qwen3.5-397b-a17b'
DEFAULT_REASONING_EFFORT = "none"


MEMORY_START_SYSTEM_PROMPT = """You are generating running watcher memory for a navigation agent.

Return JSON only:
{"memory_start":"..."}

Goal:
Compress the episode prefix from the episode start up to the pivot into one short watcher state that will be used for future decision-making.

Definition:
memory_start is not a trajectory summary. It is a compact running state that tells a future watcher:
1. what important progress has already been completed and still remains relevant,
2. what the current state is at the pivot,
3. what subtask or immediate goal is still active,
4. what recent failure fact should be remembered, only if it still matters.

Content requirements:
- Preserve only progress that is still useful for understanding the current state or the next decision.
- Describe the current position or orientation only through stable, task-relevant landmarks when possible.
- Make the active unfinished goal explicit.
- Mention a failure clue only if it is visible, recent, and important for future recovery.

Rules:
- Output JSON only.
- memory_start must be exactly 1 sentence.
- Use neutral declarative style.
- Do not use first person.
- Write a watcher state update, not a frame-by-frame narration.
- Do not list actions, count turns, or describe intermediate steps.
- Do not restate the full instruction or merely paraphrase the subtask text.
- Prefer stable, subtask-relevant landmarks over incidental details such as generic walls unless they are necessary.
- Do not mention image order, uncertainty, formatting, or missing information.
- Keep the sentence compact, information-dense, and directly useful for the next watcher decision.
"""



ROLLOUT_SYSTEM_PROMPT = """You are labeling a rollout span for a navigation watcher and updating its running memory.

Return JSON only:
{"label":"PROCEED|RESUME|FAIL","memory_end":"..."}

Overall goal:
Use memory_start together with the rollout observations to decide whether the current subtask has been completed, should continue, or has clearly failed, and then rewrite the running watcher memory for the rollout end state.

Decision procedure:
1. Choose PROCEED if the current subtask is completed by the end of the rollout, so the next subtask should begin.
2. Otherwise choose FAIL if the rollout ends clearly off-route, stalled, reversed, looping, wrongly stopped, or in a wrong area for the current subtask.
3. Otherwise choose RESUME.

Label meanings:
- PROCEED: the current subtask is completed by rollout end.
- RESUME: the current subtask is not completed, but the rollout still makes usable progress and remains broadly on track.
- FAIL: the current subtask is not completed and the rollout is clearly not usable as normal progress.

Definition of memory_end:
memory_end is the updated running watcher memory after this rollout.
It is not a trajectory summary and not a chain-of-thought explanation.
It should rewrite memory_start into a new compact state by:
1. preserving only still-relevant prior progress,
2. adding the new progress that remains relevant,
3. describing the current end state,
4. stating the unfinished goal, or the next goal if label is PROCEED,
5. mentioning one brief failure clue only if needed.

Content requirements:
- Keep earlier progress only when it is still useful for understanding the current state or the next decision.
- Drop obsolete details that no longer matter.
- Make the end state explicit.
- If label is PROCEED, make clear that the current subtask is complete and state the next immediate goal when possible.
- If label is RESUME, make clear what remains unfinished.
- If label is FAIL, make clear the wrong end state or the main failure clue.

Rules:
- Output JSON only.
- memory_end must be exactly 1 sentence.
- Use neutral declarative style.
- Do not use first person.
- Do not narrate step-by-step actions.
- Do not count turns or list intermediate moves.
- Do not output a chain-of-thought or detailed explanation.
- Do not merely copy memory_start; rewrite it using the rollout result.
- Do not mention image order, uncertainty, formatting, or missing information.
- Keep the sentence compact, cumulative, and directly useful for the next watcher decision.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate watcher rollout manifests with OpenAI.")
    parser.add_argument("--gt_image_root", type=Path, required=True)
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--manifest_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
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


def build_memory_start_messages(
    record: Dict[str, Any],
    history_images: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    user_text = (
    "Task\n"
    f"- Instruction: {record['instruction']}\n"
    f"- Current subtask id: {record['subtask_id']}\n"
    f"- Current subtask: {record['subtask_text']}\n"
    f"- Pivot frame: {int(record.get('pivot_frame', 0))}\n"
    "Context\n"
    "- Images are sampled in time order from the episode start to the pivot.\n"
    "- The first image is the episode start and the last image is the pivot state.\n"
    "Memory target\n"
    "- Write memory_start as the running watcher memory at the pivot.\n"
    "- Keep only completed progress that is still relevant.\n"
    "- Make the current pivot state explicit.\n"
    "- State the current unfinished subtask or immediate goal.\n"
    "- Mention one recent failure clue only if it still matters.\n"
    "Output rules\n"
    "- Write a watcher state update, not a trajectory summary.\n"
    "- Do not narrate frame by frame.\n"
    "- Do not list actions or count turns.\n"
    "- Do not merely restate the instruction or subtask text.\n"
    "- Keep memory_start to exactly 1 short sentence.\n"
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
    user_text = (
    "Task\n"
    f"- Instruction: {record['instruction']}\n"
    f"- Current subtask id: {record['subtask_id']}\n"
    f"- Current subtask: {record['subtask_text']}\n"
    f"- Memory start: {memory_start}\n"
    f"- Rollout actions: {record['actions']}\n"
    "Context\n"
    "- The rollout images are in time order from the pivot to the rollout end.\n"
    "- The last rollout image is the final state for label judgment.\n"
    "Decision target\n"
    "- First decide whether the current subtask is completed by the end of the rollout.\n"
    "- If the current subtask is completed, choose PROCEED.\n"
    "- Otherwise, if the rollout ends clearly off-route, stalled, reversed, looping, wrongly stopped, or in a wrong area for the current subtask, choose FAIL.\n"
    "- Otherwise choose RESUME.\n"
    "Memory target\n"
    "- Rewrite memory_start into a new running watcher memory for the rollout end state.\n"
    "- Keep only earlier progress that is still relevant.\n"
    "- Add new rollout progress only if it remains useful for understanding the current state or next decision.\n"
    "- Make the current end state explicit.\n"
    "- If label is PROCEED, state that the current subtask is complete and give the next immediate goal when possible.\n"
    "- If label is RESUME, state what remains unfinished.\n"
    "- If label is FAIL, state the wrong end state or one brief failure clue.\n"
    "Output rules\n"
    "- Do not narrate the rollout step by step.\n"
    "- Do not count turns or list intermediate actions.\n"
    "- Do not write a chain-of-thought explanation.\n"
    "- Keep memory_end to exactly 1 short sentence.\n"
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
    label = str(rollout_payload["label"]).strip().upper()
    memory_end = str(rollout_payload["memory_end"]).strip()
    if label not in {"RESUME", "PROCEED", "FAIL"}:
        raise ValueError(f"invalid label: {label}")
    if not memory_end:
        raise ValueError("memory_end is empty")
    return {
        "sample_id": str(record["sample_id"]),
        "memory_start": memory_start,
        "label": label,
        "memory_end": memory_end,
        "gt_history_images": history_images,
        "rollout_images": rollout_images,
    }


def build_persisted_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": str(annotation["sample_id"]),
        "memory_start": str(annotation["memory_start"]),
        "label": str(annotation["label"]),
        "memory_end": str(annotation["memory_end"]),
    }


def write_debug_html(debug_html_dir: Path, records: List[Dict[str, Any]]) -> None:
    debug_html_dir.mkdir(parents=True, exist_ok=True)
    sections: List[str] = []
    for record in records:
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
  <p><strong>Subtask:</strong> {html.escape(str(record.get("subtask_id", "")))} - {html.escape(str(record.get("subtask_text", "")))}</p>
  <p><strong>Actions:</strong> {html.escape(str(record.get("actions", [])))}</p>
  <p><strong>Memory Start:</strong> {html.escape(str(record.get("memory_start", "")))}</p>
  <p><strong>Label:</strong> {html.escape(str(record.get("label", "")))}</p>
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
        sample_id = str(row.get("sample_id", ""))
        print(f"[annotate] progress={idx}/{len(pending_rows)} sample={sample_id}")
        try:
            annotation = annotate_sample(
                client=client,
                model=MODEL_NAME,
                gt_image_root=args.gt_image_root,
                bundle_root=args.bundle_root,
                record=row,
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
                    "sample_id": str(row.get("sample_id", "")),
                    "instruction": str(row.get("instruction", "")),
                    "subtask_id": row.get("subtask_id"),
                    "subtask_text": str(row.get("subtask_text", "")),
                    "actions": row.get("actions", []),
                    "memory_start": annotation["memory_start"],
                    "label": annotation["label"],
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
