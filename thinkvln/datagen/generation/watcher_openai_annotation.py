from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from thinkvln.datagen.generation.watcher_utils import (
    append_jsonl,
    extract_json_object,
    load_jsonl,
    load_jsonl_by_key,
    resolve_bundle_image_path,
    select_label_image_relpaths,
)


MEMORY_START_SYSTEM_PROMPT = """You are generating watcher memory for a navigation agent.
Return JSON with one key: {"memory_start": "..."}.
Write 1-2 short sentences that keep only future-relevant state:
- important completed progress,
- the current unfinished subtask,
- recent failure only if clearly visible.
Do not narrate the full trajectory."""


ROLLOUT_SYSTEM_PROMPT = """You are labeling a rollout span for a navigation watcher.
Return JSON with exactly:
{"label":"RESUME|PROCEED|FAIL","memory_end":"..."}

Label definitions:
- PROCEED: the current subtask is completed and the next subtask should start.
- RESUME: the current subtask is not completed but the rollout is still broadly on track.
- FAIL: the rollout is clearly wrong, stuck, looping, or ends in the wrong place.

`memory_end` must be a short state update for the next watcher call, not a frame-by-frame narration."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate watcher rollout manifests with OpenAI.")
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--manifest_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--openai_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--label_image_stride", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ensure_openai_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    return api_key


def encode_image_content(image_path: Path) -> Dict[str, Any]:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def request_json_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    max_retries: int = 3,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return extract_json_object(content)
        except Exception as exc:  # pragma: no cover - retried in tests via fake client
            last_error = exc
    raise RuntimeError(f"OpenAI request failed after retries: {last_error}")


def build_memory_start_messages(record: Dict[str, Any], pivot_image_path: Path) -> List[Dict[str, Any]]:
    user_text = (
        f"Instruction: {record['instruction']}\n"
        f"Current subtask id: {record['subtask_id']}\n"
        f"Current subtask: {record['subtask_text']}\n"
        "Use the pivot image to summarize the current navigation state."
    )
    return [
        {"role": "system", "content": MEMORY_START_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                encode_image_content(pivot_image_path),
            ],
        },
    ]


def build_rollout_messages(
    record: Dict[str, Any],
    image_paths: List[Path],
    memory_start: str,
) -> List[Dict[str, Any]]:
    user_text = (
        f"Instruction: {record['instruction']}\n"
        f"Current subtask id: {record['subtask_id']}\n"
        f"Current subtask: {record['subtask_text']}\n"
        f"Memory start: {memory_start}\n"
        f"Trajectory summary: a={record['actions']}\n"
        "The first image is the pivot image. The remaining images are rollout RGB observations in time order."
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
    bundle_root: Path,
    record: Dict[str, Any],
    label_image_stride: int,
) -> Dict[str, Any]:
    base_image_path = str(record["base_image_path"])
    pivot_image_path = resolve_bundle_image_path(
        bundle_root,
        base_image_path,
        str(record["pivot_image_relpath"]),
    )
    memory_payload = request_json_completion(
        client,
        model,
        build_memory_start_messages(record, pivot_image_path),
    )
    memory_start = str(memory_payload["memory_start"]).strip()
    if not memory_start:
        raise ValueError("memory_start is empty")

    selected_relpaths = select_label_image_relpaths(
        str(record["pivot_image_relpath"]),
        [str(relpath) for relpath in record.get("rollout_image_relpaths", [])],
        label_image_stride,
    )
    selected_paths = [
        resolve_bundle_image_path(bundle_root, base_image_path, relpath)
        for relpath in selected_relpaths
    ]
    rollout_payload = request_json_completion(
        client,
        model,
        build_rollout_messages(record, selected_paths, memory_start),
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
    }


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
    api_key = ensure_openai_key()
    client = OpenAI(api_key=api_key)
    manifest_rows = load_jsonl(args.manifest_file)
    existing = load_jsonl_by_key(args.output_file, "sample_id") if args.resume else {}
    pending_rows = [
        row for row in manifest_rows
        if str(row.get("sample_id", "")) and str(row["sample_id"]) not in existing
    ]
    if not pending_rows:
        return 0

    first_row = pending_rows[0]
    first_pivot = resolve_bundle_image_path(
        args.bundle_root,
        str(first_row["base_image_path"]),
        str(first_row["pivot_image_relpath"]),
    )
    run_text_ping(client, args.openai_model)
    run_multimodal_ping(client, args.openai_model, first_pivot)

    written = 0
    for row in pending_rows:
        annotation = annotate_sample(
            client=client,
            model=args.openai_model,
            bundle_root=args.bundle_root,
            record=row,
            label_image_stride=max(1, int(args.label_image_stride)),
        )
        append_jsonl(args.output_file, annotation)
        written += 1
    return written


def main() -> None:
    args = parse_args()
    written = annotate_manifest(args)
    print(f"watcher annotations written: {written}")


if __name__ == "__main__":
    main()
