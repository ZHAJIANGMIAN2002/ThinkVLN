from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from openai import OpenAI

from thinkvln.datagen.generation.watcher_openai_annotation_deploy import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    append_jsonl,
    annotate_sample,
    enrich_record_from_summary,
    load_jsonl,
    load_jsonl_by_key,
    load_summary_full,
    normalize_plan_steps,
    resolve_bundle_image_path,
    split_plan_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual done annotation: AI writes memory, human judges done.")
    parser.add_argument("--gt_image_root", type=Path, required=True)
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--manifest_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--summary_full_path", type=Path, default=None)
    parser.add_argument("--work_dir", type=Path, default=Path("results/watcher_manual_review"))
    parser.add_argument("--image_stride", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request_timeout", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--reasoning_effort", type=str, default="high", choices=["none", "low", "medium", "high"])
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--api_base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--skip_missing_gt", action="store_true")
    return parser.parse_args()


def _write_rollout_video(video_path: Path, frame_paths: List[Path], fps: int = 4) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for rollout video generation")
    if not frame_paths:
        raise ValueError("empty rollout frames")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"failed reading frame: {frame_paths[0]}")
    h, w = first.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()


def _next_pending_step(enriched_row: Dict[str, Any]) -> str:
    plan_steps = normalize_plan_steps(enriched_row.get("plan", []))
    _, _, pending = split_plan_state(enriched_row, plan_steps)
    return pending[0] if pending else "stop"


def _manual_done_input() -> bool | None:
    while True:
        raw = input("done? [y/n/s(skip)/q(quit)]: ").strip().lower()
        if raw in {"y", "yes", "1"}:
            return True
        if raw in {"n", "no", "0"}:
            return False
        if raw in {"s", "skip"}:
            return None
        if raw in {"q", "quit"}:
            raise KeyboardInterrupt
        print("invalid input, use y/n/s/q")


def run_manual_done_annotation(args: argparse.Namespace) -> int:
    client = OpenAI(base_url=args.api_base_url, api_key=args.api_key)
    manifest_rows = load_jsonl(args.manifest_file)
    summary_lookup = load_summary_full(args.summary_full_path) if args.summary_full_path else {}
    existing = load_jsonl_by_key(args.output_file, "sample_id") if args.resume else {}

    pending_rows = [r for r in manifest_rows if str(r.get("sample_id", "")) and str(r["sample_id"]) not in existing]
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(pending_rows)
    if args.max_samples is not None:
        pending_rows = pending_rows[: max(0, int(args.max_samples))]
    if not pending_rows:
        print("no pending samples")
        return 0

    work_dir = args.work_dir
    video_dir = work_dir / "videos"
    html_dir = work_dir / "html"
    video_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    print(f"[manual] pending={len(pending_rows)} output={args.output_file}")
    for idx, row in enumerate(pending_rows, start=1):
        sample_id = str(row.get("sample_id", ""))
        enriched = enrich_record_from_summary(row, summary_lookup)
        print(f"\n[manual] {idx}/{len(pending_rows)} sample={sample_id}")

        try:
            ai = annotate_sample(
                client=client,
                model=args.model_name,
                gt_image_root=args.gt_image_root,
                bundle_root=args.bundle_root,
                record=enriched,
                image_stride=max(1, int(args.image_stride)),
                request_timeout=float(args.request_timeout),
                max_retries=max(1, int(args.max_retries)),
                reasoning_effort=str(args.reasoning_effort),
            )
        except FileNotFoundError as exc:
            if args.skip_missing_gt:
                print(f"[manual] skip missing gt: {exc}")
                continue
            raise

        rollout_frame_paths = [
            resolve_bundle_image_path(
                args.bundle_root,
                str(enriched["base_image_path"]),
                str(rel),
            )
            for rel in enriched.get("rollout_image_relpaths", [])
        ]
        mp4_path = video_dir / f"{sample_id}.mp4"
        _write_rollout_video(mp4_path, rollout_frame_paths)

        active_task = str(enriched.get("subtask_text", "")).strip() or "(unknown)"
        next_task = _next_pending_step(enriched)
        html_path = html_dir / f"{sample_id}.html"
        html_path.write_text(
            f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{sample_id}</title></head>
<body style="font-family: sans-serif; margin: 20px">
<h2>{sample_id}</h2>
<p><b>Instruction:</b> {enriched.get("instruction","")}</p>
<p><b>Current Active Task:</b> {active_task}</p>
<p><b>Next Pending Task:</b> {next_task}</p>
<p><b>AI memory_start:</b> {ai["memory_start"]}</p>
<p><b>AI memory_end:</b> {ai["memory_end"]}</p>
<p><b>AI suggested next_subtask:</b> {ai["next_subtask"]}</p>
<p><b>AI suggested done:</b> {ai["done"]}</p>
<video controls autoplay loop style="max-width: 920px; width: 100%">
  <source src="../videos/{sample_id}.mp4" type="video/mp4">
</video>
</body></html>
""",
            encoding="utf-8",
        )
        print(f"[manual] review html: {html_path}")
        print(f"[manual] rollout mp4: {mp4_path}")
        print("[manual] judge: if active task is completed and ready to switch, choose y; else n.")
        try:
            manual_done = _manual_done_input()
        except KeyboardInterrupt:
            print("\n[manual] interrupted by user")
            break
        if manual_done is None:
            print("[manual] skipped by user")
            continue

        payload = {
            "sample_id": sample_id,
            "manual_done": bool(manual_done),
            "done_ai": bool(ai["done"]),
            "done_match_ai": bool(manual_done) == bool(ai["done"]),
            "next_subtask": str(ai["next_subtask"]),
            "memory_start": str(ai["memory_start"]),
            "memory_end": str(ai["memory_end"]),
            "instruction": str(enriched.get("instruction", "")),
            "active_task": active_task,
            "pending_next_task": next_task,
            "review_html": str(html_path),
            "rollout_video": str(mp4_path),
        }
        append_jsonl(args.output_file, payload)
        written += 1
        print(f"[manual] written={written} manual_done={manual_done} ai_done={ai['done']}")

    print(f"[manual] total written: {written}")
    return written


def main() -> None:
    args = parse_args()
    run_manual_done_annotation(args)


if __name__ == "__main__":
    main()
