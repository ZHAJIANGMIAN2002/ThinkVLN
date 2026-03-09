import html
import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image


class EpisodeStepDebugger:
    def __init__(
        self,
        output_path: str,
        episode_key: str,
        enable_step_debug: bool = False,
        step_debug_format: str = "none",
    ):
        self.enable_step_debug = bool(enable_step_debug)
        self.step_debug_format = step_debug_format
        self.episode_key = str(episode_key)
        self.rows: List[Dict[str, Any]] = []
        self._checks_total = {
            "memory_frame_count_ok": 0,
            "memory_includes_past_subtask": 0,
            "progress_pass_through_ok": 0,
        }
        self._trace_handle = None
        self.debug_dir: Optional[str] = None
        self.images_dir: Optional[str] = None

        if not self.enable_step_debug:
            return

        safe_episode_key = self.episode_key.replace("/", "_")
        self.debug_dir = os.path.join(output_path, f"debug_{safe_episode_key}")
        self.images_dir = os.path.join(self.debug_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        trace_path = os.path.join(self.debug_dir, "step_trace.jsonl")
        self._trace_handle = open(trace_path, "w", encoding="utf-8")

    def record_step(self, row: Dict[str, Any], images: Optional[List[Image.Image]] = None) -> None:
        if not self.enable_step_debug or self._trace_handle is None or self.images_dir is None:
            return

        step_id = len(self.rows)
        image_paths: List[str] = []
        if images:
            for image_idx, image in enumerate(images):
                rel_path = os.path.join("images", f"step_{step_id:05d}_{image_idx:02d}.jpg")
                abs_path = os.path.join(self.debug_dir, rel_path)
                image.save(abs_path, format="JPEG", quality=90)
                image_paths.append(rel_path)

        payload = dict(row)
        payload["image_paths"] = image_paths
        self._trace_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._trace_handle.flush()
        self.rows.append(payload)

        checks = payload.get("checks", {})
        for key in self._checks_total:
            if checks.get(key) is True:
                self._checks_total[key] += 1

    def finalize(self) -> Dict[str, Any]:
        summary = {
            "episode_key": self.episode_key,
            "enabled": self.enable_step_debug,
            "steps_total": len(self.rows),
            "checks_pass_count": dict(self._checks_total),
        }

        if not self.enable_step_debug or self.debug_dir is None:
            return summary

        if self._trace_handle is not None:
            self._trace_handle.close()
            self._trace_handle = None

        summary_path = os.path.join(self.debug_dir, "check_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if self.step_debug_format == "html":
            self._write_html()
        return summary

    def _write_html(self) -> None:
        if self.debug_dir is None:
            return

        cards: List[str] = []
        for row in self.rows:
            prompt = html.escape(str(row.get("prompt", "")))
            query_tokens = html.escape(json.dumps(row.get("query_token_ids", [])))
            checks = row.get("checks", {})
            checks_html = "".join(
                [
                    f"<li>{html.escape(key)}: <b>{html.escape(str(value))}</b></li>"
                    for key, value in checks.items()
                ]
            )
            images_html = "".join(
                [
                    f'<img src="{html.escape(path)}" alt="{html.escape(path)}" loading="lazy"/>'
                    for path in row.get("image_paths", [])
                ]
            )
            header = (
                f"Episode={html.escape(str(row.get('episode_key', '')))} "
                f"| Subtask={html.escape(str(row.get('subtask_idx', '')))} "
                f"| Step={html.escape(str(row.get('rollout_step', '')))}"
            )
            cards.append(
                "<section class='card'>"
                f"<h2>{header}</h2>"
                "<h3>Prompt</h3>"
                f"<pre>{prompt}</pre>"
                "<h3>Query Tokens</h3>"
                f"<pre>{query_tokens}</pre>"
                "<h3>Checks</h3>"
                f"<ul>{checks_html}</ul>"
                "<h3>Model Input Images</h3>"
                f"<div class='images'>{images_html}</div>"
                "</section>"
            )

        html_str = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
            "<title>Step Debugger</title>"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:0;padding:16px;background:#f4f5f7;color:#111}"
            "h1{margin:0 0 12px 0} .meta{margin:0 0 16px 0;color:#444}"
            ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}"
            "pre{white-space:pre-wrap;word-break:break-word;background:#fafafa;border:1px solid #eee;padding:8px;border-radius:6px}"
            ".images{display:flex;flex-wrap:wrap;gap:8px} .images img{width:220px;height:auto;border:1px solid #ccc;border-radius:4px;background:#fff}"
            "</style></head><body>"
            "<h1>Step Debugger</h1>"
            f"<p class='meta'>Episode: {html.escape(self.episode_key)} | Steps: {len(self.rows)}</p>"
            + "".join(cards)
            + "</body></html>"
        )
        out_path = os.path.join(self.debug_dir, "index.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_str)


__all__ = ["EpisodeStepDebugger"]
