import json
from pathlib import Path

from PIL import Image

from thinkvln.eval.close_eval_debugger import EpisodeStepDebugger


def test_episode_step_debugger_writes_trace_summary_and_html(tmp_path: Path):
    debugger = EpisodeStepDebugger(
        output_path=str(tmp_path),
        episode_key="17DRP5sb8fy_517",
        enable_step_debug=True,
        step_debug_format="html",
    )

    row = {
        "episode_key": "17DRP5sb8fy_517",
        "subtask_idx": 1,
        "rollout_step": 0,
        "prompt": "Instruction: x",
        "query_token_ids": [151700, 151701],
        "checks": {
            "memory_frame_count_ok": True,
            "memory_includes_past_subtask": True,
            "progress_pass_through_ok": True,
        },
    }
    images = [Image.new("RGB", (8, 8), color=(1, 2, 3))]
    debugger.record_step(row=row, images=images)
    summary = debugger.finalize()

    debug_dir = tmp_path / "debug_17DRP5sb8fy_517"
    assert (debug_dir / "step_trace.jsonl").exists()
    assert (debug_dir / "check_summary.json").exists()
    assert (debug_dir / "index.html").exists()
    assert (debug_dir / "images").exists()
    assert summary["steps_total"] == 1

    with (debug_dir / "step_trace.jsonl").open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["episode_key"] == "17DRP5sb8fy_517"
    assert "image_paths" in payload

    html = (debug_dir / "index.html").read_text(encoding="utf-8")
    assert "Step Debugger" in html
    assert "Instruction: x" in html
