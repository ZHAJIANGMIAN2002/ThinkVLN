from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import thinkvln.datagen.generation.watcher_openai_annotation as watcher_openai_annotation
import thinkvln.datagen.generation.watcher_openai_annotation_deploy as watcher_openai_annotation_deploy
from thinkvln.datagen.generation.watcher_openai_annotation_deploy import (
    MEMORY_START_SYSTEM_PROMPT,
    ROLLOUT_SYSTEM_PROMPT,
    annotate_sample,
    extract_gt_history_image_contents,
)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        response = self._responses.pop(0)
        return _FakeResponse(response)


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def _write_video(video_path: Path, frames) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        6,
        (width, height),
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_extract_gt_history_image_contents_extracts_needed_frames_to_tmp(tmp_path: Path):
    video_root = tmp_path / "images"
    episode_dir = video_root / "17DRP5sb8fy_r2r_000123"
    video_path = episode_dir / "trajectory.mp4"
    frames = []
    for idx in range(5):
        frame = np.zeros((8, 650, 3), dtype=np.uint8)
        frame[:, :640, :] = idx * 30
        frame[:, 640:, :] = 200
        frames.append(frame)
    _write_video(video_path, frames)

    record = {
        "scene_id": "17DRP5sb8fy",
        "episode_id": 123,
        "pivot_frame": 4,
    }

    contents = extract_gt_history_image_contents(video_root, record, stride=2)

    assert len(contents) == 3
    cache_dir = Path("/tmp/thinkvln_watcher_frames/17DRP5sb8fy_r2r_000123")
    assert (cache_dir / "000000_rgb.jpg").exists()
    assert (cache_dir / "000000_map.jpg").exists()
    assert (cache_dir / "000001_rgb.jpg").exists()
    assert (cache_dir / "000001_map.jpg").exists()
    assert (cache_dir / "000002_rgb.jpg").exists()
    assert (cache_dir / "000002_map.jpg").exists()
    assert (cache_dir / "000003_rgb.jpg").exists()
    assert (cache_dir / "000003_map.jpg").exists()
    assert (cache_dir / "000004_rgb.jpg").exists()
    assert (cache_dir / "000004_map.jpg").exists()

    rgb = cv2.imread(str(cache_dir / "000004_rgb.jpg"))
    map_img = cv2.imread(str(cache_dir / "000004_map.jpg"))
    assert rgb.shape[:2] == (8, 640)
    assert map_img.shape[:2] == (8, 10)


def test_deploy_prompts_match_non_deploy_prompts():
    assert MEMORY_START_SYSTEM_PROMPT == watcher_openai_annotation.MEMORY_START_SYSTEM_PROMPT
    assert ROLLOUT_SYSTEM_PROMPT == watcher_openai_annotation.ROLLOUT_SYSTEM_PROMPT
    source = Path(watcher_openai_annotation_deploy.__file__).read_text(encoding="utf-8")
    assert "watcher_openai_annotation as base_annotation" not in source


def test_annotate_sample_exports_prompt_debug_images(tmp_path: Path, monkeypatch):
    bundle_root = tmp_path / "bundle"
    rollout_path = bundle_root / "images" / "rollout" / "ep" / "pivot_000004" / "rollout_01" / "000000_rgb.jpg"
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(rollout_path), np.zeros((8, 8, 3), dtype=np.uint8))

    sample_id = "ep_p000004_r01"
    debug_dir = Path("/tmp/thinkvln_watcher_prompt_debug") / sample_id
    if debug_dir.exists():
        for path in debug_dir.iterdir():
            path.unlink()
        debug_dir.rmdir()

    monkeypatch.setattr(
        watcher_openai_annotation_deploy,
        "extract_gt_history_rgb_paths",
        lambda gt_video_root, record, stride: [rollout_path],
    )
    annotation = annotate_sample(
        client=_FakeClient(
            [
                '{"memory_start":"Reached the hallway entrance; mid-turn facing the wall; step ongoing"}',
                '{"memory_end":"Reached the hallway entrance; mid-turn facing the wall; step ongoing","done":false,"next_subtask":"finish turning right to face down the hallway"}',
            ]
        ),
        model="gpt-test",
        gt_video_root=tmp_path / "images",
        bundle_root=bundle_root,
        record={
            "sample_id": sample_id,
            "scene_id": "17DRP5sb8fy",
            "episode_id": 123,
            "pivot_frame": 4,
            "base_image_path": "images",
            "rollout_image_relpaths": ["rollout/ep/pivot_000004/rollout_01/000000_rgb.jpg"],
            "actions": ["turn_right"],
        },
        image_stride=2,
        request_timeout=30.0,
        max_retries=1,
        reasoning_effort="none",
    )

    assert annotation["sample_id"] == sample_id
    assert debug_dir.exists()
    assert [path.name for path in sorted(debug_dir.iterdir())] == ["000000_rgb.jpg"]
