import json
from pathlib import Path

from thinkvln.datagen.generation.watcher_actor_memory_dataset import (
    parse_sample_id_to_episode_frame,
    convert_watcher_dataset_to_actor_memory_records,
)


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_convert_watcher_dataset_to_actor_memory_records_uses_memory_start(tmp_path: Path):
    watcher_dataset = tmp_path / "watcher_dataset.jsonl"
    _write_jsonl(
        watcher_dataset,
        [
            {
                "episode_key": "scene_00001",
                "pivot_frame": 12,
                "memory_start": "已经走到门口附近，当前对准入口。",
                "next_subtask": "enter the room",
            },
            {
                "episode_key": "scene_00002",
                "pivot_frame": 3,
                "memory_start": "",
            },
        ],
    )

    rows = convert_watcher_dataset_to_actor_memory_records(str(watcher_dataset))
    assert rows == [
        {
            "episode_key": "scene_00001",
            "frame_idx": 12,
            "watcher_hint": "已经走到门口附近，当前对准入口。",
        }
    ]


def test_convert_watcher_dataset_to_actor_memory_records_parses_sample_id():
    rows = convert_watcher_dataset_to_actor_memory_records.__globals__["_load_jsonl"]  # keep import path untouched
    del rows
    episode_key, frame_idx = parse_sample_id_to_episode_frame("17DRP5sb8fy_2889_p000028_r01")
    assert episode_key == "17DRP5sb8fy_2889"
    assert frame_idx == 28
