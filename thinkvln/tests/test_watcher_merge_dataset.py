import json
from pathlib import Path

from thinkvln.datagen.generation.watcher_merge_dataset import merge_dataset


class TestWatcherMergeDataset:
    def test_merge_dataset_formats_traj_and_preserves_relative_paths(self, tmp_path: Path):
        manifest_file = tmp_path / "manifest.jsonl"
        annotation_file = tmp_path / "annotations.jsonl"
        output_file = tmp_path / "dataset.jsonl"

        manifest_rows = [
            {
                "sample_id": "sample_1",
                "episode_key": "scene_1",
                "episode_id": 1,
                "scene_id": "scene",
                "pivot_frame": 10,
                "rollout_id": 1,
                "subtask_id": 2,
                "instruction": "Go forward.",
                "subtask_text": "Enter the room.",
                "base_image_path": "images",
                "pivot_image_relpath": "pivot/scene_1/pivot_000010_rgb.jpg",
                "rollout_image_relpaths": [
                    "rollout/scene_1/pivot_000010/rollout_01/000000_rgb.jpg",
                    "rollout/scene_1/pivot_000010/rollout_01/000001_rgb.jpg",
                ],
                "actions": ["forward", "turn_right"],
            },
            {
                "sample_id": "sample_missing",
                "episode_key": "scene_2",
                "episode_id": 2,
                "scene_id": "scene",
                "pivot_frame": 11,
                "rollout_id": 1,
                "subtask_id": 1,
                "instruction": "Go left.",
                "subtask_text": "Move left.",
                "base_image_path": "images",
                "pivot_image_relpath": "pivot/scene_2/pivot_000011_rgb.jpg",
                "rollout_image_relpaths": [],
                "actions": [],
            },
        ]
        annotation_rows = [
            {
                "sample_id": "sample_1",
                "memory_start": "The agent is near the doorway.",
                "label": "PROCEED",
                "memory_end": "The agent entered the room.",
            }
        ]

        with open(manifest_file, "w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row) + "\n")
        with open(annotation_file, "w", encoding="utf-8") as handle:
            for row in annotation_rows:
                handle.write(json.dumps(row) + "\n")

        written, missing = merge_dataset(manifest_file, annotation_file, output_file)

        assert written == 1
        assert missing == 1

        with open(output_file, "r", encoding="utf-8") as handle:
            merged = json.loads(handle.readline())

        assert merged["traj"] == "a=[forward,turn_right]"
        assert merged["base_image_path"] == "images"
        assert merged["pivot_image_relpath"] == "pivot/scene_1/pivot_000010_rgb.jpg"
        assert merged["rollout_image_relpaths"] == manifest_rows[0]["rollout_image_relpaths"]
