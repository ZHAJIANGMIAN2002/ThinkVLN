#!/usr/bin/env python3
"""Unit tests for ThinkVLN dataset utilities and collator."""

import json
import os
import tempfile
from unittest.mock import patch

import torch
from PIL import Image

from thinkvln.dataset.dataset import (
    ACTION_MAPPING,
    ThinkVLNDataCollator,
    ThinkVLNDataset,
)
from thinkvln.tools.dataset_utils import (
    compute_current_step_progress,
    compute_done_label,
    compute_previous_step_progress,
    crop_cot_answer,
    extract_action_chunk,
    parse_frame_key,
    select_memory_frame_indices,
)


class _DummyProcessor:
    def __init__(self):
        self.tokenizer = type("Tok", (), {"pad_token_id": 0})()
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        return "mock_text"

    def __call__(self, text, images, return_tensors="pt", padding=False):
        seq_len = 16
        num_images = len(images)
        return {
            "input_ids": torch.randint(0, 1000, (1, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
            "pixel_values": torch.randn((num_images, 8), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 1, 1]] * num_images, dtype=torch.long),
        }


class TestThinkVLNDataset:
    @staticmethod
    def create_mock_action_data(num_episodes: int = 2) -> str:
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl")
        for i in range(num_episodes):
            sample = {
                "episode_key": f"scene001_{i:05d}",
                "num_frames": 6,
                "instruction": f"Navigate to room {i}",
                "plan": ["Step 1", "Step 2"],
                "actions": [1, 1, 2, 1, 3, 0],
                "subtask_sequence": [1, 1, 1, 2, 2, 2],
            }
            temp_file.write(json.dumps(sample) + "\n")
        temp_file.close()
        return temp_file.name

    def test_dataset_initialization_action_only(self):
        action_data_path = self.create_mock_action_data(2)
        try:
            dataset = ThinkVLNDataset(action_data_path=action_data_path, cot_data_path=None, image_root="/tmp")
            assert len(dataset.action_samples) == 12
            assert len(dataset.cot_samples) == 0
            assert len(dataset.samples) == 12
            assert dataset.samples[0]["data_type"] == "action"
        finally:
            os.unlink(action_data_path)

    def test_empty_dataset(self):
        dataset = ThinkVLNDataset(action_data_path=None, cot_data_path=None, image_root="/tmp")
        assert len(dataset) == 0
        assert len(dataset.samples) == 0


class TestThinkVLNDataCollator:
    @patch("thinkvln.dataset.dataset.load_image")
    def test_collator_process_action_sample_has_scalar_progress_and_done(self, mock_load_image):
        mock_load_image.return_value = Image.new("RGB", (32, 32), color="red")
        processor = _DummyProcessor()
        collator = ThinkVLNDataCollator(processor=processor, num_query_tokens=4, image_root="/tmp")

        sample = {
            "data_type": "action",
            "episode_key": "scene001_00001",
            "frame_idx": 3,
            "instruction": "Navigate forward",
            "current_plan_step": "Move ahead",
            "current_subtask_idx": 2,
            "actions": [1, 1, 2, 1, 3, 0],
            "subtask_sequence": [1, 1, 1, 2, 2, 2],
        }
        processed = collator._process_action(sample)

        assert processed["action_labels"].shape == (4,)
        assert processed["progress_labels"].ndim == 0
        assert processed["done_labels"].ndim == 0
        assert processed["data_type"] == "action"

        prompt = processor.last_messages[0]["content"][-1]["text"]
        assert "Previous progress:" in prompt
        assert "<memory>" in prompt

    @patch("thinkvln.dataset.dataset.load_image")
    def test_collator_collate_action_batch_shapes(self, mock_load_image):
        mock_load_image.return_value = Image.new("RGB", (32, 32), color="green")
        processor = _DummyProcessor()
        collator = ThinkVLNDataCollator(processor=processor, num_query_tokens=4, image_root="/tmp")

        batch = [
            {
                "data_type": "action",
                "episode_key": "scene001_00001",
                "frame_idx": i,
                "instruction": "Navigate",
                "current_plan_step": "Move ahead",
                "current_subtask_idx": 1,
                "actions": [1, 1, 2, 1, 3, 0],
                "subtask_sequence": [1, 1, 1, 2, 2, 2],
            }
            for i in (0, 1)
        ]

        out = collator(batch)
        assert out["action_labels"].shape == (2, 4)
        assert out["progress_labels"].shape == (2,)
        assert out["done_labels"].shape == (2,)
        assert out["labels"] is None


class TestDatasetUtils:
    def test_extract_action_chunk(self):
        action_chunk, progress_chunk = extract_action_chunk(
            frame_idx=0,
            actions=[1, 2, 3, 1, 0, 0],
            subtask_sequence=[1, 1, 1, 2, 2, 2],
            num_steps=4,
        )
        assert len(action_chunk) == 4
        assert len(progress_chunk) == 4
        assert action_chunk[0] == 2

    def test_scalar_progress_previous_progress_and_done_rules(self):
        subtask_sequence = [1, 1, 1, 2, 2]
        assert compute_current_step_progress(0, subtask_sequence) == 0.0
        assert compute_current_step_progress(1, subtask_sequence) > 0.0
        assert compute_current_step_progress(2, subtask_sequence) >= compute_current_step_progress(1, subtask_sequence)
        assert compute_previous_step_progress(3, subtask_sequence) == 0.0

        assert compute_done_label(0.85, threshold=0.85) == 0.0
        assert compute_done_label(0.850001, threshold=0.85) == 1.0

    def test_memory_selection_anchors_sparse_and_cap(self):
        subtask_sequence = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
        selected = select_memory_frame_indices(
            frame_idx=10,
            subtask_sequence=subtask_sequence,
            memory_num_history_images=4,
        )
        assert 10 not in selected
        assert 0 in selected
        assert 4 in selected
        assert len(selected) <= 4

        trimmed = select_memory_frame_indices(
            frame_idx=10,
            subtask_sequence=subtask_sequence,
            memory_num_history_images=1,
        )
        assert trimmed == [0]

    def test_misc_utils(self):
        episode_key, step_id = parse_frame_key("17DRP5sb8fy_10154_000035")
        assert episode_key == "17DRP5sb8fy_10154"
        assert step_id == 35

        text = "abc [causal observation] keep this"
        assert crop_cot_answer(text).startswith("[causal observation]")

        assert ACTION_MAPPING[0] == "stop"
        assert ACTION_MAPPING[1] == "forward"
        assert ACTION_MAPPING[2] == "turn_left"
        assert ACTION_MAPPING[3] == "turn_right"
