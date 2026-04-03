#!/usr/bin/env python3
"""Watcher SFT dataset / collator tests that will drive the new trainer work."""

import json
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Sequence
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from thinkvln.dataset.watcher_sft_dataset import (
    WatcherSFTCollator,
    WatcherSFTDataset,
    build_rollout_prompt,
    format_watcher_update_json,
    select_rollout_paths,
)
from thinkvln.engine.watcher_sft_trainer import build_watcher_trainer_components


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestWatcherDatasetJoin:
    def _create_rollout_files(self, bundle_root: Path, relpaths: Sequence[str]) -> None:
        image_dir = bundle_root / "images" / "rollout"
        image_dir.mkdir(parents=True, exist_ok=True)
        for rel in relpaths:
            (bundle_root / "images" / rel).parent.mkdir(parents=True, exist_ok=True)
            (bundle_root / "images" / rel).write_text("img", encoding="utf-8")

    def test_manifest_annotation_summary_merge(self, tmp_path: Path):
        manifest = tmp_path / "manifest.jsonl"
        annotation = tmp_path / "annotation.jsonl"
        summary = tmp_path / "summary.jsonl"
        bundle = tmp_path / "bundle"
        sample_id = "scene_00001_p000000_r01"
        plan = ["Navigate ahead", "Turn right", "Approach sink"]
        actions = ["forward", "turn_left", "forward"]
        relpaths = [
            "rollout/example/000000_rgb.jpg",
            "rollout/example/000001_rgb.jpg",
            "rollout/example/000002_rgb.jpg",
            "rollout/example/000003_rgb.jpg",
            "rollout/example/000004_rgb.jpg",
        ]
        manifest_row = {
            "sample_id": sample_id,
            "episode_key": "scene_00001",
            "instruction": "Go to the sink.",
            "plan": plan,
            "actions": actions,
            "subtask_id": 1,
            "subtask_text": plan[0],
            "base_image_path": "images",
            "rollout_image_relpaths": relpaths,
        }
        annotation_row = {
            "sample_id": sample_id,
            "memory_start": "Started at the hallway; at the doorway; active step in progress",
            "memory_end": "Entered the bathroom; facing the sink; ready for next step",
            "done": False,
            "next_subtask": "Align with the sink",
        }
        summary_row = {
            "episode_key": "scene_00001",
            "plan": plan,
            "instruction": "Go to the sink.",
        }
        _write_jsonl(manifest, [manifest_row])
        _write_jsonl(annotation, [annotation_row])
        _write_jsonl(summary, [summary_row])
        self._create_rollout_files(bundle, relpaths)

        dataset = WatcherSFTDataset(
            manifest_file=str(manifest),
            annotation_file=str(annotation),
            bundle_root=str(bundle),
            summary_full_path=str(summary),
            image_stride=2,
        )

        sample = dataset[0]
        assert sample["memory_start"].startswith("Started at the hallway")
        assert sample["memory_end"] == annotation_row["memory_end"]
        assert sample["next_subtask"] == annotation_row["next_subtask"]
        assert sample["plan_steps"] == plan
        assert sample["active_step"] == plan[0]
        assert sample["pending_steps"] == plan[1:]
        assert sample["rollout_actions"] == actions
        assert sample["rollout_image_paths"] == [
            str(bundle / "images" / relpaths[i]) for i in (0, 2, 4)
        ]


class TestRolloutPromptAndFormatting:
    def test_rollout_prompt_contains_plan_state(self):
        prompt = build_rollout_prompt(
            instruction="Move ahead.",
            plan_steps=["Step A", "Step B"],
            done_steps=["none yet"],
            active_step="Step A",
            pending_steps=["Step B"],
            memory_start="path summary; current; ongoing",
            rollout_actions=["forward", "turn_right"],
            rollout_images=[Image.new("RGB", (32, 32), color="blue")],
        )
        assert prompt.system_prompt.startswith("You are the watcher")
        assert "Plan state" in prompt.user_text
        assert "- Memory start" in prompt.user_text
        assert "- Rollout actions" in prompt.user_text
        assert "Step B" in prompt.user_text

    def test_watcher_json_format_order_and_bool(self):
        formatted_true = format_watcher_update_json(
            memory_end="done",
            done=True,
            next_subtask="stop",
        )
        assert formatted_true == '{"memory_end":"done","done":true,"next_subtask":"stop"}'
        formatted_false = format_watcher_update_json(
            memory_end="progress",
            done=False,
            next_subtask="keep going",
        )
        assert formatted_false == '{"memory_end":"progress","done":false,"next_subtask":"keep going"}'


class TestStrideSelection:
    def test_keep_last_frame(self):
        paths = [f"frame_{i}" for i in range(7)]
        selected = select_rollout_paths(paths, stride=3)
        assert selected[-1] == paths[-1]
        assert selected[0] == paths[0]
        assert len(selected) == 3


class _FakeWatcherProcessor:
    def __init__(self):
        self.call_count = 0
        self.prompt_len = 4

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prefix = "PROMPT" if add_generation_prompt else "PROMPT ASSISTANT"
        return prefix

    def __call__(self, text, images, return_tensors="pt", padding=False):
        self.call_count += 1
        seq_len = self.prompt_len if self.call_count == 1 else self.prompt_len + 4
        return {
            "input_ids": torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
            "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
            "pixel_values": torch.randn((len(images), 3, 16, 16), dtype=torch.float32),
            "image_grid_thw": torch.ones((len(images), 3), dtype=torch.long),
        }


class TestWatcherCollator:
    def test_prompt_masking_applies_minus_100(self):
        processor = _FakeWatcherProcessor()
        collator = WatcherSFTCollator(processor=processor)
        sample = {
            "instruction": "Go forward.",
            "plan_steps": ["Step A"],
            "memory_start": "start",
            "rollout_actions": ["forward"],
            "memory_end": "end",
            "next_subtask": "stop",
            "rollout_images": [Image.new("RGB", (32, 32), color="yellow")],
        }
        batch = collator([sample])
        labels = batch["labels"][0]
        assert torch.all(labels[: processor.prompt_len] == -100)
        assert torch.any(labels[processor.prompt_len :] >= 0)


class TestWatcherTrainerSmoke:
    @patch("thinkvln.engine.watcher_sft_trainer.Trainer", autospec=True)
    @patch("thinkvln.engine.watcher_sft_trainer.WatcherSFTCollator", autospec=True)
    @patch("thinkvln.engine.watcher_sft_trainer.WatcherSFTDataset", autospec=True)
    def test_build_trainer_components(self, mock_dataset, mock_collator, mock_trainer):
        config = {
            "model": {"model_name_or_path": "/tmp/model"},
            "data": {
                "manifest_file": "/tmp/manifest",
                "annotation_file": "/tmp/annotation",
                "bundle_root": "/tmp/bundle",
                "summary_full_path": "/tmp/summary",
            },
            "training": {"output_dir": "/tmp/output"},
        }
        build_watcher_trainer_components(config, processor=MagicMock())
        mock_dataset.assert_called_once()
        mock_collator.assert_called_once()
        mock_trainer.assert_called_once()

    @patch("peft.get_peft_model")
    @patch("peft.LoraConfig")
    @patch("transformers.Qwen3VLForConditionalGeneration")
    def test_load_model_enables_input_require_grads_for_lora_checkpointing(
        self,
        mock_qwen_cls,
        mock_lora_config,
        mock_get_peft_model,
    ):
        base_model = MagicMock()
        base_model.gradient_checkpointing_enable = MagicMock()
        base_model.enable_input_require_grads = MagicMock()
        base_model.base_model = MagicMock()
        base_model.base_model.model = MagicMock()
        base_model.base_model.model.visual = MagicMock()
        base_model.base_model.model.visual.parameters.return_value = []
        base_model.base_model.model.language_model = MagicMock()
        base_model.base_model.model.language_model.parameters.return_value = []
        mock_qwen_cls.from_pretrained.return_value = base_model
        mock_get_peft_model.return_value = base_model

        from thinkvln.engine.watcher_sft_trainer import WatcherSFTTrainingArguments, load_model

        args = WatcherSFTTrainingArguments(
            output_dir="/tmp/output",
            remove_unused_columns=False,
            model_name_or_path="/tmp/model",
            manifest_file="/tmp/manifest",
            annotation_file="/tmp/annotation",
            bundle_root="/tmp/bundle",
            summary_full_path="/tmp/summary",
            use_lora=True,
            gradient_checkpointing=True,
            bf16=False,
            fp16=False,
            report_to=[],
        )

        load_model(args)

        base_model.enable_input_require_grads.assert_called_once()
