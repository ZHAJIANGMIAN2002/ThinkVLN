import json
import sys
import types
from pathlib import Path

import torch
from PIL import Image

from streamvln.dataset.streamvln_actor_dataset import (
    StreamVLNActorDataset,
    build_materialized_streamvln_actor_records,
    build_streamvln_actor_prompt,
    load_streamvln_actor_samples,
    streamvln_actor_collate_fn,
)


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_streamvln_actor_samples_uses_hint_or_raw_per_frame(tmp_path: Path):
    summary_path = tmp_path / "summary_full.jsonl"
    watcher_path = tmp_path / "watcher_memory.jsonl"

    _write_jsonl(
        summary_path,
        [
            {
                "episode_key": "scene_00001",
                "num_frames": 5,
                "instruction": "Walk to the sink.",
                "plan": ["Walk down the hall.", "Stop at the sink."],
                "actions": [-1, 1, 1, 2, 0],
                "subtask_sequence": [1, 1, 1, 2, 2],
            }
        ],
    )
    _write_jsonl(
        watcher_path,
        [
            {
                "episode_key": "scene_00001",
                "pivot_frame": 1,
                "memory_start": "已经沿着走廊前进，并且朝向下一步入口。",
            }
        ],
    )

    samples = load_streamvln_actor_samples(
        summary_path=str(summary_path),
        watcher_memory_path=str(watcher_path),
        watcher_memory_ratio=1.0,
        done_threshold=0.85,
        seed=0,
    )

    assert len(samples) == 5

    frame_one = [sample for sample in samples if sample["episode_key"] == "scene_00001" and sample["frame_idx"] == 1]
    assert len(frame_one) == 1
    assert frame_one[0]["watcher_hint"] == "已经沿着走廊前进，并且朝向下一步入口。"
    assert frame_one[0]["subtask"] == "Walk down the hall."
    assert frame_one[0]["action_labels"] == [1, 0, 0, 0]

    frame_two = next(sample for sample in samples if sample["frame_idx"] == 2 and sample["watcher_hint"] is None)
    assert frame_two["progress_label"] == 1.0
    assert frame_two["done_label"] == 1.0


def test_build_materialized_streamvln_actor_records_combines_r2r_and_scalevln(tmp_path: Path):
    r2r_summary = tmp_path / "r2r_summary.jsonl"
    scale_summary = tmp_path / "scale_summary.jsonl"
    watcher_path = tmp_path / "watcher_memory.jsonl"

    _write_jsonl(
        r2r_summary,
        [
            {
                "episode_key": "scene_00001",
                "num_frames": 3,
                "instruction": "Walk to the sink.",
                "plan": ["Walk down the hall.", "Stop at the sink."],
                "actions": [-1, 1, 1],
                "subtask_sequence": [1, 1, 2],
                "video": "images/scene_r2r_000001",
            }
        ],
    )
    _write_jsonl(
        scale_summary,
        [
            {
                "episode_key": "scale_scene_00001",
                "num_frames": 2,
                "instruction": "Walk to the chair.",
                "plan": ["Walk forward.", "Stop by the chair."],
                "actions": [-1, 1],
                "subtask_sequence": [1, 2],
                "video": "images/scale_scalevln_000001",
            }
        ],
    )
    _write_jsonl(
        watcher_path,
        [
            {
                "sample_id": "scene_00001_p000001_r01",
                "memory_start": "已经朝向浴室入口。",
            }
        ],
    )

    records = build_materialized_streamvln_actor_records(
        summary_specs=[
            {"dataset_name": "r2r", "summary_path": str(r2r_summary), "image_root": "/tmp/r2r"},
            {"dataset_name": "scalevln", "summary_path": str(scale_summary), "image_root": "/tmp/scale"},
        ],
        watcher_memory_path=str(watcher_path),
        watcher_memory_ratio=1.0,
        done_threshold=0.85,
        memory_num_history_images=3,
        seed=0,
    )

    assert len(records) == 5
    r2r_hint = next(record for record in records if record["episode_key"] == "scene_00001" and record["frame_idx"] == 1)
    assert r2r_hint["dataset_name"] == "r2r"
    assert r2r_hint["watcher_hint"] == "已经朝向浴室入口。"
    assert r2r_hint["input_mode"] == "instruction_subtask_hint"
    assert r2r_hint["image_path"].endswith("/tmp/r2r/scene_r2r_000001/000001_rgb.jpg")
    assert r2r_hint["history_frame_indices"] == [0]
    assert r2r_hint["history_image_paths"] == ["/tmp/r2r/scene_r2r_000001/000000_rgb.jpg"]

    scale_raw = next(record for record in records if record["episode_key"] == "scale_scene_00001" and record["frame_idx"] == 0)
    assert scale_raw["dataset_name"] == "scalevln"
    assert scale_raw["watcher_hint"] is None
    assert scale_raw["input_mode"] == "instruction_subtask"
    assert scale_raw["history_frame_indices"] == []
    assert scale_raw["history_image_paths"] == []


def test_build_streamvln_actor_prompt_formats_optional_hint():
    prompt = build_streamvln_actor_prompt(
        instruction="Walk to the sink.",
        subtask="Turn right into the bathroom.",
        watcher_hint="You already cleared the dining area.",
        include_visual_memory=True,
        previous_progress=0.25,
    )
    assert "Instruction: Walk to the sink." in prompt
    assert "Current subtask: Turn right into the bathroom." in prompt
    assert "Watcher hint: You already cleared the dining area." in prompt
    assert "Previous progress: 0.2500" in prompt
    assert "<memory>" in prompt

    prompt_without_hint = build_streamvln_actor_prompt(
        instruction="Walk to the sink.",
        subtask="Turn right into the bathroom.",
        watcher_hint=None,
        previous_progress=0.0,
    )
    assert "Instruction: Walk to the sink." in prompt_without_hint
    assert "Current subtask: Turn right into the bathroom." in prompt_without_hint
    assert "Watcher hint:" not in prompt_without_hint
    assert "Previous progress: 0.0000" in prompt_without_hint
    assert "<memory>" not in prompt_without_hint


def test_streamvln_actor_collate_fn_batches_aux_labels():
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "labels": torch.tensor([-100, 2, 3], dtype=torch.long),
            "images": torch.ones((1, 3, 2, 2), dtype=torch.float32),
            "time_ids": torch.tensor([0], dtype=torch.long),
            "task_type": 0,
            "progress_labels": torch.tensor(0.25, dtype=torch.float32),
            "done_labels": torch.tensor(0.0, dtype=torch.float32),
        },
        {
            "input_ids": torch.tensor([4, 5], dtype=torch.long),
            "labels": torch.tensor([-100, 5], dtype=torch.long),
            "images": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
            "time_ids": torch.tensor([3], dtype=torch.long),
            "task_type": 0,
            "progress_labels": torch.tensor(1.0, dtype=torch.float32),
            "done_labels": torch.tensor(1.0, dtype=torch.float32),
        },
    ]

    class _Tokenizer:
        pad_token_id = 0
        model_max_length = 32

    collated = streamvln_actor_collate_fn(batch, tokenizer=_Tokenizer())
    assert tuple(collated["input_ids"].shape) == (2, 3)
    assert tuple(collated["labels"].shape) == (2, 3)
    assert tuple(collated["images"].shape) == (2, 1, 3, 2, 2)
    assert tuple(collated["time_ids"].shape) == (2, 1)
    assert tuple(collated["progress_labels"].shape) == (2,)
    assert tuple(collated["done_labels"].shape) == (2,)


def test_streamvln_actor_dataset_uses_materialized_history_image_paths(tmp_path: Path, monkeypatch):
    history0 = tmp_path / "history0.jpg"
    history1 = tmp_path / "history1.jpg"
    current = tmp_path / "current.jpg"
    for path, color in [(history0, (255, 0, 0)), (history1, (0, 255, 0)), (current, (0, 0, 255))]:
        Image.new("RGB", (4, 4), color=color).save(path)

    dataset_path = tmp_path / "actor_materialized.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "episode_key": "scene_00001",
                "frame_idx": 2,
                "instruction": "Walk to the sink.",
                "subtask": "Enter the bathroom.",
                "watcher_hint": None,
                "action_labels": [1, 0, 0, 0],
                "progress_label": 0.5,
                "done_label": 0.0,
                "history_frame_indices": [0, 1],
                "history_image_paths": [str(history0), str(history1)],
                "image_path": str(current),
                "input_mode": "instruction_subtask",
            }
        ],
    )

    class _Tokenizer:
        pad_token_id = 0
        model_max_length = 128
        _vocab = {"<image>": 101, "<memory>": 102}

        def add_tokens(self, tokens, special_tokens=False):
            for token in tokens:
                self._vocab.setdefault(token, len(self._vocab) + 200)

        def convert_tokens_to_ids(self, token):
            return self._vocab.get(token)

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            rendered = "\n".join(message["content"] for message in messages)
            if add_generation_prompt:
                rendered += "\nassistant:"
            return rendered

        def __call__(self, text, return_tensors="pt"):
            tokens = text.replace("\n", " ").split()
            ids = [self._vocab.setdefault(tok, len(self._vocab) + 200) for tok in tokens] or [1]
            return types.SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    class _Processor:
        def preprocess(self, images, return_tensors="pt"):
            pixel = images.getpixel((0, 0))
            value = float(sum(pixel)) / 255.0
            return {"pixel_values": torch.full((1, 3, 2, 2), value, dtype=torch.float32)}

    fake_llava = types.ModuleType("llava")
    fake_llava_model = types.ModuleType("llava.model")
    fake_llava_mm = types.ModuleType("llava.model.multimodal_encoder")
    fake_siglip = types.ModuleType("llava.model.multimodal_encoder.siglip_encoder")
    fake_siglip.SigLipImageProcessor = _Processor
    monkeypatch.setitem(sys.modules, "llava", fake_llava)
    monkeypatch.setitem(sys.modules, "llava.model", fake_llava_model)
    monkeypatch.setitem(sys.modules, "llava.model.multimodal_encoder", fake_llava_mm)
    monkeypatch.setitem(sys.modules, "llava.model.multimodal_encoder.siglip_encoder", fake_siglip)

    data_args = types.SimpleNamespace(
        image_root="",
        image_folder="",
        summary_data_path=str(dataset_path),
        data_path=str(dataset_path),
        watcher_memory_path=None,
        watcher_memory_ratio=1.0,
        watcher_memory_seed=42,
        done_threshold=0.85,
        memory_num_history_images=2,
        num_history=2,
    )

    dataset = StreamVLNActorDataset(tokenizer=_Tokenizer(), data_args=data_args, task_id=0)
    item = dataset[0]

    assert tuple(item["images"].shape) == (3, 3, 2, 2)


def test_streamvln_actor_dataset_falls_back_from_r2r_images_dir_to_frame_dir(tmp_path: Path, monkeypatch):
    base = tmp_path / "data" / "trajectory_data" / "R2R_back"
    wrong_dir = base / "images" / "scene_r2r_000001"
    right_dir = base / "r2r" / "scene_r2r_000001"
    wrong_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    (wrong_dir / "trajectory.mp4").write_bytes(b"")
    for idx, color in [(0, (255, 0, 0)), (1, (0, 255, 0))]:
        Image.new("RGB", (4, 4), color=color).save(right_dir / f"{idx:06d}_rgb.jpg")

    dataset_path = tmp_path / "actor_r2r_materialized.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "episode_key": "scene_00001",
                "frame_idx": 1,
                "instruction": "Walk to the sink.",
                "subtask": "Enter the bathroom.",
                "watcher_hint": None,
                "action_labels": [1, 0, 0, 0],
                "progress_label": 0.5,
                "done_label": 0.0,
                "history_frame_indices": [0],
                "history_image_paths": [str(wrong_dir / "000000_rgb.jpg")],
                "image_path": str(wrong_dir / "000001_rgb.jpg"),
                "input_mode": "instruction_subtask",
            }
        ],
    )

    class _Tokenizer:
        pad_token_id = 0
        model_max_length = 128
        _vocab = {"<image>": 101, "<memory>": 102}

        def add_tokens(self, tokens, special_tokens=False):
            for token in tokens:
                self._vocab.setdefault(token, len(self._vocab) + 200)

        def convert_tokens_to_ids(self, token):
            return self._vocab.get(token)

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            rendered = "\n".join(message["content"] for message in messages)
            if add_generation_prompt:
                rendered += "\nassistant:"
            return rendered

        def __call__(self, text, return_tensors="pt"):
            tokens = text.replace("\n", " ").split()
            ids = [self._vocab.setdefault(tok, len(self._vocab) + 200) for tok in tokens] or [1]
            return types.SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    class _Processor:
        def preprocess(self, images, return_tensors="pt"):
            pixel = images.getpixel((0, 0))
            value = float(sum(pixel)) / 255.0
            return {"pixel_values": torch.full((1, 3, 2, 2), value, dtype=torch.float32)}

    fake_llava = types.ModuleType("llava")
    fake_llava_model = types.ModuleType("llava.model")
    fake_llava_mm = types.ModuleType("llava.model.multimodal_encoder")
    fake_siglip = types.ModuleType("llava.model.multimodal_encoder.siglip_encoder")
    fake_siglip.SigLipImageProcessor = _Processor
    monkeypatch.setitem(sys.modules, "llava", fake_llava)
    monkeypatch.setitem(sys.modules, "llava.model", fake_llava_model)
    monkeypatch.setitem(sys.modules, "llava.model.multimodal_encoder", fake_llava_mm)
    monkeypatch.setitem(sys.modules, "llava.model.multimodal_encoder.siglip_encoder", fake_siglip)

    data_args = types.SimpleNamespace(
        image_root="",
        image_folder="",
        summary_data_path=str(dataset_path),
        data_path=str(dataset_path),
        watcher_memory_path=None,
        watcher_memory_ratio=1.0,
        watcher_memory_seed=42,
        done_threshold=0.85,
        memory_num_history_images=1,
        num_history=1,
    )

    dataset = StreamVLNActorDataset(tokenizer=_Tokenizer(), data_args=data_args, task_id=0)
    item = dataset[0]

    assert tuple(item["images"].shape) == (2, 3, 2, 2)
