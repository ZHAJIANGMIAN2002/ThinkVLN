import json
import sys
import types
from pathlib import Path

import torch
from PIL import Image

from streamvln.dataset.streamvln_actor_dataset import (
    SequentialSubtaskSampler,
    StreamVLNActorDataset,
    _tokenize_actor_sample,
    build_materialized_streamvln_actor_records,
    build_streamvln_actor_prompt,
    load_streamvln_actor_samples,
    streamvln_actor_collate_fn,
)
from streamvln.utils.utils import ANCHOR_TOKEN_INDEX
from thinkvln.tools.dataset_utils import NEXT_SUBTASK_SENTINEL


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
    assert r2r_hint["anchor_frame_idx"] == 0
    assert r2r_hint["anchor_image_path"] == "/tmp/r2r/scene_r2r_000001/000000_rgb.jpg"
    assert r2r_hint["history_frame_indices"] == []
    assert r2r_hint["history_image_paths"] == []

    scale_raw = next(record for record in records if record["episode_key"] == "scale_scene_00001" and record["frame_idx"] == 0)
    assert scale_raw["dataset_name"] == "scalevln"
    assert scale_raw["watcher_hint"] is None
    assert scale_raw["input_mode"] == "instruction_subtask"
    assert scale_raw["image_path"].endswith("/tmp/scale/images/scale_scalevln_000001/000001_rgb.jpg")
    assert scale_raw["history_frame_indices"] == []
    assert scale_raw["history_image_paths"] == []

    scale_next = next(record for record in records if record["episode_key"] == "scale_scene_00001" and record["frame_idx"] == 1)
    assert scale_next["image_path"].endswith("/tmp/scale/images/scale_scalevln_000001/000002_rgb.jpg")
    assert scale_next["history_frame_indices"] == [0]
    assert scale_next["history_image_paths"] == ["/tmp/scale/images/scale_scalevln_000001/000001_rgb.jpg"]


def test_build_materialized_streamvln_actor_records_prefers_r2r_frame_dir_when_present(tmp_path: Path):
    summary_path = tmp_path / "r2r_summary.jsonl"
    image_root = tmp_path / "R2R_back"
    (image_root / "r2r" / "scene_r2r_000001").mkdir(parents=True)
    _write_jsonl(
        summary_path,
        [
            {
                "episode_key": "scene_00001",
                "num_frames": 2,
                "instruction": "Walk to the sink.",
                "plan": ["Walk down the hall."],
                "actions": [-1, 1],
                "subtask_sequence": [1, 1],
                "video": "images/scene_r2r_000001",
            }
        ],
    )

    records = build_materialized_streamvln_actor_records(
        summary_specs=[
            {"dataset_name": "r2r", "summary_path": str(summary_path), "image_root": str(image_root)},
        ],
        memory_num_history_images=2,
        seed=0,
    )

    frame1 = next(record for record in records if record["frame_idx"] == 1)
    assert frame1["anchor_image_path"].endswith("/R2R_back/r2r/scene_r2r_000001/000000_rgb.jpg")
    assert frame1["image_path"].endswith("/R2R_back/r2r/scene_r2r_000001/000001_rgb.jpg")


def test_build_materialized_streamvln_actor_records_supports_v2_fields(tmp_path: Path):
    summary_path = tmp_path / "summary.jsonl"
    _write_jsonl(
        summary_path,
        [
            {
                "episode_key": "scene_00001",
                "num_frames": 5,
                "instruction": "Walk to the sink.",
                "plan": ["Go down the hall.", "Stop at the sink."],
                "actions": [1, 1, 2, 1, 0],
                "subtask_sequence": [1, 1, 1, 1, 2],
                "video": "images/scene_r2r_000001",
            }
        ],
    )

    records = build_materialized_streamvln_actor_records(
        summary_specs=[
            {"dataset_name": "r2r", "summary_path": str(summary_path), "image_root": "/tmp/r2r"},
        ],
        memory_num_history_images=3,
        memory_pre_anchor_count=1,
        memory_post_anchor_count=2,
        use_next_token=True,
        use_sliding_window=True,
        action_history_len=2,
        seed=0,
    )

    row = next(record for record in records if record["frame_idx"] == 3)

    assert row["action_labels"] == [NEXT_SUBTASK_SENTINEL] * 4
    assert row["next_subtask"] == "Stop at the sink."
    assert row["subtask_position"] == "1/2"
    assert row["steps_in_subtask"] == 3
    assert row["action_history_str"] == "↑ ←"
    assert row["subtask_idx"] == 0
    assert row["anchor_frame_idx"] == 0
    assert row["history_frame_indices"] == [1, 2]


def test_sequential_subtask_sampler_orders_subset_by_subtask_and_frame():
    class _Dataset:
        samples = [
            {"episode_key": "ep1", "subtask_position": "1/2", "frame_idx": 1},
            {"episode_key": "ep1", "subtask_position": "1/2", "frame_idx": 0},
            {"episode_key": "ep1", "subtask_position": "2/2", "frame_idx": 3},
            {"episode_key": "ep1", "subtask_position": "2/2", "frame_idx": 2},
        ]

        def __len__(self):
            return len(self.samples)

    subset = torch.utils.data.Subset(_Dataset(), [0, 1, 2, 3])

    sampler = SequentialSubtaskSampler(subset, shuffle=False, seed=0)

    assert list(iter(sampler)) == [1, 0, 3, 2]


def test_sequential_subtask_sampler_returns_subset_relative_indices():
    class _Dataset:
        samples = [
            {"episode_key": "ep0", "subtask_position": "1/1", "frame_idx": 0},
            {"episode_key": "ep0", "subtask_position": "1/1", "frame_idx": 1},
            {"episode_key": "ep1", "subtask_position": "1/2", "frame_idx": 1},
            {"episode_key": "ep1", "subtask_position": "1/2", "frame_idx": 0},
            {"episode_key": "ep1", "subtask_position": "2/2", "frame_idx": 3},
            {"episode_key": "ep1", "subtask_position": "2/2", "frame_idx": 2},
        ]

        def __len__(self):
            return len(self.samples)

    subset = torch.utils.data.Subset(_Dataset(), [2, 3, 4, 5])

    sampler = SequentialSubtaskSampler(subset, shuffle=False, seed=0)

    assert list(iter(sampler)) == [1, 0, 3, 2]


def test_build_streamvln_actor_prompt_formats_optional_hint():
    prompt = build_streamvln_actor_prompt(
        instruction="Walk to the sink.",
        subtask="Turn right into the bathroom.",
        watcher_hint="You already cleared the dining area.",
        include_visual_memory=True,
        include_anchor_frame=True,
    )
    assert "Instruction: Walk to the sink." in prompt
    assert "Current subtask: Turn right into the bathroom." in prompt
    assert "Watcher hint: You already cleared the dining area." in prompt
    assert "Previous progress:" not in prompt
    assert "Subtask start observation: <anchor>" in prompt
    assert "<memory>" in prompt

    prompt_without_hint = build_streamvln_actor_prompt(
        instruction="Walk to the sink.",
        subtask="Turn right into the bathroom.",
        watcher_hint=None,
        include_anchor_frame=True,
    )
    assert "Instruction: Walk to the sink." in prompt_without_hint
    assert "Current subtask: Turn right into the bathroom." in prompt_without_hint
    assert "Watcher hint:" not in prompt_without_hint
    assert "Previous progress:" not in prompt_without_hint
    assert "<anchor>" in prompt_without_hint
    assert "<memory>" not in prompt_without_hint


def test_tokenize_actor_sample_replaces_anchor_token():
    class _Tokenizer:
        pad_token_id = 0
        model_max_length = 128
        _vocab = {"<image>": 101, "<memory>": 102, "<anchor>": 103}

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

    prompt = build_streamvln_actor_prompt(
        instruction="Walk to the sink.",
        subtask="Turn right into the bathroom.",
        include_visual_memory=True,
        include_anchor_frame=True,
    )

    input_ids, _ = _tokenize_actor_sample(_Tokenizer(), prompt, "↑ STOP STOP STOP")

    assert int(ANCHOR_TOKEN_INDEX) in input_ids.tolist()


def test_load_materialized_actor_samples_backfills_progress_and_done(tmp_path: Path):
    materialized_path = tmp_path / "materialized_actor.jsonl"
    _write_jsonl(
        materialized_path,
        [
            {
                "episode_key": "scene_00001",
                "frame_idx": 0,
                "instruction": "Walk to the sink.",
                "subtask": "Start moving.",
                "action_labels": [1, 1, 1, 1],
                "progress_label": 0.0,
            },
            {
                "episode_key": "scene_00001",
                "frame_idx": 1,
                "instruction": "Walk to the sink.",
                "subtask": "Keep moving.",
                "action_labels": [1, 1, 0, 0],
                "progress_label": 0.5,
            },
            {
                "episode_key": "scene_00001",
                "frame_idx": 2,
                "instruction": "Walk to the sink.",
                "subtask": "Stop at sink.",
                "action_labels": [0, 0, 0, 0],
                "progress_label": 1.0,
            },
        ],
    )

    rows = load_streamvln_actor_samples(
        summary_path=str(materialized_path),
        watcher_memory_path=None,
        watcher_memory_ratio=1.0,
        done_threshold=0.85,
        seed=0,
    )

    assert [round(float(row["previous_progress"]), 4) for row in rows] == [0.0, 0.0, 0.5]
    assert [float(row["done_label"]) for row in rows] == [0.0, 0.0, 1.0]
    assert all("history_frame_indices" in row for row in rows)
    assert all("history_image_paths" in row for row in rows)


def test_load_materialized_actor_samples_remaps_hints_from_watcher_file(tmp_path: Path):
    materialized_path = tmp_path / "materialized_actor.jsonl"
    watcher_path = tmp_path / "watcher_memory.jsonl"
    _write_jsonl(
        materialized_path,
        [
            {
                "episode_key": "scene_00001",
                "frame_idx": 0,
                "instruction": "Walk to the sink.",
                "subtask": "Start moving.",
                "action_labels": [1, 1, 1, 1],
                "progress_label": 0.0,
                "previous_progress": 0.0,
                "done_label": 0.0,
                "watcher_hint": "old hint should be replaced",
            },
            {
                "episode_key": "scene_00001",
                "frame_idx": 1,
                "instruction": "Walk to the sink.",
                "subtask": "Keep moving.",
                "action_labels": [1, 1, 0, 0],
                "progress_label": 0.5,
                "previous_progress": 0.0,
                "done_label": 0.0,
                "watcher_hint": "old hint should be replaced",
            },
        ],
    )
    _write_jsonl(
        watcher_path,
        [
            {
                "episode_key": "scene_00001",
                "pivot_frame": 1,
                "memory_start": "new watcher hint",
            }
        ],
    )

    rows = load_streamvln_actor_samples(
        summary_path=str(materialized_path),
        watcher_memory_path=str(watcher_path),
        watcher_memory_ratio=1.0,
        done_threshold=0.85,
        seed=0,
    )

    frame0 = next(row for row in rows if int(row["frame_idx"]) == 0)
    frame1 = next(row for row in rows if int(row["frame_idx"]) == 1)
    assert frame0["watcher_hint"] is None
    assert frame0["input_mode"] == "instruction_subtask"
    assert frame1["watcher_hint"] == "new watcher hint"
    assert frame1["input_mode"] == "instruction_subtask_hint"


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
                "anchor_frame_idx": 1,
                "anchor_image_path": str(history1),
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
        _vocab = {"<image>": 101, "<memory>": 102, "<anchor>": 103}

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

    assert tuple(item["images"].shape) == (4, 3, 2, 2)


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
                "anchor_frame_idx": 0,
                "anchor_image_path": str(wrong_dir / "000000_rgb.jpg"),
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
        _vocab = {"<image>": 101, "<memory>": 102, "<anchor>": 103}

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

    assert tuple(item["images"].shape) == (3, 3, 2, 2)


def test_resolve_existing_image_path_clamps_to_latest_available_frame(tmp_path: Path):
    from streamvln.dataset.streamvln_actor_dataset import _resolve_existing_image_path

    base = tmp_path / "data" / "trajectory_data" / "R2R_back"
    image_dir = base / "images" / "scene_r2r_000001"
    frame_dir = base / "r2r" / "scene_r2r_000001"
    image_dir.mkdir(parents=True)
    frame_dir.mkdir(parents=True)
    for idx in range(24):
        Image.new("RGB", (4, 4), color=(idx, idx, idx)).save(frame_dir / f"{idx:06d}_rgb.jpg")

    resolved = _resolve_existing_image_path(str(image_dir / "000026_rgb.jpg"))

    assert resolved == str(frame_dir / "000023_rgb.jpg")
