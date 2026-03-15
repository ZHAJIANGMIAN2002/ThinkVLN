import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from thinkvln.eval.close_eval_runner import VLNEvaluator
from thinkvln.models.navigation_model import ThinkVLNActorNavigationModel

_CLOSE_EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "close_eval.py"
_SPEC = importlib.util.spec_from_file_location("close_eval_for_test", _CLOSE_EVAL_PATH)
_CLOSE_EVAL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_CLOSE_EVAL)

build_subtask_spans = _CLOSE_EVAL.build_subtask_spans
compute_step_budget = _CLOSE_EVAL.compute_step_budget
summarize_subtask_aggregation = _CLOSE_EVAL.summarize_subtask_aggregation
timeline_progress = _CLOSE_EVAL.timeline_progress


class _DummyProcessor:
    def __init__(self):
        self.prompts = []
        self.last_image_count = 0
        self.received_images = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prompt = messages[0]["content"][-1]["text"]
        self.prompts.append(prompt)
        self.last_image_count = sum(1 for item in messages[0]["content"] if item["type"] == "image")
        return "prompt"

    def __call__(self, text, images, return_tensors, padding):
        assert len(text) == 1
        assert len(images) >= 1
        self.received_images = list(images)
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "pixel_values": torch.ones((len(images), 3, 8, 8), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 8, 8]] * len(images), dtype=torch.long),
        }


class _DummyActor(torch.nn.Module):
    def __init__(
        self,
        progress_value: float,
        done_prob: float,
        query_token_layout: str = "interleaved",
    ):
        super().__init__()
        self.config = SimpleNamespace(
            num_query_tokens=4,
            action_query_token_id=151700,
            progress_query_token_id=151701,
            query_token_layout=query_token_layout,
            num_action_query_tokens=4,
            num_progress_query_tokens=4,
        )
        self.progress_value = progress_value
        self.done_prob = done_prob
        self.last_kwargs = None

    def eval(self):
        return self

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        action_logits = torch.tensor(
            [[[0.1, 0.2, 0.8, 0.0], [0.3, 0.4, 0.2, 0.1], [0.4, 0.2, 0.1, 0.3], [0.9, 0.0, 0.0, 0.1]]],
            dtype=torch.float32,
        )
        return {
            "action_logits": action_logits,
            "progress_preds": torch.tensor([self.progress_value], dtype=torch.float32),
            "done_preds": torch.tensor([self.done_prob], dtype=torch.float32),
        }


class CloseEvalLadderUtilsTest:
    def test_build_subtask_spans_extracts_contiguous_ranges(self):
        spans = build_subtask_spans([1, 1, 2, 2, 2, 3, 1, 1])
        assert spans == [(1, 0, 1), (2, 2, 4), (3, 5, 5), (1, 6, 7)]

    def test_timeline_progress_handles_zero_and_normal_cases(self):
        assert timeline_progress(0, 0) == 1.0
        assert timeline_progress(0, 4) == 0.25
        assert timeline_progress(3, 4) == 1.0
        assert timeline_progress(10, 4) == 1.0

    def test_compute_step_budget_uses_ceiling_and_min_one(self):
        assert compute_step_budget(0, 2.0) == 1
        assert compute_step_budget(3, 2.0) == 6
        assert compute_step_budget(3, 1.2) == 4

    def test_subtask_aggregation_math(self):
        stats = {
            "subtasks_total": 4.0,
            "subtasks_success": 3.0,
            "steps_success_sum": 12.0,
            "steps_success_count": 3.0,
            "progress_abs_error_sum": 2.5,
            "progress_count": 10.0,
        }
        summary = summarize_subtask_aggregation(stats)
        assert summary["subtask_success_rate"] == 0.75
        assert summary["steps_to_subgoal"] == 4.0
        assert summary["progress_mae"] == 0.25

class TestActorWrapper:
    def test_build_inputs_includes_image_tensors(self):
        model = _DummyActor(progress_value=0.1, done_prob=0.0)
        processor = _DummyProcessor()
        wrapper = ThinkVLNActorNavigationModel(model=model, processor=processor, device="cpu")

        images = [
            Image.new("RGB", (8, 8), color=(10, 10, 10)),
            Image.new("RGB", (8, 8), color=(20, 20, 20)),
        ]

        model_inputs = wrapper._build_inputs(images, "prompt text")

        assert processor.last_image_count == 2
        assert processor.received_images is not None
        assert len(processor.received_images) == 2
        assert "pixel_values" in model_inputs
        assert "image_grid_thw" in model_inputs
        assert tuple(model_inputs["pixel_values"].shape) == (2, 3, 8, 8)
        assert tuple(model_inputs["image_grid_thw"].shape) == (2, 3)

    def test_predict_action_progress_done_and_memory(self):
        model = _DummyActor(progress_value=1.7, done_prob=0.9)
        processor = _DummyProcessor()
        wrapper = ThinkVLNActorNavigationModel(model=model, processor=processor, device="cpu")

        action, progress, done = wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(0, 0, 0)),
            instruction="go to the room",
            subgoal="turn right and move forward",
            episode_key="scene_001",
        )

        assert action == 2
        assert progress == 1.0
        assert done is True
        assert tuple(model.last_kwargs["action_labels"].shape) == (1, 4)
        assert tuple(model.last_kwargs["progress_labels"].shape) == (1,)
        assert tuple(model.last_kwargs["done_labels"].shape) == (1,)
        assert "Previous progress: 0.000" in processor.prompts[-1]

        model.progress_value = 0.2
        model.done_prob = 0.1
        action2, progress2, done2 = wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(0, 0, 0)),
            instruction="go to the room",
            subgoal="turn right and move forward",
            episode_key="scene_001",
        )
        assert action2 == 2
        assert progress2 == pytest.approx(0.2, rel=1e-5, abs=1e-6)
        assert done2 is False
        assert processor.last_image_count >= 2
        assert "Historical observations are provided." in processor.prompts[-1]
        assert "Predict the next 4 actions and the current-step subgoal progress." in processor.prompts[-1]

    def test_query_tokens_follow_training_interleaved_order(self):
        model = _DummyActor(
            progress_value=0.3,
            done_prob=0.4,
            query_token_layout="action_then_progress",
        )
        processor = _DummyProcessor()
        wrapper = ThinkVLNActorNavigationModel(model=model, processor=processor, device="cpu")

        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(5, 5, 5)),
            instruction="go straight",
            subgoal="move to hallway",
            episode_key="scene_010",
        )

        input_ids = model.last_kwargs["input_ids"][0].tolist()
        expected_tail = [151700, 151701, 151700, 151701, 151700, 151701, 151700, 151701]
        assert input_ids[-len(expected_tail):] == expected_tail

    def test_subgoal_switch_resets_previous_progress(self):
        model = _DummyActor(progress_value=0.6, done_prob=0.0)
        processor = _DummyProcessor()
        wrapper = ThinkVLNActorNavigationModel(model=model, processor=processor, device="cpu")

        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(1, 1, 1)),
            instruction="instr",
            subgoal="subgoal a",
            episode_key="scene_002",
        )
        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(1, 1, 1)),
            instruction="instr",
            subgoal="subgoal a",
            episode_key="scene_002",
        )
        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(1, 1, 1)),
            instruction="instr",
            subgoal="subgoal b",
            episode_key="scene_002",
        )

        assert "Previous progress: 0.000" in processor.prompts[0]
        assert "Previous progress: 0.600" in processor.prompts[1]
        assert "Previous progress: 0.000" in processor.prompts[2]

    def test_explicit_subtask_id_resets_progress_and_uses_memory_bank(self):
        model = _DummyActor(progress_value=0.4, done_prob=0.0)
        processor = _DummyProcessor()
        wrapper = ThinkVLNActorNavigationModel(model=model, processor=processor, device="cpu")

        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(40, 40, 40)),
            instruction="instr",
            subgoal="subgoal two",
            episode_key="scene_003",
            subtask_id=2,
        )
        assert "Previous progress: 0.000" in processor.prompts[-1]

        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(45, 45, 45)),
            instruction="instr",
            subgoal="subgoal two",
            episode_key="scene_003",
            subtask_id=2,
        )
        assert "Previous progress: 0.400" in processor.prompts[-1]
        assert processor.last_image_count >= 2

        model.progress_value = 0.7
        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(50, 50, 50)),
            instruction="instr",
            subgoal="subgoal three",
            episode_key="scene_003",
            subtask_id=3,
        )
        assert "Previous progress: 0.000" in processor.prompts[-1]

    def test_disable_memory_keeps_single_current_image(self):
        model = _DummyActor(progress_value=0.4, done_prob=0.0)
        processor = _DummyProcessor()
        wrapper = ThinkVLNActorNavigationModel(
            model=model,
            processor=processor,
            device="cpu",
            use_memory=False,
        )

        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(40, 40, 40)),
            instruction="instr",
            subgoal="subgoal two",
            episode_key="scene_003",
            subtask_id=2,
        )
        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(45, 45, 45)),
            instruction="instr",
            subgoal="subgoal two",
            episode_key="scene_003",
            subtask_id=2,
        )

        assert processor.last_image_count == 1
        assert "Historical observations are provided." not in processor.prompts[-1]


class _ReplayEnv:
    def __init__(self, frames):
        self.frames = frames
        self.current_episode = None
        self.episode_over = False
        self._frame_idx = 0
        self.step_actions = []

    def reset(self):
        self.episode_over = False
        self._frame_idx = 0
        self.step_actions = []
        return {"rgb": self.frames[0]}

    def step(self, action):
        self.step_actions.append(action)
        if self._frame_idx < len(self.frames) - 1:
            self._frame_idx += 1
        return {"rgb": self.frames[self._frame_idx]}

    def get_metrics(self):
        return {}


class TestSubtaskReplayMemoryPriming:
    def test_prepare_model_image_uses_rgb_only(self):
        rgb = np.full((8, 8, 3), 7, dtype=np.uint8)
        info = {
            "top_down_map": {
                "map": np.ones((8, 8), dtype=np.uint8),
                "fog_of_war_mask": np.ones((8, 8), dtype=np.uint8),
                "agent_map_coord": (4, 4),
                "agent_angle": 0.0,
            }
        }

        image = VLNEvaluator.prepare_model_image(None, rgb, info)

        assert image.size == (8, 8)
        assert np.array(image).tolist() == rgb.tolist()

    def test_replay_to_frame_primes_memory_path_and_resets_per_call(self):
        model = _DummyActor(progress_value=0.3, done_prob=0.0)
        processor = _DummyProcessor()
        nav_model = ThinkVLNActorNavigationModel(model=model, processor=processor, device="cpu")

        evaluator = VLNEvaluator.__new__(VLNEvaluator)
        evaluator.nav_model = nav_model

        frames = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8, 3), dtype=np.uint8),
            np.full((8, 8, 3), 2, dtype=np.uint8),
            np.full((8, 8, 3), 3, dtype=np.uint8),
        ]
        env = _ReplayEnv(frames)
        actions = [1, 1, 1]
        subtask_sequence = [1, 1, 2, 2]

        observations = evaluator._replay_to_frame_with_memory(
            env=env,
            episode=SimpleNamespace(episode_id="ep-1"),
            actions=actions,
            subtask_sequence=subtask_sequence,
            target_frame=3,
            episode_key="scene_004_ep-1",
        )

        assert observations["rgb"].tolist() == frames[3].tolist()
        assert nav_model.memory_bank_subtasks == [1, 1, 2]
        assert nav_model.subtask_start_bank_indices == [0, 2]
        assert len(nav_model.memory_bank_images) == 3
        assert all(image.size == (8, 8) for image in nav_model.memory_bank_images)

        evaluator._replay_to_frame_with_memory(
            env=env,
            episode=SimpleNamespace(episode_id="ep-1"),
            actions=actions,
            subtask_sequence=subtask_sequence,
            target_frame=1,
            episode_key="scene_004_ep-1",
        )
        assert nav_model.memory_bank_subtasks == [1]
        assert nav_model.subtask_start_bank_indices == [0]

    def test_warmup_subtask_one_turns_24_steps_and_keeps_6_memory_frames(self):
        model = _DummyActor(progress_value=0.3, done_prob=0.0)
        processor = _DummyProcessor()
        nav_model = ThinkVLNActorNavigationModel(model=model, processor=processor, device="cpu")

        evaluator = VLNEvaluator.__new__(VLNEvaluator)
        evaluator.nav_model = nav_model

        frames = [np.full((8, 8, 3), idx, dtype=np.uint8) for idx in range(25)]
        env = _ReplayEnv(frames)

        observations = evaluator._warmup_subtask_one_with_memory(
            env=env,
            episode=SimpleNamespace(episode_id="ep-2"),
            episode_key="scene_004_ep-2",
            subtask_id=1,
        )

        assert observations["rgb"].tolist() == frames[24].tolist()
        assert env.step_actions == [3] * 24
        assert nav_model.memory_bank_subtasks == [1] * 6
        assert nav_model.subtask_start_bank_indices == [0]
        assert len(nav_model.memory_bank_images) == 6

    def test_warmup_subtask_one_skips_non_first_subtask(self):
        model = _DummyActor(progress_value=0.3, done_prob=0.0)
        processor = _DummyProcessor()
        nav_model = ThinkVLNActorNavigationModel(model=model, processor=processor, device="cpu")

        evaluator = VLNEvaluator.__new__(VLNEvaluator)
        evaluator.nav_model = nav_model

        frames = [np.full((8, 8, 3), idx, dtype=np.uint8) for idx in range(3)]
        env = _ReplayEnv(frames)

        observations = evaluator._warmup_subtask_one_with_memory(
            env=env,
            episode=SimpleNamespace(episode_id="ep-3"),
            episode_key="scene_004_ep-3",
            subtask_id=2,
        )

        assert observations["rgb"].tolist() == frames[0].tolist()
        assert env.step_actions == []
        assert nav_model.memory_bank_subtasks == []

    def test_subtask_eval_forbids_stop_action(self, tmp_path, monkeypatch):
        model = _DummyActor(progress_value=0.3, done_prob=0.0)
        processor = _DummyProcessor()
        nav_model = ThinkVLNActorNavigationModel(model=model, processor=processor, device="cpu")

        evaluator = VLNEvaluator.__new__(VLNEvaluator)
        evaluator.nav_model = nav_model
        evaluator.output_path = str(tmp_path)
        evaluator.env_num = 1
        evaluator.args = SimpleNamespace(
            debug_log_interval=0,
            subtask_step_budget_factor=1.0,
            subgoal_success_distance=0.1,
        )
        evaluator.target_episode_key = ""
        evaluator.enable_step_debug = False
        evaluator.step_debug_format = "none"
        evaluator.config_path = "config/vln_r2r.yaml"
        evaluator._episode_instruction = lambda config_path, episode: "go forward"
        evaluator._iter_assigned_episodes = lambda env, idx: [("scene_010", env.episodes[0])]

        class _FakeSim:
            def __init__(self):
                self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            def get_agent_state(self):
                return SimpleNamespace(position=self.position.copy())

            def geodesic_distance(self, start, goal):
                return float(np.linalg.norm(np.array(start) - np.array(goal)))

        class _FakeEnv:
            def __init__(self):
                self.episodes = [SimpleNamespace(scene_id="scene_010.glb", episode_id="ep-1")]
                self.current_episode = None
                self.episode_over = False
                self.sim = _FakeSim()

            def reset(self):
                self.episode_over = False
                self.sim.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                return {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}

            def step(self, action):
                if int(action) == 1:
                    self.sim.position = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                return {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}

            def get_metrics(self):
                return {}

            def close(self):
                return None

        env = _FakeEnv()
        evaluator.config_env = lambda: env

        captured = {}

        def _predict(**kwargs):
            captured["forbidden_actions"] = kwargs.get("forbidden_actions")
            return 1, 0.5, False

        monkeypatch.setattr(nav_model, "predict_action_with_progress_and_done", _predict)
        monkeypatch.setattr(
            "thinkvln.eval.close_eval_runner.write_jsonl_record",
            lambda handle, payload, sync_to_disk=True: handle.write("ok\n"),
        )

        summary_full = {
            "scene_010_ep-1": {
                "actions": [1, 1],
                "subtask_sequence": [1, 1],
                "plan": ["go forward"],
            }
        }

        evaluator.eval_subtask_closed_loop(0, summary_full)

        assert captured["forbidden_actions"] == [0]


class TestReplayActionNormalization:
    def test_strip_leading_negative_one_sentinel(self):
        normalized, meta = VLNEvaluator._normalize_actions_for_replay([-1, 1, 2, 3], "scene_005_ep-1")
        assert normalized == [1, 2, 3]
        assert meta["leading_sentinel_stripped"] is True
        assert meta["original_len"] == 4
        assert meta["normalized_len"] == 3

    def test_keep_actions_without_sentinel(self):
        normalized, meta = VLNEvaluator._normalize_actions_for_replay([1, 2, 3], "scene_005_ep-2")
        assert normalized == [1, 2, 3]
        assert meta["leading_sentinel_stripped"] is False
        assert meta["original_len"] == 3
        assert meta["normalized_len"] == 3


class TestActorDebugSnapshot:
    def test_snapshot_contains_prompt_query_memory_and_progress(self):
        model = _DummyActor(progress_value=0.55, done_prob=0.2)
        processor = _DummyProcessor()
        wrapper = ThinkVLNActorNavigationModel(model=model, processor=processor, device="cpu")
        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(12, 12, 12)),
            instruction="go to kitchen",
            subgoal="walk to doorway",
            episode_key="scene_007",
            subtask_id=1,
        )

        snapshot = wrapper.get_last_debug_snapshot()
        assert snapshot is not None
        assert "prompt" in snapshot
        assert "query_token_ids" in snapshot
        assert "selected_indices" in snapshot
        assert "selected_subtask_ids" in snapshot
        assert "prev_progress_input" in snapshot
        assert "predicted_progress" in snapshot
        assert "memory_bank_size" in snapshot
        assert "memory_bank_subtasks" in snapshot
        assert "selected_images" in snapshot
        assert snapshot["query_token_ids"] == [151700, 151701, 151700, 151701, 151700, 151701, 151700, 151701]
        assert snapshot["memory_bank_size"] >= 1
        assert len(snapshot["selected_images"]) >= 1
