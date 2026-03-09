import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from thinkvln.eval.close_eval_runner import VLNEvaluator
from thinkvln.eval.close_eval_cli import build_parser
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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prompt = messages[0]["content"][-1]["text"]
        self.prompts.append(prompt)
        self.last_image_count = sum(1 for item in messages[0]["content"] if item["type"] == "image")
        return "prompt"

    def __call__(self, text, images, return_tensors, padding):
        assert len(text) == 1
        assert len(images) >= 1
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
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


class TestCloseEvalCliArgs:
    def test_parser_includes_minimal_recovery_args(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.startup_scan_turns == 0
        assert args.recovery_turn_steps == 2

    @pytest.mark.parametrize(
        "arg_name",
        ["--startup_scan_turns", "--recovery_turn_steps"],
    )
    def test_parser_rejects_negative_recovery_args(self, arg_name):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([arg_name, "-1"])


class TestStuckRecoveryHelpers:
    def test_collision_count_supports_count_and_is_collision_shapes(self):
        count, delta, is_collision = VLNEvaluator._collision_info_to_count(
            {"collisions": {"count": 3}},
            prev_collision_count=2.0,
        )
        assert count == 3.0
        assert delta == 1.0
        assert is_collision is True

        count2, delta2, is_collision2 = VLNEvaluator._collision_info_to_count(
            {"collisions": {"is_collision": True}},
            prev_collision_count=3.0,
        )
        assert count2 == 4.0
        assert delta2 == 1.0
        assert is_collision2 is True

    def test_detect_stuck_uses_forward_collision_or_window_stationary(self):
        recent_steps = [
            {"step_displacement": 0.01, "distance_improve": 0.0},
            {"step_displacement": 0.01, "distance_improve": 0.0},
            {"step_displacement": 0.01, "distance_improve": 0.0},
        ]
        stuck_collision, reason_collision = VLNEvaluator._detect_stuck(
            executed_action=1,
            collision_delta=1.0,
            recent_steps=recent_steps,
        )
        assert stuck_collision is True
        assert reason_collision == "forward_collision"

        stuck_window, reason_window = VLNEvaluator._detect_stuck(
            executed_action=2,
            collision_delta=0.0,
            recent_steps=recent_steps,
        )
        assert stuck_window is True
        assert reason_window == "stationary_no_progress"

    def test_recovery_trigger_blocked_by_cooldown_and_zero_steps(self):
        assert VLNEvaluator._can_trigger_recovery(
            recovery_turn_steps=2,
            cooldown_remaining=1,
            startup_scan_phase=False,
            recovery_active=False,
        ) is False
        assert VLNEvaluator._can_trigger_recovery(
            recovery_turn_steps=0,
            cooldown_remaining=0,
            startup_scan_phase=False,
            recovery_active=False,
        ) is False
        assert VLNEvaluator._can_trigger_recovery(
            recovery_turn_steps=2,
            cooldown_remaining=0,
            startup_scan_phase=False,
            recovery_active=False,
        ) is True


class TestActorWrapper:
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
