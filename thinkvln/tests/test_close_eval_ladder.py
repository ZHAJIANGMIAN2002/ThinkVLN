import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from thinkvln.models.navigation_model import ThinkVLNActorNavigationModel

_CLOSE_EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "close_eval.py"
_SPEC = importlib.util.spec_from_file_location("close_eval_for_test", _CLOSE_EVAL_PATH)
_CLOSE_EVAL = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_CLOSE_EVAL)

build_subtask_spans = _CLOSE_EVAL.build_subtask_spans
compute_step_budget = _CLOSE_EVAL.compute_step_budget
oracle_subtask_at_step = _CLOSE_EVAL.oracle_subtask_at_step
summarize_subtask_aggregation = _CLOSE_EVAL.summarize_subtask_aggregation
timeline_progress = _CLOSE_EVAL.timeline_progress


class CloseEvalLadderUtilsTest(unittest.TestCase):
    def test_build_subtask_spans_extracts_contiguous_ranges(self):
        spans = build_subtask_spans([1, 1, 2, 2, 2, 3, 1, 1])
        self.assertEqual(spans, [(1, 0, 1), (2, 2, 4), (3, 5, 5), (1, 6, 7)])

    def test_timeline_progress_handles_zero_and_normal_cases(self):
        self.assertEqual(timeline_progress(0, 0), 1.0)
        self.assertAlmostEqual(timeline_progress(0, 4), 0.25)
        self.assertAlmostEqual(timeline_progress(3, 4), 1.0)
        self.assertAlmostEqual(timeline_progress(10, 4), 1.0)

    def test_compute_step_budget_uses_ceiling_and_min_one(self):
        self.assertEqual(compute_step_budget(0, 2.0), 1)
        self.assertEqual(compute_step_budget(3, 2.0), 6)
        self.assertEqual(compute_step_budget(3, 1.2), 4)

    def test_oracle_subtask_lookup_by_step_index(self):
        spans = build_subtask_spans([1, 1, 2, 2, 2, 3, 3])
        self.assertEqual(oracle_subtask_at_step(0, spans), 1)
        self.assertEqual(oracle_subtask_at_step(2, spans), 2)
        self.assertEqual(oracle_subtask_at_step(6, spans), 3)
        self.assertEqual(oracle_subtask_at_step(100, spans), 3)

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
        self.assertAlmostEqual(summary["subtask_success_rate"], 0.75)
        self.assertAlmostEqual(summary["steps_to_subgoal"], 4.0)
        self.assertAlmostEqual(summary["progress_mae"], 0.25)


class _DummyProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def __call__(self, text, images, return_tensors, padding):
        self._assert(text, images)
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    @staticmethod
    def _assert(text, images):
        assert len(text) == 1
        assert len(images) == 1


class _DummyActor(torch.nn.Module):
    def __init__(self, progress_value: float):
        super().__init__()
        self.config = SimpleNamespace(
            num_query_tokens=4,
            action_query_token_id=151700,
            progress_query_token_id=151701,
        )
        self.progress_value = progress_value
        self.last_kwargs = None

    def eval(self):
        return self

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        action_logits = torch.tensor(
            [[[0.1, 0.2, 0.8, 0.0], [0.3, 0.4, 0.2, 0.1], [0.4, 0.2, 0.1, 0.3], [0.9, 0.0, 0.0, 0.1]]],
            dtype=torch.float32,
        )
        progress_preds = torch.tensor(
            [[self.progress_value, 0.1, 0.2, 0.3]],
            dtype=torch.float32,
        )
        return {
            "action_logits": action_logits,
            "progress_preds": progress_preds,
        }


class ActorWrapperTest(unittest.TestCase):
    def test_predict_action_with_progress_uses_first_step_and_clips(self):
        for progress_value, expected_progress in [(1.7, 1.0), (-0.2, 0.0)]:
            with self.subTest(progress_value=progress_value):
                model = _DummyActor(progress_value=progress_value)
                wrapper = ThinkVLNActorNavigationModel(
                    model=model,
                    processor=_DummyProcessor(),
                    device="cpu",
                )
                action, progress = wrapper.predict_action_with_progress(
                    observation=Image.new("RGB", (8, 8), color=(0, 0, 0)),
                    instruction="go to the room",
                    subgoal="turn right and move forward",
                )

                self.assertEqual(action, 2)
                self.assertAlmostEqual(progress, expected_progress)
                self.assertEqual(tuple(model.last_kwargs["action_labels"].shape), (1, 4))
                self.assertEqual(tuple(model.last_kwargs["progress_labels"].shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
