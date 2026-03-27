from types import SimpleNamespace

from thinkvln.models.navigation_model import StreamVLNNavigationModel
from thinkvln.eval.close_eval_runner import supports_progress_done_actor_model


class _FakeModel:
    def __init__(self):
        self.reset_calls = []
        self._vision_tower = SimpleNamespace(image_processor=SimpleNamespace(crop_size={"height": 384, "width": 384}))

    def get_vision_tower(self):
        return self._vision_tower

    def reset(self, world_size):
        self.reset_calls.append(("reset", int(world_size)))

    def reset_for_env(self, env_id):
        self.reset_calls.append(("reset_for_env", int(env_id)))

    def eval(self):
        return None


def test_streamvln_navigation_model_init_resets_episode_state():
    model = _FakeModel()
    nav = StreamVLNNavigationModel(
        model=model,
        tokenizer=object(),
        device="cpu",
        env_id=3,
    )

    assert isinstance(nav.action_seq, list)
    assert nav.step_count == 0
    assert ("reset", 1) in model.reset_calls
    assert ("reset_for_env", 3) in model.reset_calls


def test_streamvln_navigation_model_formats_actor_text_lines():
    text = StreamVLNNavigationModel._augment_instruction(
        instruction="Walk to the sink.",
        subgoal="Turn right into the bathroom.",
        hint="You already cleared the dining area.",
    )

    assert "Instruction: Walk to the sink." in text
    assert "Current subtask: Turn right into the bathroom." in text
    assert "Watcher hint: You already cleared the dining area." in text


def test_supports_progress_done_actor_model_uses_capability_not_class():
    class _CapableNavModel:
        def predict_action_with_progress_and_done(self, *args, **kwargs):
            return 0, 0.0, False

    assert supports_progress_done_actor_model(_CapableNavModel()) is True
    assert supports_progress_done_actor_model(object()) is False
