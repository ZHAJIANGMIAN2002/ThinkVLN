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
        previous_progress=0.5,
    )

    assert "Instruction: Walk to the sink." in text
    assert "Current subtask: Turn right into the bathroom." in text
    assert "Watcher hint: You already cleared the dining area." in text
    assert "Previous progress: 0.5000" in text


def test_supports_progress_done_actor_model_uses_capability_not_class():
    class _CapableNavModel:
        def predict_action_with_progress_and_done(self, *args, **kwargs):
            return 0, 0.0, False

    assert supports_progress_done_actor_model(_CapableNavModel()) is True
    assert supports_progress_done_actor_model(object()) is False


def test_initialize_vision_tokenizer_does_not_shrink_padded_vocab():
    import torch
    from llava.model.llava_arch import LlavaMetaForCausalLM

    class _DummyTokenizer:
        def __len__(self):
            return 151647

        def add_tokens(self, tokens, special_tokens=True):
            return len(tokens)

    class _DummyLlavaModel(LlavaMetaForCausalLM):
        def __init__(self):
            self.resize_calls = []
            self.input_embeddings = torch.nn.Embedding(152064, 8)
            self.output_embeddings = torch.nn.Linear(8, 152064, bias=False)

        def get_model(self):
            return self

        def resize_token_embeddings(self, new_size):
            self.resize_calls.append(int(new_size))

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return self.output_embeddings

    model = _DummyLlavaModel()
    tokenizer = _DummyTokenizer()
    model_args = SimpleNamespace(
        mm_use_im_patch_token=True,
        mm_use_im_start_end=False,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    )

    model.initialize_vision_tokenizer(model_args, tokenizer)

    assert model.resize_calls == []
