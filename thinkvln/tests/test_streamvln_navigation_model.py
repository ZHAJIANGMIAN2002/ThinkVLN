from types import SimpleNamespace

import pytest
import torch

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


class _FakeStreamTokenizer:
    def batch_decode(self, sequences, skip_special_tokens=False):
        del sequences, skip_special_tokens
        return ["↑ ← → STOP"]


class _FakeStreamModel:
    def __init__(self):
        self.reset_calls = []
        self.generate_calls = 0
        self.progress_calls = 0
        self.generate_kwargs = []
        self._vision_tower = SimpleNamespace(
            image_processor=SimpleNamespace(
                crop_size={"height": 384, "width": 384},
                preprocess=lambda images, return_tensors: {
                    "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32)
                },
            )
        )

    def get_vision_tower(self):
        return self._vision_tower

    def reset(self, world_size):
        self.reset_calls.append(("reset", int(world_size)))

    def reset_for_env(self, env_id):
        self.reset_calls.append(("reset_for_env", int(env_id)))

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls += 1
        self.generate_kwargs.append(dict(kwargs))
        return SimpleNamespace(
            sequences=torch.tensor([[1, 2, 3]], dtype=torch.long),
            past_key_values=("kv", self.generate_calls),
        )

    def predict_progress_done(self, **kwargs):
        del kwargs
        self.progress_calls += 1
        return (
            torch.tensor([0.9], dtype=torch.float32),
            torch.tensor([0.8], dtype=torch.float32),
        )


def _patch_streamvln_test_runtime(monkeypatch, nav):
    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "depth_camera_filtering",
        types.SimpleNamespace(filter_depth=lambda depth, blur_type=None: depth),
    )
    monkeypatch.setitem(
        sys.modules,
        "streamvln.utils.utils",
        types.SimpleNamespace(
            DEFAULT_MEMORY_TOKEN="<memory>",
            DEFAULT_VIDEO_TOKEN="<video>",
            dict_to_cuda=lambda payload, device: payload,
        ),
    )
    monkeypatch.setattr(
        nav,
        "_preprocess_qwen",
        lambda sources, has_image=False, add_system=False: (torch.tensor([[7, 8]], dtype=torch.long), ["prompt"]),
    )
    monkeypatch.setattr(
        nav,
        "_preprocess_depth_image",
        lambda depth_image, do_depth_scale=True, depth_scale=1000: (torch.zeros((8, 8), dtype=torch.float32).numpy(), (8, 8)),
    )
    monkeypatch.setattr(nav, "_get_axis_align_matrix", lambda: torch.eye(4, dtype=torch.float64))


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


def test_streamvln_navigation_model_reuses_chunk_cache_and_emits_fresh_metadata_once(monkeypatch):
    model = _FakeStreamModel()
    nav = StreamVLNNavigationModel(
        model=model,
        tokenizer=_FakeStreamTokenizer(),
        device="cpu",
        env_id=2,
    )
    _patch_streamvln_test_runtime(monkeypatch, nav)

    observation = {"rgb": torch.zeros((8, 8, 3), dtype=torch.uint8).numpy()}

    first = nav.predict_action_with_progress_and_done(
        observation=observation,
        instruction="go to the sink",
        subgoal="enter the hall",
        episode_key="scene_1",
    )
    snapshot1 = nav.get_last_debug_snapshot()
    second = nav.predict_action_with_progress_and_done(
        observation=observation,
        instruction="go to the sink",
        subgoal="enter the hall",
        episode_key="scene_1",
    )
    snapshot2 = nav.get_last_debug_snapshot()
    third = nav.predict_action_with_progress_and_done(
        observation=observation,
        instruction="go to the sink",
        subgoal="enter the hall",
        episode_key="scene_1",
    )
    fourth = nav.predict_action_with_progress_and_done(
        observation=observation,
        instruction="go to the sink",
        subgoal="enter the hall",
        episode_key="scene_1",
    )
    fifth = nav.predict_action_with_progress_and_done(
        observation=observation,
        instruction="go to the sink",
        subgoal="enter the hall",
        episode_key="scene_1",
    )

    assert first[0] == 1
    assert first[1] == pytest.approx(0.9, rel=1e-5, abs=1e-6)
    assert first[2] is True
    assert second[0] == 2
    assert third[0] == 3
    assert fourth[0] == 0
    assert fifth[0] == 1
    assert model.generate_calls == 2
    assert model.progress_calls == 2
    assert snapshot1["fresh_actor_metadata"] is True
    assert snapshot1["used_cached_action_seq"] is False
    assert snapshot2["fresh_actor_metadata"] is False
    assert snapshot2["used_cached_action_seq"] is True


def test_streamvln_navigation_model_stop_clears_remaining_cached_actions(monkeypatch):
    model = _FakeStreamModel()
    tokenizer = _FakeStreamTokenizer()
    tokenizer.batch_decode = lambda sequences, skip_special_tokens=False: ["↑ STOP ← →"]
    nav = StreamVLNNavigationModel(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        env_id=2,
    )
    _patch_streamvln_test_runtime(monkeypatch, nav)

    observation = {"rgb": torch.zeros((8, 8, 3), dtype=torch.uint8).numpy()}

    assert nav.predict_action_with_progress_and_done(observation, "instr", "subgoal", episode_key="scene_2")[0] == 1
    assert nav.predict_action_with_progress_and_done(observation, "instr", "subgoal", episode_key="scene_2")[0] == 0
    assert model.generate_calls == 1
    assert nav.predict_action_with_progress_and_done(observation, "instr", "subgoal", episode_key="scene_2")[0] == 1
    assert model.generate_calls == 2


def test_streamvln_navigation_model_cached_steps_still_update_history_and_step_count(monkeypatch):
    model = _FakeStreamModel()
    nav = StreamVLNNavigationModel(
        model=model,
        tokenizer=_FakeStreamTokenizer(),
        device="cpu",
        env_id=2,
        num_frames=8,
    )
    _patch_streamvln_test_runtime(monkeypatch, nav)

    observations = [
        {"rgb": torch.full((8, 8, 3), fill_value=i, dtype=torch.uint8).numpy()}
        for i in range(4)
    ]

    for obs in observations:
        nav.predict_action_with_progress_and_done(obs, "instr", "subgoal", episode_key="scene_hist")

    assert nav.step_count == 4
    assert len(nav.rgb_list) == 4
    assert len(nav.depth_list) == 4
    assert len(nav.pose_list) == 4
    assert len(nav.intrinsic_list) == 4
    assert nav.time_ids == [0, 1, 2, 3]
    assert model.generate_calls == 1


def test_streamvln_navigation_model_num_frames_reset_follows_real_env_steps_not_actor_calls(monkeypatch):
    model = _FakeStreamModel()
    nav = StreamVLNNavigationModel(
        model=model,
        tokenizer=_FakeStreamTokenizer(),
        device="cpu",
        env_id=2,
        num_frames=4,
    )
    _patch_streamvln_test_runtime(monkeypatch, nav)

    observations = [
        {"rgb": torch.full((8, 8, 3), fill_value=i, dtype=torch.uint8).numpy()}
        for i in range(5)
    ]

    for obs in observations[:4]:
        nav.predict_action_with_progress_and_done(obs, "instr", "subgoal", episode_key="scene_reset")

    assert len(nav.rgb_list) == 4
    assert nav.step_count == 4

    nav.predict_action_with_progress_and_done(observations[4], "instr", "subgoal", episode_key="scene_reset")

    assert ("reset_for_env", 2) in model.reset_calls
    assert nav.step_count == 5
    assert len(nav.rgb_list) == 1


def test_streamvln_navigation_model_does_not_reuse_text_outputs_or_kv_between_fresh_calls(monkeypatch):
    model = _FakeStreamModel()
    nav = StreamVLNNavigationModel(
        model=model,
        tokenizer=_FakeStreamTokenizer(),
        device="cpu",
        env_id=2,
        num_frames=32,
    )
    _patch_streamvln_test_runtime(monkeypatch, nav)

    observations = [
        {"rgb": torch.full((8, 8, 3), fill_value=i, dtype=torch.uint8).numpy()}
        for i in range(5)
    ]

    for obs in observations[:4]:
        nav.predict_action_with_progress_and_done(obs, "instr", "subgoal", episode_key="scene_cache")

    nav.predict_action_with_progress_and_done(observations[4], "instr", "subgoal", episode_key="scene_cache")

    assert model.generate_calls == 2
    assert model.generate_kwargs[0].get("past_key_values") is None
    assert model.generate_kwargs[1].get("past_key_values") is None
    assert tuple(model.generate_kwargs[0]["inputs"].shape) == (1, 2)
    assert tuple(model.generate_kwargs[1]["inputs"].shape) == (1, 2)
