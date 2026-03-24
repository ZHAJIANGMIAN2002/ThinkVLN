import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import thinkvln.eval.two_system_eval as two_system_eval
from thinkvln.models.navigation_model import ThinkVLNActorNavigationModel

from thinkvln.eval.two_system_eval import (
    ApiWatcherBackend,
    LocalWatcherBackend,
    TwoSystemEpisodeRunner,
    WatcherDecision,
    WatcherTodoState,
    build_init_prompt,
    build_parser,
    build_update_prompt,
    load_config,
)


class _DummyProcessor:
    def __init__(self):
        self.prompts = []
        self.last_image_count = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        content = messages[-1]["content"]
        prompt = content[-1]["text"]
        self.prompts.append(prompt)
        self.last_image_count = sum(1 for item in content if item["type"] == "image")
        return "prompt"

    def __call__(self, text, images, return_tensors, padding):
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "pixel_values": torch.ones((len(images), 3, 8, 8), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 8, 8]] * len(images), dtype=torch.long),
        }


class _DummyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_query_tokens=4,
            action_query_token_id=151700,
            progress_query_token_id=151701,
        )

    def eval(self):
        return self

    def forward(self, **kwargs):
        return {
            "action_logits": torch.tensor(
                [[[0.1, 0.9, 0.2, 0.0], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]],
                dtype=torch.float32,
            ),
            "progress_preds": torch.tensor([0.4], dtype=torch.float32),
            "done_preds": torch.tensor([0.1], dtype=torch.float32),
        }


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._responses.pop(0))


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


class _FakeWatcher:
    def __init__(self):
        self.init_calls = []
        self.update_calls = []

    def initialize(self, instruction, plan_steps, first_observation, episode_key):
        self.init_calls.append(
            {
                "instruction": instruction,
                "plan_steps": list(plan_steps),
                "episode_key": episode_key,
            }
        )
        return WatcherDecision(
            memory="start memory",
            done=False,
            subtask="reach the doorway",
            raw_response={"memory": "start memory", "done": False, "subtask": "reach the doorway"},
            wakeup_reason="init",
        )

    def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
        self.update_calls.append(
            {
                "instruction": instruction,
                "todo_state": todo_state,
                "memory_text": memory_text,
                "rollout_slice_len": len(rollout_slice),
                "episode_key": episode_key,
            }
        )
        if len(self.update_calls) == 1:
            return WatcherDecision(
                memory="hall memory",
                done=True,
                subtask="approach the sink",
                raw_response={"memory": "hall memory", "done": True, "subtask": "approach the sink"},
                wakeup_reason="progress_threshold",
            )
        return WatcherDecision(
            memory="final memory",
            done=True,
            subtask="stop",
            raw_response={"memory": "final memory", "done": True, "subtask": "stop"},
            wakeup_reason="stop_action",
        )


class _LoopingWatcher:
    def initialize(self, instruction, plan_steps, first_observation, episode_key):
        del instruction, plan_steps, first_observation, episode_key
        return WatcherDecision(
            memory="loop memory",
            done=False,
            subtask="keep moving",
            raw_response={"memory": "loop memory", "done": False, "subtask": "keep moving"},
            wakeup_reason="init",
        )

    def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
        del instruction, todo_state, memory_text, rollout_slice, episode_key
        return WatcherDecision(
            memory="loop memory",
            done=False,
            subtask="keep moving",
            raw_response={"memory": "loop memory", "done": False, "subtask": "keep moving"},
            wakeup_reason="max_steps_per_wakeup",
        )


class _FailingWatcher:
    def initialize(self, instruction, plan_steps, first_observation, episode_key):
        del instruction, plan_steps, first_observation, episode_key
        return WatcherDecision(
            memory="start memory",
            done=False,
            subtask="move forward",
            raw_response={"memory": "start memory", "done": False, "subtask": "move forward"},
            wakeup_reason="init",
        )

    def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
        del instruction, todo_state, memory_text, rollout_slice, episode_key
        raise RuntimeError("api down")


class _FailingInitWatcher:
    def initialize(self, instruction, plan_steps, first_observation, episode_key):
        del instruction, plan_steps, first_observation, episode_key
        raise RuntimeError("boot failed")


class _EpisodeOverWatcher:
    def initialize(self, instruction, plan_steps, first_observation, episode_key):
        del instruction, plan_steps, first_observation, episode_key
        return WatcherDecision(
            memory="start memory",
            done=False,
            subtask="move forward",
            raw_response={"memory": "start memory", "done": False, "subtask": "move forward"},
            wakeup_reason="init",
        )

    def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
        del instruction, todo_state, memory_text, rollout_slice, episode_key
        return WatcherDecision(
            memory="still going",
            done=False,
            subtask="move forward",
            raw_response={"memory": "still going", "done": False, "subtask": "move forward"},
            wakeup_reason="env_episode_over",
        )


class _StopBeforeStepWatcher:
    def initialize(self, instruction, plan_steps, first_observation, episode_key):
        del instruction, plan_steps, first_observation, episode_key
        return WatcherDecision(
            memory="start memory",
            done=False,
            subtask="stop if needed",
            raw_response={"memory": "start memory", "done": False, "subtask": "stop if needed"},
            wakeup_reason="init",
        )

    def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
        del instruction, todo_state, memory_text, episode_key
        assert len(rollout_slice) == 1
        assert rollout_slice[0]["action"] == "stop"
        return WatcherDecision(
            memory="done",
            done=True,
            subtask="stop",
            raw_response={"memory": "done", "done": True, "subtask": "stop"},
            wakeup_reason="stop_action",
        )


class _FakeNavModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def eval(self):
        return self

    def predict_action_with_progress_and_done(
        self,
        observation,
        instruction,
        subgoal,
        episode_key=None,
        subtask_id=None,
        hint=None,
        forbidden_actions=None,
    ):
        self.calls.append(
            {
                "instruction": instruction,
                "subgoal": subgoal,
                "episode_key": episode_key,
                "subtask_id": subtask_id,
                "hint": hint,
                "forbidden_actions": list(forbidden_actions or []),
            }
        )
        return self.responses.pop(0)


class _FakeEnv:
    def __init__(self):
        self.current_episode = None
        self.episode_over = False
        self.step_count = 0
        self._frames = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8, 3), dtype=np.uint8),
            np.full((8, 8, 3), 2, dtype=np.uint8),
            np.full((8, 8, 3), 3, dtype=np.uint8),
        ]
        self.sim = SimpleNamespace(get_agent_state=self._get_agent_state)

    def _get_agent_state(self):
        return SimpleNamespace(
            position=np.array([float(self.step_count), 0.0, 0.0], dtype=np.float32),
            rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def reset(self):
        self.step_count = 0
        self.episode_over = False
        return {"rgb": self._frames[0]}

    def step(self, action):
        del action
        self.step_count = min(self.step_count + 1, len(self._frames) - 1)
        if self.step_count >= len(self._frames) - 1:
            self.episode_over = True
        return {"rgb": self._frames[self.step_count]}

    def get_metrics(self):
        return {
            "success": 1.0 if self.step_count >= 3 else 0.0,
            "distance_to_goal": float(max(0, 3 - self.step_count)),
        }


class _LongFakeEnv:
    def __init__(self):
        self.current_episode = None
        self.episode_over = False
        self.step_count = 0
        self.sim = SimpleNamespace(get_agent_state=self._get_agent_state)

    def _get_agent_state(self):
        return SimpleNamespace(
            position=np.array([float(self.step_count), 0.0, 0.0], dtype=np.float32),
            rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def reset(self):
        self.step_count = 0
        self.episode_over = False
        return {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}

    def step(self, action):
        del action
        self.step_count += 1
        return {"rgb": np.full((8, 8, 3), self.step_count % 255, dtype=np.uint8)}

    def get_metrics(self):
        return {"success": 0.0, "distance_to_goal": float(max(0, 20 - self.step_count))}


class _StagnantEnv:
    def __init__(self):
        self.current_episode = None
        self.episode_over = False
        self.step_count = 0
        self._frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.sim = SimpleNamespace(get_agent_state=self._get_agent_state)

    def _get_agent_state(self):
        return SimpleNamespace(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def reset(self):
        self.step_count = 0
        self.episode_over = False
        return {"rgb": self._frame}

    def step(self, action):
        del action
        self.step_count += 1
        return {"rgb": self._frame}

    def get_metrics(self):
        return {"success": 0.0, "distance_to_goal": 9.0}


class _EpisodeOverEnv:
    def __init__(self):
        self.current_episode = None
        self.episode_over = False
        self.step_count = 0
        self._frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.sim = SimpleNamespace(get_agent_state=self._get_agent_state)

    def _get_agent_state(self):
        return SimpleNamespace(
            position=np.array([float(self.step_count), 0.0, 0.0], dtype=np.float32),
            rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def reset(self):
        self.step_count = 0
        self.episode_over = False
        return {"rgb": self._frame}

    def step(self, action):
        del action
        assert self.episode_over is False, "Episode over, call reset before calling step"
        self.step_count += 1
        self.episode_over = True
        return {"rgb": self._frame}

    def get_metrics(self):
        return {"success": 0.0, "distance_to_goal": 5.0}


class _NoStepEnv:
    def __init__(self):
        self.current_episode = None
        self.episode_over = False
        self.sim = SimpleNamespace(get_agent_state=self._get_agent_state)

    def _get_agent_state(self):
        return SimpleNamespace(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def reset(self):
        self.episode_over = False
        return {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}

    def step(self, action):
        raise AssertionError(f"env.step should not be called, got action={action}")

    def get_metrics(self):
        return {"success": 0.0, "distance_to_goal": 1.0}


def _write_config(path: Path) -> None:
    payload = {
        "actor": {
            "model_type": "thinkvln_actor",
            "model_path": "/tmp/actor",
            "base_model_path": "/tmp/base",
            "device": "cpu",
            "memory_num_history_images": 6,
            "done_threshold": 0.85,
        },
        "watcher": {
            "backend": "api",
            "model_name": "gpt-test",
            "model_path": "/tmp/watcher",
            "base_model_path": "/tmp/watcher-base",
            "image_stride": 3,
            "api_base_url": "http://localhost:11451/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "reasoning_effort": "none",
            "request_timeout": 30.0,
            "max_retries": 2,
        },
        "env": {
            "habitat_config_path": "config/vln_r2r.yaml",
            "summary_full_path": "data/trajectory_data/R2R_back/summary_full.jsonl",
            "eval_split": "train",
            "sample_rate": 1.0,
            "target_episode_key": "",
        },
        "rollout": {
            "max_steps_per_wakeup": 10,
            "progress_threshold": 0.85,
            "max_watcher_wakeups": 10,
            "episode_step_cap": 50,
            "forbidden_actions": [0],
        },
        "output": {
            "output_dir": "results/two_system_eval",
            "save_trace_jsonl": True,
            "save_summary_json": True,
            "save_step_debug_html": False,
            "save_debug_video": False,
            "debug_video_fps": 2,
        },
        "debug": {
            "enabled": False,
            "episode_key": "",
            "output_dir": "",
            "video_fps": 2,
        },
        "runtime": {
            "log_level": "INFO",
            "world_size": 1,
            "dist_url": "env://",
            "dist_timeout_minutes": 120,
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class TestTwoSystemEval:
    def test_parser_uses_config_only(self):
        parser = build_parser()
        flags = set(parser._option_string_actions.keys())

        assert "--config" in flags
        assert "--model_path" not in flags
        assert "--summary_full_path" not in flags

    def test_load_config_reads_all_sections(self, tmp_path: Path):
        config_path = tmp_path / "two_system_eval.yaml"
        _write_config(config_path)

        config = load_config(config_path)

        assert set(config.keys()) == {"actor", "watcher", "env", "rollout", "output", "debug", "runtime"}
        assert config["watcher"]["backend"] == "api"
        assert config["env"]["summary_full_path"].endswith("summary_full.jsonl")
        assert config["debug"]["enabled"] is False

    def test_sample_config_exists_and_is_complete(self):
        config = load_config("config/two_system_eval.yaml")

        assert config["actor"]["model_type"] in {"thinkvln_actor", "thinkvln_fm_actor"}
        assert "summary_full_path" in config["env"]
        assert config["watcher"]["image_stride"] == 3
        assert config["rollout"]["max_steps_per_wakeup"] == 10
        assert config["debug"]["enabled"] is False

    def test_local_debug20_config_exists_and_uses_local_watcher(self):
        config = load_config("config/two_system_eval.local_debug20.yaml")

        assert config["watcher"]["backend"] == "local"
        assert config["watcher"]["model_path"] == "/mnt/swx/ThinkVLN/model_weights/qwen3vl-8"
        assert config["output"]["save_debug_video"] is True
        assert config["debug"]["sample_limit"] == 20
        assert config["debug"]["sample_seed"] == 0
        assert config["rollout"]["max_watcher_wakeups"] == 10

    def test_prompt_builders_use_unified_schema(self):
        image = Image.new("RGB", (8, 8), color=(1, 2, 3))
        todo = WatcherTodoState(done_steps=["leave the bedroom"], active_step="enter the hall", pending_steps=["stop"])

        init_prompt = build_init_prompt(
            instruction="go to the sink",
            plan_steps=["leave the bedroom", "enter the hall", "stop"],
            first_observation=image,
        )
        update_prompt = build_update_prompt(
            instruction="go to the sink",
            todo_state=todo,
            memory_text="hall memory",
            rollout_images=[image],
            rollout_actions=["forward", "turn_right"],
        )

        assert '{"memory":"...","done":false,"subtask":"..."}' in init_prompt.system_prompt
        assert '{"memory":"...","done":true/false,"subtask":"..."}' in update_prompt.system_prompt
        assert "Plan" in init_prompt.user_text
        assert "Done" in update_prompt.user_text
        assert "Active" in update_prompt.user_text
        assert "Pending" in update_prompt.user_text

    def test_update_prompt_includes_plan_progress_counts(self):
        prompt = build_update_prompt(
            instruction="go to the sink",
            todo_state=WatcherTodoState(
                done_steps=["leave the bedroom"],
                active_step="enter the hall",
                pending_steps=["approach the sink", "stop"],
            ),
            memory_text="hall memory",
            rollout_images=[Image.new("RGB", (8, 8), color=(1, 2, 3))],
            rollout_actions=["forward"],
        )

        assert "Plan Progress" in prompt.user_text
        assert "1/4 completed" in prompt.user_text

    def test_api_watcher_backend_normalizes_payload(self):
        backend = ApiWatcherBackend(
            client=_FakeClient(['{"memory":"state","done":false,"subtask":"move forward"}']),
            model_name="gpt-test",
            request_timeout=10.0,
            max_retries=1,
            reasoning_effort="none",
        )

        decision = backend.initialize(
            instruction="go ahead",
            plan_steps=["move forward"],
            first_observation=Image.new("RGB", (8, 8), color=(0, 0, 0)),
            episode_key="scene_1",
        )

        assert decision.memory == "state"
        assert decision.done is False
        assert decision.subtask == "move forward"

    def test_api_watcher_backend_retries_invalid_json(self):
        backend = ApiWatcherBackend(
            client=_FakeClient(["not json", '{"memory":"state","done":true,"subtask":"stop"}']),
            model_name="gpt-test",
            request_timeout=10.0,
            max_retries=2,
            reasoning_effort="none",
        )

        decision = backend.initialize(
            instruction="go ahead",
            plan_steps=["move forward"],
            first_observation=Image.new("RGB", (8, 8), color=(0, 0, 0)),
            episode_key="scene_1",
        )

        assert decision.done is True
        assert len(backend.client.chat.completions.calls) == 2

    def test_api_watcher_backend_update_uses_datagen_rollout_prompt(self):
        backend = ApiWatcherBackend(
            client=_FakeClient(['{"memory_end":"hall; near door; step ongoing","done":false,"next_subtask":"continue forward"}']),
            model_name="qwen-test",
            request_timeout=10.0,
            max_retries=1,
            reasoning_effort="none",
        )

        decision = backend.update(
            instruction="go ahead",
            todo_state=WatcherTodoState(
                done_steps=["leave bedroom"],
                active_step="enter hall",
                pending_steps=["approach sink"],
            ),
            memory_text="left the room; at doorway; step ongoing",
            rollout_slice=[
                {
                    "image": Image.new("RGB", (8, 8), color=(1, 2, 3)),
                    "action": "forward",
                    "step_index": 0,
                }
            ],
            episode_key="scene_1",
        )

        call = backend.client.chat.completions.calls[0]
        user_text = call["messages"][1]["content"][0]["text"]

        assert call["model"] == "qwen-test"
        assert "Plan state\nDone\n- leave bedroom\nActive\n- enter hall\nPending\n- approach sink" in user_text
        assert "- Memory start: left the room; at doorway; step ongoing" in user_text
        assert "- Rollout actions: ['forward']" in user_text
        assert "memory_end" in call["messages"][0]["content"]
        assert "next_subtask" in call["messages"][0]["content"]
        assert decision.memory == "hall; near door; step ongoing"
        assert decision.done is False
        assert decision.subtask == "continue forward"

    def test_local_watcher_backend_uses_generation_helper(self, monkeypatch):
        backend = LocalWatcherBackend(
            model_path="/tmp/watcher",
            base_model_path="/tmp/base",
            device="cpu",
        )

        monkeypatch.setattr(backend, "_ensure_loaded", lambda: None)
        monkeypatch.setattr(
            backend,
            "_generate_json",
            lambda prompt: {"memory": "local state", "done": True, "subtask": "stop"},
        )

        decision = backend.initialize(
            instruction="go ahead",
            plan_steps=["move forward"],
            first_observation=Image.new("RGB", (8, 8), color=(0, 0, 0)),
            episode_key="scene_1",
        )

        assert decision.memory == "local state"
        assert decision.done is True
        assert decision.subtask == "stop"

    def test_local_watcher_backend_samples_rollout_images(self, monkeypatch):
        backend = LocalWatcherBackend(
            model_path="/tmp/watcher",
            base_model_path="/tmp/base",
            device="cpu",
            image_stride=2,
        )
        captured = {}

        monkeypatch.setattr(backend, "_ensure_loaded", lambda: None)

        def _capture_prompt(prompt):
            captured["pixels"] = [image.getpixel((0, 0))[0] for image in prompt.images]
            return {"memory": "sampled", "done": False, "subtask": "continue"}

        monkeypatch.setattr(backend, "_generate_json", _capture_prompt)

        rollout_slice = [
            {
                "image": Image.new("RGB", (8, 8), color=(idx, 0, 0)),
                "action": "forward",
            }
            for idx in range(5)
        ]
        backend.update(
            instruction="go ahead",
            todo_state=WatcherTodoState(done_steps=[], active_step="move", pending_steps=[]),
            memory_text="state",
            rollout_slice=rollout_slice,
            episode_key="scene_1",
        )

        assert captured["pixels"] == [0, 2, 4]

    def test_actor_prompt_appends_hint_when_present(self):
        wrapper = ThinkVLNActorNavigationModel(
            model=_DummyActor(),
            processor=_DummyProcessor(),
            device="cpu",
        )

        wrapper.predict_action_with_progress_and_done(
            observation=Image.new("RGB", (8, 8), color=(0, 0, 0)),
            instruction="go to the sink",
            subgoal="enter the hall",
            episode_key="scene_1",
            hint="The agent is near the hall entrance.",
        )

        prompt = wrapper.processor.prompts[-1]
        assert "Hint from watcher: The agent is near the hall entrance." in prompt

    def test_episode_runner_records_handoffs_and_trace(self):
        actor = _FakeNavModel(
            [
                (1, 0.4, False),
                (1, 0.9, False),
                (1, 0.3, False),
                (0, 0.4, True),
            ]
        )
        watcher = _FakeWatcher()
        runner = TwoSystemEpisodeRunner(
            nav_model=actor,
            watcher_backend=watcher,
            max_steps_per_wakeup=5,
            progress_threshold=0.85,
            episode_step_cap=20,
            forbidden_actions=[0],
        )

        result = runner.run_episode(
            env=_FakeEnv(),
            episode_key="scene_1",
            instruction="go to the sink",
            plan_steps=["reach the doorway", "approach the sink"],
        )

        assert result["watcher_complete"] is True
        assert result["steps_total"] == 3
        assert result["watcher_wakeups"] == 2
        assert result["done_steps"] == ["reach the doorway", "approach the sink"]
        assert actor.calls[0]["hint"] == "start memory"
        assert actor.calls[2]["hint"] == "hall memory"
        assert len(result["trace"]["steps"]) == 3
        assert result["trace"]["steps"][0]["env_step_index"] == 0
        assert result["trace"]["steps"][-1]["env_step_index"] == 2
        assert len(result["trace"]["watcher_events"]) == 3

    def test_episode_runner_terminates_on_stagnation(self):
        actor = _FakeNavModel([(1, 0.2, False)] * 4)
        runner = TwoSystemEpisodeRunner(
            nav_model=actor,
            watcher_backend=_LoopingWatcher(),
            max_steps_per_wakeup=2,
            progress_threshold=0.95,
            episode_step_cap=10,
            forbidden_actions=[0],
        )

        result = runner.run_episode(
            env=_StagnantEnv(),
            episode_key="scene_stall",
            instruction="go ahead",
            plan_steps=["move forward"],
        )

        assert result["watcher_complete"] is False
        assert result["failed"] is True
        assert result["failure_reason"] == "stagnation"
        assert result["watcher_wakeups"] == 2

    def test_episode_runner_stops_when_actor_call_cap_is_hit(self):
        actor = _FakeNavModel([(1, 0.2, False)] * 5)
        runner = TwoSystemEpisodeRunner(
            nav_model=actor,
            watcher_backend=_LoopingWatcher(),
            max_steps_per_wakeup=10,
            progress_threshold=0.95,
            episode_step_cap=20,
            forbidden_actions=[0],
            max_actor_calls_per_episode=2,
        )

        result = runner.run_episode(
            env=_FakeEnv(),
            episode_key="scene_actor_cap",
            instruction="go ahead",
            plan_steps=["move forward"],
        )

        assert result["watcher_complete"] is False
        assert result["failed"] is True
        assert result["failure_reason"] == "actor_call_cap_exceeded"
        assert result["steps_total"] == 2
        assert len(actor.calls) == 2

    def test_episode_runner_captures_watcher_update_failures(self):
        actor = _FakeNavModel([(1, 0.9, False)])
        runner = TwoSystemEpisodeRunner(
            nav_model=actor,
            watcher_backend=_FailingWatcher(),
            max_steps_per_wakeup=5,
            progress_threshold=0.85,
            episode_step_cap=10,
            forbidden_actions=[0],
        )

        result = runner.run_episode(
            env=_FakeEnv(),
            episode_key="scene_fail",
            instruction="go ahead",
            plan_steps=["move forward"],
        )

        assert result["watcher_complete"] is False
        assert result["failed"] is True
        assert result["failure_reason"] == "watcher_update_error"
        assert "api down" in result["error"]
        assert result["trace"]["watcher_events"][-1]["type"] == "error"

    def test_episode_runner_captures_watcher_initialize_failures(self):
        runner = TwoSystemEpisodeRunner(
            nav_model=_FakeNavModel([(1, 0.9, False)]),
            watcher_backend=_FailingInitWatcher(),
            max_steps_per_wakeup=5,
            progress_threshold=0.85,
            episode_step_cap=10,
            forbidden_actions=[0],
        )

        result = runner.run_episode(
            env=_FakeEnv(),
            episode_key="scene_init_fail",
            instruction="go ahead",
            plan_steps=["move forward"],
        )

        assert result["watcher_complete"] is False
        assert result["failed"] is True
        assert result["failure_reason"] == "watcher_initialize_error"
        assert "boot failed" in result["error"]
        assert result["trace"]["watcher_events"][0]["stage"] == "watcher_initialize"

    def test_episode_runner_stops_cleanly_when_env_episode_is_over(self):
        runner = TwoSystemEpisodeRunner(
            nav_model=_FakeNavModel([(1, 0.2, False)]),
            watcher_backend=_EpisodeOverWatcher(),
            max_steps_per_wakeup=5,
            progress_threshold=0.95,
            episode_step_cap=10,
            forbidden_actions=[],
        )

        result = runner.run_episode(
            env=_EpisodeOverEnv(),
            episode_key="scene_terminal",
            instruction="go ahead",
            plan_steps=["move forward"],
        )

        assert result["failed"] is False
        assert result["watcher_complete"] is False
        assert result["steps_total"] == 1
        assert result["watcher_wakeups"] == 1
        assert result["trace"]["watcher_events"][-1]["wakeup_reason"] == "env_episode_over"

    def test_episode_runner_ignores_actor_done_for_wakeup(self):
        class _WakeupReasonWatcher:
            def initialize(self, instruction, plan_steps, first_observation, episode_key):
                del instruction, plan_steps, first_observation, episode_key
                return WatcherDecision(
                    memory="start memory",
                    done=False,
                    subtask="move forward",
                    raw_response={"memory": "start memory", "done": False, "subtask": "move forward"},
                    wakeup_reason="init",
                )

            def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
                del instruction, todo_state, memory_text, episode_key
                assert len(rollout_slice) == 2
                return WatcherDecision(
                    memory="done",
                    done=True,
                    subtask="stop",
                    raw_response={"memory": "done", "done": True, "subtask": "stop"},
                    wakeup_reason="",
                )

        runner = TwoSystemEpisodeRunner(
            nav_model=_FakeNavModel([(1, 0.2, True), (1, 0.2, False)]),
            watcher_backend=_WakeupReasonWatcher(),
            max_steps_per_wakeup=2,
            progress_threshold=0.95,
            episode_step_cap=10,
            forbidden_actions=[],
        )

        result = runner.run_episode(
            env=_FakeEnv(),
            episode_key="scene_ignore_actor_done",
            instruction="go ahead",
            plan_steps=["move forward"],
        )

        assert result["failed"] is False
        assert result["watcher_wakeups"] == 1
        assert result["trace"]["watcher_events"][-1]["wakeup_reason"] == "max_steps_per_wakeup"

    def test_episode_runner_increments_subtask_id_when_watcher_rewrites_subtask(self):
        class _RewriteSubtaskWatcher:
            def __init__(self):
                self.update_calls = 0

            def initialize(self, instruction, plan_steps, first_observation, episode_key):
                del instruction, plan_steps, first_observation, episode_key
                return WatcherDecision(
                    memory="start memory",
                    done=False,
                    subtask="reach the doorway",
                    raw_response={"memory": "start memory", "done": False, "subtask": "reach the doorway"},
                    wakeup_reason="init",
                )

            def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
                del instruction, todo_state, memory_text, rollout_slice, episode_key
                self.update_calls += 1
                if self.update_calls == 1:
                    return WatcherDecision(
                        memory="new memory",
                        done=False,
                        subtask="cross the open area",
                        raw_response={"memory": "new memory", "done": False, "subtask": "cross the open area"},
                        wakeup_reason="progress_threshold",
                    )
                return WatcherDecision(
                    memory="done",
                    done=True,
                    subtask="stop",
                    raw_response={"memory": "done", "done": True, "subtask": "stop"},
                    wakeup_reason="stop_action",
                )

        actor = _FakeNavModel([(1, 0.9, False), (0, 0.1, False)])
        runner = TwoSystemEpisodeRunner(
            nav_model=actor,
            watcher_backend=_RewriteSubtaskWatcher(),
            max_steps_per_wakeup=5,
            progress_threshold=0.85,
            episode_step_cap=10,
            forbidden_actions=[],
        )

        result = runner.run_episode(
            env=_FakeEnv(),
            episode_key="scene_subtask_rewrite",
            instruction="go ahead",
            plan_steps=["move forward"],
        )

        assert result["failed"] is False
        assert [call["subtask_id"] for call in actor.calls] == [1, 2]

    def test_episode_runner_stops_when_watcher_wakeup_cap_is_hit(self):
        actor = _FakeNavModel([(1, 0.9, False)] * 4)
        runner = TwoSystemEpisodeRunner(
            nav_model=actor,
            watcher_backend=_LoopingWatcher(),
            max_steps_per_wakeup=5,
            progress_threshold=0.85,
            episode_step_cap=10,
            forbidden_actions=[],
            max_watcher_wakeups=2,
        )

        result = runner.run_episode(
            env=_LongFakeEnv(),
            episode_key="scene_wakeup_cap",
            instruction="go ahead",
            plan_steps=["move forward"],
        )

        assert result["failed"] is True
        assert result["failure_reason"] == "watcher_wakeup_cap_exceeded"
        assert result["watcher_wakeups"] == 2
        assert result["trace"]["watcher_events"][-1]["stage"] == "watcher_wakeup_cap"

    def test_episode_runner_does_not_step_env_for_stop_action(self):
        runner = TwoSystemEpisodeRunner(
            nav_model=_FakeNavModel([(0, 0.1, False)]),
            watcher_backend=_StopBeforeStepWatcher(),
            max_steps_per_wakeup=5,
            progress_threshold=0.95,
            episode_step_cap=10,
            forbidden_actions=[],
        )

        result = runner.run_episode(
            env=_NoStepEnv(),
            episode_key="scene_stop",
            instruction="go ahead",
            plan_steps=["stop if needed"],
        )

        assert result["failed"] is False
        assert result["watcher_complete"] is True
        assert result["steps_total"] == 0
        assert result["watcher_wakeups"] == 1
        assert len(result["trace"]["steps"]) == 1
        assert result["trace"]["steps"][0]["action"] == "stop"
        assert result["trace"]["watcher_events"][-1]["wakeup_reason"] == "stop_action"

    def test_episode_runner_logs_actor_and_watcher_events(self, caplog):
        runner = TwoSystemEpisodeRunner(
            nav_model=_FakeNavModel([(0, 0.1, False)]),
            watcher_backend=_StopBeforeStepWatcher(),
            max_steps_per_wakeup=5,
            progress_threshold=0.95,
            episode_step_cap=10,
            forbidden_actions=[],
        )

        with caplog.at_level(logging.INFO, logger=two_system_eval.logger.name):
            runner.run_episode(
                env=_NoStepEnv(),
                episode_key="scene_log",
                instruction="go ahead",
                plan_steps=["stop if needed"],
            )

        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "watcher init" in messages
        assert "actor decision" in messages
        assert "watcher update" in messages

    def test_load_config_missing_required_key_fails_clearly(self, tmp_path: Path):
        config_path = tmp_path / "bad_two_system_eval.yaml"
        _write_config(config_path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        payload["watcher"].pop("model_name")
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="watcher.model_name"):
            load_config(config_path)

    def test_private_helpers_cover_reasoning_messages_and_backend_builders(self):
        assert two_system_eval._build_reasoning_config("none") == {"effort": "none"}
        assert two_system_eval._build_reasoning_config("low") == {"effort": "low", "exclude": True}

        prompt = two_system_eval.WatcherPrompt(
            system_prompt="system",
            user_text="user",
            images=[Image.new("RGB", (4, 4), color=(9, 8, 7))],
        )
        messages = two_system_eval._build_api_messages(prompt)
        assert messages[0]["role"] == "system"
        assert messages[1]["content"][0]["type"] == "text"
        assert messages[1]["content"][1]["type"] == "image_url"

        config = load_config("config/two_system_eval.yaml")
        assert isinstance(two_system_eval._build_watcher_backend(config, device="cpu"), ApiWatcherBackend)
        assert two_system_eval._build_actor_args(config, device="cpu").use_memory is False

        local_config = json.loads(json.dumps(config))
        local_config["watcher"]["backend"] = "local"
        assert isinstance(two_system_eval._build_watcher_backend(local_config, device="cpu"), LocalWatcherBackend)

        local_config["watcher"]["backend"] = "bogus"
        with pytest.raises(ValueError, match="Unsupported watcher backend"):
            two_system_eval._build_watcher_backend(local_config, device="cpu")

    def test_local_watcher_backend_resolves_lora_base_model_path(self, tmp_path: Path):
        adapter_dir = tmp_path / "watcher_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "/tmp/base-model"}),
            encoding="utf-8",
        )

        backend = LocalWatcherBackend(model_path=str(adapter_dir), base_model_path=None, device="cpu")

        assert backend._resolve_base_model_path() == "/tmp/base-model"

    def test_write_debug_html_and_main_are_exercised(self, tmp_path: Path, monkeypatch):
        html_path = tmp_path / "debug" / "ep.html"
        two_system_eval._write_debug_html(
            html_path,
            {
                "episode_key": "scene_1",
                "trace": {
                    "steps": [
                        {
                            "step_index": 0,
                            "env_step_index": 0,
                            "instruction": "go to the sink",
                            "action": "forward",
                            "actor_progress": 0.2,
                            "actor_done": False,
                            "active_plan_step": "move",
                            "watcher_hint": "hint",
                            "watcher_subtask": "reach door",
                            "actor_prompt": "Instruction: go to the sink\nSubgoal: reach door\nHint from watcher: hint",
                        }
                    ],
                    "watcher_events": [
                        {
                            "type": "init",
                            "wakeup_reason": "init",
                            "done": False,
                            "subtask": "reach door",
                            "memory": "find the door",
                            "previous_memory": "",
                            "rollout_step_indices": [],
                        }
                    ],
                },
            },
        )
        html = html_path.read_text(encoding="utf-8")
        assert "scene_1" in html
        assert "forward" in html
        assert "Watcher Events" in html
        assert "reach door" in html
        assert "go to the sink" in html
        assert "Hint from watcher: hint" in html

        sections = two_system_eval._build_debug_video_sections(
            episode_key="scene_1",
            step={
                "step_index": 0,
                "env_step_index": 0,
                "instruction": "go to the sink",
                "action": "forward",
                "actor_progress": 0.2,
                "actor_done": False,
                "active_plan_step": "move",
                "watcher_hint": "hint",
                "watcher_subtask": "reach door",
                "actor_prompt": "Instruction: go to the sink\nSubgoal: reach door\nHint from watcher: hint",
            },
            watcher_event={
                "wakeup_reason": "init",
                "done": False,
                "subtask": "reach door",
                "memory": "find the door",
            },
        )
        flat_lines = "\n".join(f"{title}: {label}: {value}" for title, fields in sections for label, value in fields)
        assert "Actor Input: Instruction: go to the sink" in flat_lines
        assert "Actor Input: Subtask: reach door" in flat_lines
        assert "Actor Input: Hint: hint" in flat_lines
        assert "Actor Input: Prompt: Instruction: go to the sink" in flat_lines

        captured = {}

        class _FakeWriter:
            def __init__(self, path, fourcc, fps, size):
                captured["path"] = path
                captured["fourcc"] = fourcc
                captured["fps"] = fps
                captured["size"] = size
                captured["frames"] = []

            def write(self, frame):
                captured["frames"].append(frame)

            def release(self):
                captured["released"] = True

        fake_cv2 = SimpleNamespace(
            VideoWriter=lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size),
            VideoWriter_fourcc=lambda *args: 1234,
            COLOR_RGB2BGR=1,
            cvtColor=lambda frame, code: frame,
        )
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

        video_path = tmp_path / "debug" / "ep.mp4"
        two_system_eval._write_debug_video(
            video_path,
            {
                "episode_key": "scene_1",
                "trace": {
                    "steps": [
                        {
                            "step_index": 0,
                            "env_step_index": 0,
                            "image": Image.new("RGB", (32, 24), color=(10, 20, 30)),
                            "instruction": "go to the sink",
                            "action": "forward",
                            "actor_progress": 0.2,
                            "actor_done": False,
                            "active_plan_step": "move",
                            "watcher_hint": "hint",
                            "watcher_subtask": "reach door",
                            "actor_prompt": "Instruction: go to the sink\nSubgoal: reach door\nHint from watcher: hint",
                        },
                        {
                            "step_index": 1,
                            "env_step_index": None,
                            "image": Image.new("RGB", (32, 24), color=(30, 20, 10)),
                            "action": "stop",
                            "actor_progress": 0.9,
                            "actor_done": True,
                            "active_plan_step": "stop",
                            "watcher_hint": "hint 2",
                            "watcher_subtask": "stop",
                        },
                    ],
                    "watcher_events": [
                        {
                            "type": "init",
                            "wakeup_reason": "init",
                            "done": False,
                            "subtask": "reach door",
                            "memory": "find the door",
                            "previous_memory": "",
                            "rollout_step_indices": [],
                        },
                        {
                            "type": "update",
                            "wakeup_reason": "progress_threshold",
                            "done": True,
                            "subtask": "stop",
                            "memory": "at goal",
                            "previous_memory": "find the door",
                            "rollout_step_indices": [0],
                            "raw_response": {"done": True, "next_subtask": "stop", "memory_end": "at goal"},
                        },
                    ],
                },
            },
            fps=3,
        )

        assert captured["fps"] == 3
        assert len(captured["frames"]) == 1
        assert captured["size"][0] > 32
        assert captured["size"][1] >= 24
        assert captured["released"] is True

    def test_write_debug_video_pads_variable_height_frames(self, tmp_path: Path, monkeypatch):
        captured = {}

        class _FakeWriter:
            def __init__(self, path, fourcc, fps, size):
                captured["path"] = path
                captured["fourcc"] = fourcc
                captured["fps"] = fps
                captured["size"] = size
                captured["frames"] = []

            def write(self, frame):
                assert frame.shape[1] == captured["size"][0]
                assert frame.shape[0] == captured["size"][1]
                captured["frames"].append(frame.shape[:2])

            def release(self):
                captured["released"] = True

        fake_cv2 = SimpleNamespace(
            VideoWriter=lambda path, fourcc, fps, size: _FakeWriter(path, fourcc, fps, size),
            VideoWriter_fourcc=lambda *args: 1234,
            COLOR_RGB2BGR=1,
            cvtColor=lambda frame, code: frame,
        )
        monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

        video_path = tmp_path / "debug" / "variable_height.mp4"
        two_system_eval._write_debug_video(
            video_path,
            {
                "episode_key": "scene_variable_height",
                "trace": {
                    "steps": [
                        {
                            "step_index": 0,
                            "env_step_index": 0,
                            "image": Image.new("RGB", (32, 24), color=(10, 20, 30)),
                            "instruction": "go to the sink",
                            "action": "forward",
                            "actor_progress": 0.2,
                            "actor_done": False,
                            "active_plan_step": "move",
                            "watcher_hint": "hint",
                            "watcher_subtask": "reach door",
                            "actor_prompt": "short prompt",
                        },
                        {
                            "step_index": 1,
                            "env_step_index": 1,
                            "image": Image.new("RGB", (32, 24), color=(30, 20, 10)),
                            "instruction": "go to the sink",
                            "action": "forward",
                            "actor_progress": 0.9,
                            "actor_done": True,
                            "active_plan_step": "move",
                            "watcher_hint": "long hint " * 20,
                            "watcher_subtask": "move carefully toward the far side of the room",
                            "actor_prompt": "long prompt " * 40,
                        },
                    ],
                    "watcher_events": [],
                },
            },
            fps=3,
        )

        assert captured["fps"] == 3
        assert len(captured["frames"]) == 2
        assert captured["frames"][0] == captured["frames"][1]
        assert captured["released"] is True

        config_path = tmp_path / "two_system_eval.yaml"
        _write_config(config_path)
        called = {}

        def _fake_evaluate(config):
            called["config"] = config
            return {"mode": "eval"}

        def _fake_evaluate_debug(config):
            called["debug_config"] = config
            return {"mode": "debug"}

        monkeypatch.setattr(two_system_eval, "evaluate", _fake_evaluate)
        monkeypatch.setattr(two_system_eval, "evaluate_debug", _fake_evaluate_debug)
        result = two_system_eval.main(["--config", str(config_path)])

        assert result["mode"] == "eval"
        assert called["config"]["watcher"]["backend"] == "api"
        assert "debug_config" not in called

        debug_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        debug_payload["debug"]["enabled"] = True
        debug_payload["debug"]["episode_key"] = "scene_1"
        config_path.write_text(yaml.safe_dump(debug_payload), encoding="utf-8")

        debug_result = two_system_eval.main(["--config", str(config_path)])

        assert debug_result["mode"] == "debug"
        assert called["debug_config"]["debug"]["episode_key"] == "scene_1"

    def test_evaluate_preloads_local_watcher_before_episode_loop(self, tmp_path: Path, monkeypatch):
        _write_config(tmp_path / "two_system_eval.yaml")
        config = load_config(tmp_path / "two_system_eval.yaml")
        config["watcher"]["backend"] = "local"
        config["output"]["output_dir"] = str(tmp_path / "results")

        class _EvalWatcher:
            def __init__(self):
                self.ensure_loaded_calls = 0

            def _ensure_loaded(self):
                self.ensure_loaded_calls += 1

            def initialize(self, instruction, plan_steps, first_observation, episode_key):
                del instruction, plan_steps, first_observation, episode_key
                return WatcherDecision(
                    memory="start",
                    done=False,
                    subtask="move",
                    raw_response={"memory": "start", "done": False, "subtask": "move"},
                    wakeup_reason="init",
                )

            def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
                del instruction, todo_state, memory_text, rollout_slice, episode_key
                return WatcherDecision(
                    memory="done",
                    done=True,
                    subtask="stop",
                    raw_response={"memory": "done", "done": True, "subtask": "stop"},
                    wakeup_reason="stop_action",
                )

        class _EvalNavModel(_FakeNavModel):
            def reset_episode_state(self, episode_key=None):
                del episode_key

        class _EvalEpisode:
            episode_id = "1"
            instruction = SimpleNamespace(instruction_text="go ahead")

        class _EvalEvaluator:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def config_env(self):
                return _FakeEnv()

            def _iter_assigned_episodes(self, env, rank):
                del env, rank
                return [("scene", _EvalEpisode())]

            @staticmethod
            def _episode_instruction(config_path, episode):
                del config_path
                return episode.instruction.instruction_text

        watcher_backend = _EvalWatcher()

        monkeypatch.setattr(two_system_eval, "_build_watcher_backend", lambda config, device: watcher_backend)
        monkeypatch.setattr(two_system_eval, "load_summary_full", lambda path: {"scene_1": {"plan": ["move"]}})
        monkeypatch.setattr("thinkvln.eval.close_eval_dist.init_dist_mode", lambda timeout_minutes=120: (0, 1, 0))
        monkeypatch.setattr("thinkvln.eval.close_eval_dist.all_reduce_scalar_dict", lambda stats, device: stats)
        monkeypatch.setattr("thinkvln.eval.close_eval_models.build_nav_model", lambda args, device, rank, world_size: _EvalNavModel([(0, 0.1, True)]))
        monkeypatch.setattr("thinkvln.eval.close_eval_runner.VLNEvaluator", _EvalEvaluator)

        summary = two_system_eval.evaluate(config)

        assert watcher_backend.ensure_loaded_calls == 1
        assert summary["episodes_evaluated"] == 1

    def test_evaluate_writes_json_safe_trace_payload(self, tmp_path: Path, monkeypatch):
        _write_config(tmp_path / "two_system_eval.yaml")
        config = load_config(tmp_path / "two_system_eval.yaml")
        config["output"]["output_dir"] = str(tmp_path / "results")
        config["output"]["save_trace_jsonl"] = True

        class _EvalEpisode:
            episode_id = "1"
            instruction = SimpleNamespace(instruction_text="go ahead")

        class _EvalEvaluator:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def config_env(self):
                return _FakeEnv()

            def _iter_assigned_episodes(self, env, rank):
                del env, rank
                return [("scene", _EvalEpisode())]

            @staticmethod
            def _episode_instruction(config_path, episode):
                del config_path
                return episode.instruction.instruction_text

        class _EvalWatcher:
            def initialize(self, instruction, plan_steps, first_observation, episode_key):
                del instruction, plan_steps, first_observation, episode_key
                return WatcherDecision(
                    memory="start",
                    done=False,
                    subtask="move",
                    raw_response={"memory": "start", "done": False, "subtask": "move"},
                    wakeup_reason="init",
                )

            def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
                del instruction, todo_state, memory_text, rollout_slice, episode_key
                return WatcherDecision(
                    memory="done",
                    done=True,
                    subtask="stop",
                    raw_response={"memory": "done", "done": True, "subtask": "stop"},
                    wakeup_reason="stop_action",
                )

        class _EvalNavModel(_FakeNavModel):
            def reset_episode_state(self, episode_key=None):
                del episode_key

        monkeypatch.setattr(two_system_eval, "_build_watcher_backend", lambda config, device: _EvalWatcher())
        monkeypatch.setattr(two_system_eval, "load_summary_full", lambda path: {"scene_1": {"plan": ["move"]}})
        monkeypatch.setattr("thinkvln.eval.close_eval_dist.init_dist_mode", lambda timeout_minutes=120: (0, 1, 0))
        monkeypatch.setattr("thinkvln.eval.close_eval_dist.all_reduce_scalar_dict", lambda stats, device: stats)
        monkeypatch.setattr("thinkvln.eval.close_eval_models.build_nav_model", lambda args, device, rank, world_size: _EvalNavModel([(0, 0.1, True)]))
        monkeypatch.setattr("thinkvln.eval.close_eval_runner.VLNEvaluator", _EvalEvaluator)

        def _run_episode(self, env, episode_key, instruction, plan_steps):
            del self, env, episode_key, instruction, plan_steps
            return {
                "episode_key": "scene_1",
                "watcher_complete": True,
                "failed": False,
                "failure_reason": "",
                "error": "",
                "steps_total": 1,
                "watcher_wakeups": 1,
                "done_steps": ["move"],
                "final_memory": "done",
                "trace": {
                    "steps": [
                        {
                            "step_index": 0,
                            "predicted_waypoint": np.array([1.0, 2.0], dtype=np.float32),
                        }
                    ],
                    "watcher_events": [],
                },
                "metrics": {
                    "success": np.float32(1.0),
                    "distance_to_goal": np.float32(0.0),
                    "top_down_map": {
                        "map": np.zeros((2, 2), dtype=np.uint8),
                        "agent_angle": np.array(0.5, dtype=np.float32),
                    },
                },
            }

        monkeypatch.setattr(two_system_eval.TwoSystemEpisodeRunner, "run_episode", _run_episode)

        summary = two_system_eval.evaluate(config)
        trace_path = Path(config["output"]["output_dir"]) / "rank_00_episodes.jsonl"
        payload = json.loads(trace_path.read_text(encoding="utf-8").strip())

        assert summary["episodes_evaluated"] == 1
        assert payload["trace"]["steps"][0]["predicted_waypoint"] == [1.0, 2.0]
        assert payload["metrics"]["success"] == 1.0
        assert "top_down_map" not in payload["metrics"]

    def test_evaluate_writes_debug_video_when_enabled(self, tmp_path: Path, monkeypatch):
        _write_config(tmp_path / "two_system_eval.yaml")
        config = load_config(tmp_path / "two_system_eval.yaml")
        config["output"]["output_dir"] = str(tmp_path / "results")
        config["output"]["save_debug_video"] = True
        config["output"]["debug_video_fps"] = 4

        class _EvalEpisode:
            episode_id = "1"
            instruction = SimpleNamespace(instruction_text="go ahead")

        class _EvalEvaluator:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def config_env(self):
                return _FakeEnv()

            def _iter_assigned_episodes(self, env, rank):
                del env, rank
                return [("scene", _EvalEpisode())]

            @staticmethod
            def _episode_instruction(config_path, episode):
                del config_path
                return episode.instruction.instruction_text

        class _EvalWatcher:
            def initialize(self, instruction, plan_steps, first_observation, episode_key):
                del instruction, plan_steps, first_observation, episode_key
                return WatcherDecision(
                    memory="start",
                    done=False,
                    subtask="move",
                    raw_response={"memory": "start", "done": False, "subtask": "move"},
                    wakeup_reason="init",
                )

            def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
                del instruction, todo_state, memory_text, rollout_slice, episode_key
                return WatcherDecision(
                    memory="done",
                    done=True,
                    subtask="stop",
                    raw_response={"memory": "done", "done": True, "subtask": "stop"},
                    wakeup_reason="progress_threshold",
                )

        class _EvalNavModel(_FakeNavModel):
            def reset_episode_state(self, episode_key=None):
                del episode_key

        captured = {}

        monkeypatch.setattr(two_system_eval, "_build_watcher_backend", lambda config, device: _EvalWatcher())
        monkeypatch.setattr(two_system_eval, "load_summary_full", lambda path: {"scene_1": {"plan": ["move"]}})
        monkeypatch.setattr("thinkvln.eval.close_eval_dist.init_dist_mode", lambda timeout_minutes=120: (0, 1, 0))
        monkeypatch.setattr("thinkvln.eval.close_eval_dist.all_reduce_scalar_dict", lambda stats, device: stats)
        monkeypatch.setattr("thinkvln.eval.close_eval_models.build_nav_model", lambda args, device, rank, world_size: _EvalNavModel([(1, 0.9, False)]))
        monkeypatch.setattr("thinkvln.eval.close_eval_runner.VLNEvaluator", _EvalEvaluator)
        monkeypatch.setattr(
            two_system_eval,
            "_write_debug_video",
            lambda path, episode_result, fps: captured.update(
                {"path": path, "fps": fps, "steps": len(episode_result["trace"]["steps"])}
            ),
        )

        summary = two_system_eval.evaluate(config)

        assert summary["episodes_evaluated"] == 1
        assert captured["fps"] == 4
        assert captured["steps"] == 1
        assert captured["path"].name == "scene_1.mp4"

    def test_evaluate_debug_writes_single_episode_artifacts(self, tmp_path: Path, monkeypatch):
        _write_config(tmp_path / "two_system_eval.yaml")
        config = load_config(tmp_path / "two_system_eval.yaml")
        config["output"]["output_dir"] = str(tmp_path / "results")
        config["debug"]["enabled"] = True
        config["debug"]["episode_key"] = "scene_1"

        class _EvalEpisode:
            episode_id = "1"
            instruction = SimpleNamespace(instruction_text="go ahead")

        captured = {}

        class _EvalEvaluator:
            def __init__(self, *args, **kwargs):
                captured["target_episode_key"] = kwargs["args"].target_episode_key
                captured["output_path"] = kwargs["output_path"]

            def config_env(self):
                return _FakeEnv()

            def _iter_assigned_episodes(self, env, rank):
                del env, rank
                return [("scene", _EvalEpisode())]

            @staticmethod
            def _episode_instruction(config_path, episode):
                del config_path
                return episode.instruction.instruction_text

        class _EvalWatcher:
            def initialize(self, instruction, plan_steps, first_observation, episode_key):
                del instruction, plan_steps, first_observation, episode_key
                return WatcherDecision(
                    memory="start",
                    done=False,
                    subtask="move",
                    raw_response={"memory": "start", "done": False, "subtask": "move"},
                    wakeup_reason="init",
                )

            def update(self, instruction, todo_state, memory_text, rollout_slice, episode_key):
                del instruction, todo_state, memory_text, rollout_slice, episode_key
                return WatcherDecision(
                    memory="done",
                    done=True,
                    subtask="stop",
                    raw_response={"memory": "done", "done": True, "subtask": "stop"},
                    wakeup_reason="progress_threshold",
                )

        class _EvalNavModel(_FakeNavModel):
            def reset_episode_state(self, episode_key=None):
                del episode_key

        def _run_episode(self, env, episode_key, instruction, plan_steps):
            del self, env
            assert episode_key == "scene_1"
            assert instruction == "go ahead"
            assert plan_steps == ["move"]
            return {
                "episode_key": "scene_1",
                "watcher_complete": True,
                "failed": False,
                "failure_reason": "",
                "error": "",
                "steps_total": 1,
                "watcher_wakeups": 1,
                "done_steps": ["move"],
                "final_memory": "done",
                "trace": {
                    "steps": [
                        {
                            "step_index": 0,
                            "env_step_index": 0,
                            "action": "forward",
                            "actor_progress": 0.9,
                            "actor_done": False,
                            "active_plan_step": "move",
                            "watcher_hint": "start",
                            "watcher_subtask": "move",
                        }
                    ],
                    "watcher_events": [
                        {
                            "type": "init",
                            "wakeup_reason": "init",
                            "done": False,
                            "subtask": "move",
                            "memory": "start",
                            "previous_memory": "",
                            "rollout_step_indices": [],
                        }
                    ],
                },
                "metrics": {"success": 1.0},
            }

        monkeypatch.setattr(two_system_eval, "_build_watcher_backend", lambda config, device: _EvalWatcher())
        monkeypatch.setattr(two_system_eval, "load_summary_full", lambda path: {"scene_1": {"plan": ["move"]}})
        monkeypatch.setattr("thinkvln.eval.close_eval_dist.init_dist_mode", lambda timeout_minutes=120: (0, 1, 0))
        monkeypatch.setattr("thinkvln.eval.close_eval_models.build_nav_model", lambda args, device, rank, world_size: _EvalNavModel([(1, 0.9, False)]))
        monkeypatch.setattr("thinkvln.eval.close_eval_runner.VLNEvaluator", _EvalEvaluator)
        monkeypatch.setattr(two_system_eval.TwoSystemEpisodeRunner, "run_episode", _run_episode)
        monkeypatch.setattr(
            two_system_eval,
            "_write_debug_video",
            lambda path, episode_result, fps: path.parent.mkdir(parents=True, exist_ok=True)
            or path.write_text(json.dumps({"fps": fps, "episode_key": episode_result["episode_key"]}), encoding="utf-8"),
        )

        summary = two_system_eval.evaluate_debug(config)
        debug_dir = Path(config["output"]["output_dir"]) / "debug_scene_1"

        assert summary["episode_key"] == "scene_1"
        assert captured["target_episode_key"] == "scene_1"
        assert captured["output_path"] == str(debug_dir)
        assert (debug_dir / "episode.json").is_file()
        assert (debug_dir / "summary.json").is_file()
        assert (debug_dir / "debug" / "scene_1.html").is_file()
        assert (debug_dir / "debug_video" / "scene_1.mp4").is_file()

    def test_evaluate_debug_samples_multiple_episodes_and_writes_videos(self, tmp_path: Path, monkeypatch):
        _write_config(tmp_path / "two_system_eval.yaml")
        config = load_config(tmp_path / "two_system_eval.yaml")
        config["output"]["output_dir"] = str(tmp_path / "results")
        config["debug"]["enabled"] = True
        config["debug"]["episode_key"] = ""
        config["debug"]["sample_limit"] = 2
        config["debug"]["sample_seed"] = 0
        config["debug"]["output_dir"] = str(tmp_path / "results" / "debug_sample_2")
        config["debug"]["video_fps"] = 5

        class _EvalEpisode:
            def __init__(self, episode_id: str, text: str):
                self.episode_id = episode_id
                self.instruction = SimpleNamespace(instruction_text=text)

        captured = {}

        class _EvalEvaluator:
            def __init__(self, *args, **kwargs):
                captured["target_episode_key"] = kwargs["args"].target_episode_key
                captured["sample_rate"] = kwargs["args"].sample_rate
                captured["output_path"] = kwargs["output_path"]

            def config_env(self):
                return _FakeEnv()

            def _iter_assigned_episodes(self, env, rank):
                del env, rank
                return [
                    ("scene", _EvalEpisode("1", "go ahead")),
                    ("scene", _EvalEpisode("2", "turn right")),
                    ("scene", _EvalEpisode("3", "stop there")),
                ]

            @staticmethod
            def _episode_instruction(config_path, episode):
                del config_path
                return episode.instruction.instruction_text

        class _EvalNavModel(_FakeNavModel):
            def reset_episode_state(self, episode_key=None):
                del episode_key

        selected_episode_keys = []

        def _run_episode(self, env, episode_key, instruction, plan_steps):
            del self, env
            outcomes = {
                "scene_2": ("turn right", ["turn"], False, 0.0),
                "scene_3": ("stop there", ["stop"], True, 1.0),
            }
            selected_episode_keys.append(episode_key)
            expected_instruction, expected_plan, watcher_complete, success = outcomes[episode_key]
            assert instruction == expected_instruction
            assert plan_steps == expected_plan
            return {
                "episode_key": episode_key,
                "watcher_complete": watcher_complete,
                "failed": False,
                "failure_reason": "",
                "error": "",
                "steps_total": 1,
                "watcher_wakeups": 1,
                "done_steps": list(expected_plan) if watcher_complete else [],
                "final_memory": "done" if watcher_complete else "keep going",
                "trace": {
                    "steps": [
                        {
                            "step_index": 0,
                            "env_step_index": 0,
                            "action": "forward",
                            "actor_progress": 0.9 if watcher_complete else 0.2,
                            "actor_done": False,
                            "active_plan_step": expected_plan[0],
                            "watcher_hint": "hint",
                            "watcher_subtask": expected_plan[0],
                        }
                    ],
                    "watcher_events": [
                        {
                            "type": "init",
                            "wakeup_reason": "init",
                            "done": False,
                            "subtask": expected_plan[0],
                            "memory": "start",
                            "previous_memory": "",
                            "rollout_step_indices": [],
                        }
                    ],
                },
                "metrics": {"success": success},
            }

        monkeypatch.setattr(two_system_eval, "_build_watcher_backend", lambda config, device: object())
        monkeypatch.setattr(
            two_system_eval,
            "load_summary_full",
            lambda path: {
                "scene_1": {"plan": ["move"]},
                "scene_2": {"plan": ["turn"]},
                "scene_3": {"plan": ["stop"]},
            },
        )
        monkeypatch.setattr("thinkvln.eval.close_eval_dist.init_dist_mode", lambda timeout_minutes=120: (0, 1, 0))
        monkeypatch.setattr("thinkvln.eval.close_eval_models.build_nav_model", lambda args, device, rank, world_size: _EvalNavModel([(1, 0.9, False)]))
        monkeypatch.setattr("thinkvln.eval.close_eval_runner.VLNEvaluator", _EvalEvaluator)
        monkeypatch.setattr(two_system_eval.TwoSystemEpisodeRunner, "run_episode", _run_episode)
        monkeypatch.setattr(
            two_system_eval,
            "_write_debug_video",
            lambda path, episode_result, fps: path.parent.mkdir(parents=True, exist_ok=True)
            or path.write_text(json.dumps({"fps": fps, "episode_key": episode_result["episode_key"]}), encoding="utf-8"),
        )

        summary = two_system_eval.evaluate_debug(config)
        debug_dir = Path(config["debug"]["output_dir"])

        assert summary["episodes_evaluated"] == 2
        assert summary["watcher_complete_rate"] == 0.5
        assert summary["avg_success"] == 0.5
        assert selected_episode_keys == ["scene_2", "scene_3"]
        assert captured["target_episode_key"] == ""
        assert captured["sample_rate"] == 1.0
        assert captured["output_path"] == str(debug_dir)
        assert (debug_dir / "summary.json").is_file()
        assert (debug_dir / "episodes.jsonl").is_file()
        assert (debug_dir / "episodes" / "scene_2.json").is_file()
        assert (debug_dir / "episodes" / "scene_3.json").is_file()
        assert not (debug_dir / "episodes" / "scene_1.json").exists()
        assert (debug_dir / "debug" / "scene_2.html").is_file()
        assert (debug_dir / "debug" / "scene_3.html").is_file()
        assert (debug_dir / "debug_video" / "scene_2.mp4").is_file()
        assert (debug_dir / "debug_video" / "scene_3.mp4").is_file()

    def test_json_safe_handles_zero_dim_ndarray(self):
        payload = two_system_eval._json_safe(
            {
                "metrics": {
                    "distance_to_goal": np.array(0.0, dtype=np.float32),
                }
            }
        )

        assert payload["metrics"]["distance_to_goal"] == 0.0
