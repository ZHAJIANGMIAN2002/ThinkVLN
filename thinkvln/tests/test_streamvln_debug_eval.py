import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import thinkvln.eval.two_system_eval as two_system_eval
import thinkvln.eval.streamvln_debug_eval as streamvln_debug_eval


class _FakeInstruction:
    def __init__(self, text: str):
        self.instruction_text = text


class _FakeEpisode:
    def __init__(self, scene_path: str, episode_id: str, instruction: str):
        self.scene_id = scene_path
        self.episode_id = episode_id
        self.instruction = _FakeInstruction(instruction)


class _FakeEnv:
    def close(self):
        return None


class _FakeEvaluator:
    def __init__(self, items):
        self._items = list(items)

    def config_env(self):
        return _FakeEnv()

    def _iter_assigned_episodes(self, env, rank):
        del env, rank
        for item in self._items:
            yield item

    @staticmethod
    def _episode_instruction(config_path, episode):
        del config_path
        return episode.instruction.instruction_text


def _fake_episode_result(episode_key: str):
    return {
        "episode_key": episode_key,
        "nav_success": True,
        "failed": False,
        "failure_reason": "",
        "error": "",
        "steps_total": 3,
        "metrics": {
            "success": 1.0,
            "spl": 0.5,
            "oracle_success": 1.0,
            "distance_to_goal": 1.25,
        },
        "trace": {
            "steps": [
                {
                    "step_index": 0,
                    "env_step_index": 0,
                    "image": Image.new("RGB", (32, 24), color=(10, 20, 30)),
                    "action": "forward",
                    "actor_progress": 0.0,
                    "actor_done": False,
                    "actor_prompt": "Instruction: go forward",
                }
            ],
            "watcher_events": [],
        },
        "debug_top_down_map": None,
        "debug_reference_path_map_coords": [],
    }


class TestStreamVLNDebugEval:
    def test_build_debug_prompt_only_keeps_instruction_as_model_input(self):
        prompt = streamvln_debug_eval._build_debug_prompt(
            {
                "instruction": "walk to the kitchen",
                "step_id": 0,
                "generated_this_step": True,
                "used_cached_output": False,
                "time_ids": [0],
                "selected_frame_count": 1,
                "history_frame_count": 0,
                "model_prompt": "You are an autonomous navigation assistant.",
                "raw_output": "↑ ↑ STOP",
                "generated_actions": [1, 1, 0],
                "action_buffer_before": [1, 1, 0],
                "returned_action": 1,
                "action_buffer_after": [1, 0],
            }
        )

        assert "Instruction: walk to the kitchen" in prompt
        assert "Model Input:" in prompt
        assert "time_ids" not in prompt.lower()
        assert "Input Frames:" not in prompt
        assert "Action Buffer Before:" not in prompt

    def test_select_debug_candidates_matches_two_system_sampling(self):
        items = [
            ("scene_a", _FakeEpisode("/data/scene_a/scene.glb", "1", "instruction a")),
            ("scene_b", _FakeEpisode("/data/scene_b/scene.glb", "2", "instruction b")),
            ("scene_c", _FakeEpisode("/data/scene_c/scene.glb", "3", "instruction c")),
        ]
        evaluator = _FakeEvaluator(items)
        env = evaluator.config_env()
        summary_full = {
            "scene_a_1": {"plan": ["a"]},
            "scene_b_2": {"plan": ["b"]},
            "scene_c_3": {"plan": ["c"]},
        }
        config = {
            "env": {"habitat_config_path": "config/vln_r2r.yaml"},
            "debug": {"sample_limit": 2, "sample_seed": 7},
        }

        selected = streamvln_debug_eval._collect_debug_candidates(
            evaluator=evaluator,
            env=env,
            rank=0,
            summary_full=summary_full,
        )
        actual_keys = [
            episode_key
            for episode_key, _, _ in streamvln_debug_eval._select_debug_candidates(
                candidates=selected,
                config=config,
            )
        ]
        expected_keys = [
            episode_key
            for episode_key, _, _ in two_system_eval._sample_debug_candidates(
                selected,
                sample_limit=2,
                sample_seed=7,
            )
        ]

        assert actual_keys == expected_keys

    def test_evaluate_debug_writes_batch_summary_and_episode_manifest(self, tmp_path: Path, monkeypatch):
        items = [
            ("scene_a", _FakeEpisode("/data/scene_a/scene.glb", "1", "instruction a")),
            ("scene_b", _FakeEpisode("/data/scene_b/scene.glb", "2", "instruction b")),
            ("scene_c", _FakeEpisode("/data/scene_c/scene.glb", "3", "instruction c")),
        ]
        evaluator = _FakeEvaluator(items)
        config = {
            "env": {
                "habitat_config_path": "config/vln_r2r.yaml",
                "eval_split": "train",
                "summary_full_path": "/tmp/summary_full.jsonl",
                "sample_rate": 1.0,
            },
            "debug": {
                "sample_limit": 2,
                "sample_seed": 3,
            },
        }
        summary_full = {
            "scene_a_1": {"plan": ["a"]},
            "scene_b_2": {"plan": ["b"]},
            "scene_c_3": {"plan": ["c"]},
        }
        written = []

        monkeypatch.setattr(streamvln_debug_eval, "_load_sample_config", lambda path: config)
        monkeypatch.setattr(streamvln_debug_eval, "load_summary_full", lambda path: summary_full)
        monkeypatch.setattr(streamvln_debug_eval, "_build_env_evaluator", lambda args, sample_config: evaluator)
        monkeypatch.setattr(streamvln_debug_eval, "_load_streamvln_policy", lambda args, device, rank, world_size: object())
        monkeypatch.setattr(
            streamvln_debug_eval,
            "run_streamvln_episode",
            lambda **kwargs: _fake_episode_result(kwargs["episode_key"]),
        )

        def _fake_write_artifacts(output_dir, episode_result, episode_payload, summary_payload, fps, single_episode):
            written.append(
                {
                    "episode_key": episode_result["episode_key"],
                    "fps": fps,
                    "single_episode": single_episode,
                    "summary_payload": dict(summary_payload),
                }
            )

        monkeypatch.setattr(streamvln_debug_eval, "_write_debug_episode_artifacts", _fake_write_artifacts)

        args = SimpleNamespace(
            sample_config_path="config/two_system_eval.local_debug20.yaml",
            model_path="/mnt/swx/ThinkVLN/model_weights/streamvln",
            output_path=str(tmp_path / "streamvln_eval"),
            device="cpu",
            debug_video_fps=4,
            model_max_length=4096,
            num_frames=32,
            num_future_steps=4,
            num_history=8,
        )

        summary = streamvln_debug_eval.evaluate_debug(args)

        assert summary["episodes_total"] == 2
        assert summary["success_rate"] == 1.0
        assert len(written) == 2
        assert all(item["fps"] == 4 for item in written)
        assert all(item["single_episode"] is False for item in written)

        episodes_path = tmp_path / "streamvln_eval" / "episodes.jsonl"
        summary_path = tmp_path / "streamvln_eval" / "summary.json"
        result_path = tmp_path / "streamvln_eval" / "result.json"

        assert episodes_path.is_file()
        assert summary_path.is_file()
        assert result_path.is_file()

        rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 2
        assert {row["episode_key"] for row in rows} == {item["episode_key"] for item in written}

        result_rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(result_rows) == 3
        assert result_rows[-1]["length"] == 2
