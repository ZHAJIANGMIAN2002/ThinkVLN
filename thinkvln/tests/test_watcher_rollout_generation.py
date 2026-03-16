import logging
from random import Random
from types import SimpleNamespace
import json

import numpy as np
import torch

import thinkvln.datagen.generation.watcher_rollout_generation as watcher_rollout_generation
from thinkvln.datagen.generation.watcher_rollout_generation import (
    classify_sim_label,
    collect_episode_cache,
    flush_episode_outputs,
    generate_bundle,
    load_existing_episode_keys,
    restore_episode_frame,
)
from thinkvln.datagen.generation.watcher_utils import (
    build_pivot_image_relpath,
    build_rollout_image_relpath,
    select_pivots,
)
from thinkvln.models.navigation_model import ThinkVLNActorNavigationModel


class TestWatcherRolloutGeneration:
    def test_select_pivots_uses_fixed_sample_count(self):
        spans = [(1, 0, 4), (2, 5, 27), (3, 28, 36)]

        pivots = select_pivots(spans, num_pivots=4, rng=Random(7))

        assert len(pivots) == 4
        assert len({pivot["pivot_frame"] for pivot in pivots}) == 4
        assert all(0 <= pivot["pivot_frame"] <= 36 for pivot in pivots)
        assert all("pivot_type" not in pivot for pivot in pivots)

    def test_relative_image_paths_match_bundle_layout(self):
        assert build_pivot_image_relpath("scene_001", 12) == "pivot/scene_001/pivot_000012_rgb.jpg"
        assert (
            build_rollout_image_relpath("scene_001", 12, 3, 4)
            == "rollout/scene_001/pivot_000012/rollout_03/000004_rgb.jpg"
        )

    def test_select_action_from_logits_is_seeded(self):
        logits = torch.zeros(4, dtype=torch.float32)

        gen_a = torch.Generator(device="cpu").manual_seed(13)
        gen_b = torch.Generator(device="cpu").manual_seed(13)
        action_a = ThinkVLNActorNavigationModel.select_action_from_logits(
            logits,
            sample_action=True,
            action_generator=gen_a,
        )
        action_b = ThinkVLNActorNavigationModel.select_action_from_logits(
            logits,
            sample_action=True,
            action_generator=gen_b,
        )

        assert action_a == action_b

        sampled_actions = {
            ThinkVLNActorNavigationModel.select_action_from_logits(
                logits,
                sample_action=True,
                action_generator=torch.Generator(device="cpu").manual_seed(seed),
            )
            for seed in range(20)
        }
        assert len(sampled_actions) > 1
        assert (
            ThinkVLNActorNavigationModel.select_action_from_logits(logits, sample_action=False)
            == 0
        )

    def test_classify_sim_label_marks_proceed_when_path_reaches_goal(self):
        rollout_positions = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ]
        gt_path = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ]
        goal_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        label = classify_sim_label(
            rollout_positions=rollout_positions,
            gt_path_positions=gt_path,
            goal_pos=goal_pos,
            goal_reached_dist=0.75,
            off_path_max_dist=1.5,
        )

        assert label == "PROCEED"

    def test_classify_sim_label_marks_fail_when_max_deviation_is_large(self):
        rollout_positions = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([4.0, 0.0, 0.0], dtype=np.float32),
        ]
        gt_path = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ]
        goal_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        label = classify_sim_label(
            rollout_positions=rollout_positions,
            gt_path_positions=gt_path,
            goal_pos=goal_pos,
            goal_reached_dist=0.75,
            off_path_max_dist=1.5,
        )

        assert label == "FAIL"

    def test_classify_sim_label_marks_resume_when_on_path_but_not_at_goal(self):
        rollout_positions = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.0, 0.0], dtype=np.float32),
        ]
        gt_path = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.5, 0.0, 0.0], dtype=np.float32),
            np.array([1.5, 0.0, 0.0], dtype=np.float32),
        ]
        goal_pos = np.array([1.5, 0.0, 0.0], dtype=np.float32)

        label = classify_sim_label(
            rollout_positions=rollout_positions,
            gt_path_positions=gt_path,
            goal_pos=goal_pos,
            goal_reached_dist=0.75,
            off_path_max_dist=1.5,
        )

        assert label == "RESUME"

    def test_collect_episode_cache_reuses_single_gt_replay(self):
        class _ReplayEnv:
            def __init__(self, frames):
                self.frames = frames
                self.current_episode = None
                self.episode_over = False
                self._frame_idx = 0
                self._rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            def reset(self):
                self.episode_over = False
                self._frame_idx = 0
                return {"rgb": self.frames[0]}

            def step(self, action):
                del action
                if self._frame_idx < len(self.frames) - 1:
                    self._frame_idx += 1
                return {"rgb": self.frames[self._frame_idx]}

            @property
            def sim(self):
                env = self

                class _Sim:
                    def get_agent_state(self_inner):
                        return SimpleNamespace(
                            position=np.array([env._frame_idx, 0.0, 0.0], dtype=np.float32),
                            rotation=np.array(env._rotation, dtype=np.float32),
                        )

                return _Sim()

        frames = [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 2, dtype=np.uint8),
        ]
        env = _ReplayEnv(frames)
        episode = SimpleNamespace(episode_id="ep-1")

        cache = collect_episode_cache(env=env, episode=episode, actions=[1, 1])

        assert len(cache["positions"]) == 3
        assert len(cache["rotations"]) == 3
        assert len(cache["rgb_frames"]) == 3
        assert cache["rgb_frames"][2].tolist() == frames[2].tolist()

    def test_restore_episode_frame_uses_cached_state_when_available(self):
        class _RestoreEnv:
            def __init__(self, frames):
                self.frames = frames
                self.current_episode = None
                self.episode_over = False
                self._frame_idx = 0
                self.reset_calls = 0
                self.set_calls = 0

                env = self

                class _Sim:
                    def get_agent_state(self_inner):
                        return SimpleNamespace(
                            position=np.array([env._frame_idx, 0.0, 0.0], dtype=np.float32),
                            rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                        )

                    def set_agent_state(self_inner, position, rotation, reset_sensors=True):
                        del rotation, reset_sensors
                        env.set_calls += 1
                        env._frame_idx = int(position[0])
                        return True

                    def get_observations_at(self_inner, position=None, rotation=None, keep_agent_at_new_pose=False):
                        del rotation, keep_agent_at_new_pose
                        if position is not None:
                            env._frame_idx = int(position[0])
                        return {"rgb": env.frames[env._frame_idx]}

                self.sim = _Sim()

            def reset(self):
                self.reset_calls += 1
                self.episode_over = False
                self._frame_idx = 0
                return {"rgb": self.frames[0]}

        frames = [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.ones((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 2, dtype=np.uint8),
        ]
        env = _RestoreEnv(frames)
        cache = {
            "positions": [
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([2.0, 0.0, 0.0], dtype=np.float32),
            ],
            "rotations": [
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ],
        }

        observations = restore_episode_frame(
            env=env,
            episode=SimpleNamespace(episode_id="ep-1"),
            episode_cache=cache,
            target_frame=2,
        )

        assert env.reset_calls == 1
        assert env.set_calls == 1
        assert observations["rgb"].tolist() == frames[2].tolist()

    def test_flush_episode_outputs_writes_manifest_and_images_in_batch(self, tmp_path):
        manifest_file = tmp_path / "manifest.jsonl"
        bundle_root = tmp_path / "bundle"
        pending_rows = [{"sample_id": "s1", "actions": ["forward"]}]
        pending_images = [
            (
                np.full((4, 4, 3), 7, dtype=np.uint8),
                bundle_root / "images" / "rollout" / "ep" / "000000_rgb.jpg",
            )
        ]

        flush_episode_outputs(
            manifest_file=manifest_file,
            pending_rows=pending_rows,
            pending_images=pending_images,
        )

        with open(manifest_file, "r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        assert rows == pending_rows
        assert pending_images[0][1].exists()

    def test_load_existing_episode_keys_marks_episode_complete_once_any_row_exists(self, tmp_path):
        manifest_file = tmp_path / "manifest.jsonl"
        manifest_file.write_text(
            "\n".join(
                [
                    json.dumps({"sample_id": "s1", "episode_key": "ep-1"}),
                    json.dumps({"sample_id": "s2", "episode_key": "ep-1"}),
                    json.dumps({"sample_id": "s3", "episode_key": "ep-2"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        episode_keys = load_existing_episode_keys(manifest_file)

        assert episode_keys == {"ep-1", "ep-2"}

    def test_generate_bundle_resume_skips_any_episode_already_in_manifest(self, tmp_path, monkeypatch):
        summary_full_path = tmp_path / "summary_full.jsonl"
        summary_full_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "episode_key": "scene_1",
                            "episode_id": 1,
                            "scene_id": "scene",
                            "instruction": "go",
                            "plan": ["step"],
                            "subtask_sequence": [1],
                            "actions": [1],
                        }
                    ),
                    json.dumps(
                        {
                            "episode_key": "scene_2",
                            "episode_id": 2,
                            "scene_id": "scene",
                            "instruction": "go",
                            "plan": ["step"],
                            "subtask_sequence": [1],
                            "actions": [1],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_file = tmp_path / "bundle" / "manifest" / "watcher_rollout_manifest.jsonl"
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps({"sample_id": "s1", "episode_key": "scene_1"}) + "\n", encoding="utf-8")

        episodes = [
            SimpleNamespace(episode_id="1", scene_id="root/scene/glb"),
            SimpleNamespace(episode_id="2", scene_id="root/scene/glb"),
        ]
        rollout_calls = []

        class _Env:
            def __init__(self):
                self.episodes = episodes
                self.current_episode = None

            def close(self):
                return None

        class _Evaluator:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.nav_model = object()

            def config_env(self):
                return _Env()

            def prepare_model_image(self, rgb, info):
                del info
                return rgb

            def _current_position(self, env):
                del env
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        monkeypatch.setattr(watcher_rollout_generation.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(watcher_rollout_generation, "build_nav_model", lambda *args, **kwargs: object())
        monkeypatch.setattr(watcher_rollout_generation, "VLNEvaluator", _Evaluator)
        monkeypatch.setattr(
            watcher_rollout_generation,
            "collect_episode_cache",
            lambda env, episode, actions: {
                "positions": [np.array([0.0, 0.0, 0.0], dtype=np.float32)],
                "rgb_frames": [np.zeros((2, 2, 3), dtype=np.uint8)],
            },
        )
        monkeypatch.setattr(
            watcher_rollout_generation,
            "run_rollout",
            lambda **kwargs: (
                rollout_calls.append(kwargs["episode_key"]) or (["stop"], [], [np.array([0.0, 0.0, 0.0], dtype=np.float32)])
            ),
        )
        monkeypatch.setattr(watcher_rollout_generation, "copy_pivot_rgb", lambda **kwargs: "pivot.jpg")
        monkeypatch.setattr(watcher_rollout_generation, "classify_sim_label", lambda **kwargs: "PROCEED")

        args = SimpleNamespace(
            summary_full_path=summary_full_path,
            habitat_config_path="config/vln_r2r.yaml",
            model_path="model",
            base_model_path=None,
            bundle_root=tmp_path / "bundle",
            manifest_file=None,
            max_episodes=None,
            num_pivots=1,
            num_rollouts=1,
            min_rollout_steps=1,
            max_rollout_steps=1,
            seed=0,
            resume=True,
        )

        processed_samples = generate_bundle(args)

        assert processed_samples == 1
        assert rollout_calls == ["scene_2"]
        rows = [json.loads(line) for line in manifest_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[-1]["plan"] == ["step"]

    def test_generate_bundle_logs_remaining_episode_count_when_resuming(self, tmp_path, monkeypatch, caplog):
        summary_full_path = tmp_path / "summary_full.jsonl"
        summary_full_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "episode_key": "scene_1",
                            "episode_id": 1,
                            "scene_id": "scene",
                            "instruction": "go",
                            "plan": ["step"],
                            "subtask_sequence": [1],
                            "actions": [1],
                        }
                    ),
                    json.dumps(
                        {
                            "episode_key": "scene_2",
                            "episode_id": 2,
                            "scene_id": "scene",
                            "instruction": "go",
                            "plan": ["step"],
                            "subtask_sequence": [1],
                            "actions": [1],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_file = tmp_path / "bundle" / "manifest" / "watcher_rollout_manifest.jsonl"
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps({"sample_id": "s1", "episode_key": "scene_1"}) + "\n", encoding="utf-8")

        episodes = [
            SimpleNamespace(episode_id="1", scene_id="root/scene/glb"),
            SimpleNamespace(episode_id="2", scene_id="root/scene/glb"),
        ]

        class _Env:
            def __init__(self):
                self.episodes = episodes
                self.current_episode = None

            def close(self):
                return None

        class _Evaluator:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                self.nav_model = object()

            def config_env(self):
                return _Env()

            def prepare_model_image(self, rgb, info):
                del info
                return rgb

            def _current_position(self, env):
                del env
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        monkeypatch.setattr(watcher_rollout_generation.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(watcher_rollout_generation, "build_nav_model", lambda *args, **kwargs: object())
        monkeypatch.setattr(watcher_rollout_generation, "VLNEvaluator", _Evaluator)
        monkeypatch.setattr(
            watcher_rollout_generation,
            "collect_episode_cache",
            lambda env, episode, actions: {
                "positions": [np.array([0.0, 0.0, 0.0], dtype=np.float32)],
                "rgb_frames": [np.zeros((2, 2, 3), dtype=np.uint8)],
            },
        )
        monkeypatch.setattr(
            watcher_rollout_generation,
            "run_rollout",
            lambda **kwargs: (["stop"], [], [np.array([0.0, 0.0, 0.0], dtype=np.float32)]),
        )
        monkeypatch.setattr(watcher_rollout_generation, "copy_pivot_rgb", lambda **kwargs: "pivot.jpg")
        monkeypatch.setattr(watcher_rollout_generation, "classify_sim_label", lambda **kwargs: "PROCEED")

        args = SimpleNamespace(
            summary_full_path=summary_full_path,
            habitat_config_path="config/vln_r2r.yaml",
            model_path="model",
            base_model_path=None,
            bundle_root=tmp_path / "bundle",
            manifest_file=None,
            max_episodes=None,
            num_pivots=1,
            num_rollouts=1,
            min_rollout_steps=1,
            max_rollout_steps=1,
            seed=0,
            resume=True,
        )

        with caplog.at_level(logging.INFO):
            generate_bundle(args)

        episode_logs = [record.message for record in caplog.records if record.message.startswith("episode ")]
        assert "episode 1/1 key=scene_2 pivots=1" in episode_logs
