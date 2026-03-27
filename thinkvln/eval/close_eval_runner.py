import argparse
import os
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

# Quiet Habitat-Sim C++ logs by default. Set THINKVLN_QUIET_HABITAT_SIM=0 to disable.
if os.environ.get("THINKVLN_QUIET_HABITAT_SIM", "1") == "1":
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    os.environ.setdefault("GLOG_minloglevel", "2")

import habitat
import numpy as np
import torch
from habitat import Env
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image

from thinkvln.habitat_extensions import measures  # noqa: F401
from thinkvln.models.navigation_model import (
    NavigationModel,
    ThinkVLNActorNavigationModel,
)

from thinkvln.eval.close_eval_debugger import EpisodeStepDebugger
from thinkvln.eval.close_eval_dist import get_rank
from thinkvln.eval.close_eval_utils import (
    _normalize_subtask_idx,
    build_episode_key,
    build_subtask_spans,
    compute_step_budget,
    extract_scene_id,
    normalize_action,
    parse_plan_steps,
    shard_items_round_robin,
    should_sample_episode,
    timeline_progress,
    write_jsonl_record,
)


logger = logging.getLogger(__name__)


def supports_progress_done_actor_model(nav_model: NavigationModel) -> bool:
    return callable(getattr(nav_model, "predict_action_with_progress_and_done", None))


def supports_memory_bank_replay(nav_model: NavigationModel) -> bool:
    if isinstance(nav_model, ThinkVLNActorNavigationModel):
        return True
    return bool(getattr(nav_model, "supports_memory_bank_replay", False))


class VLNEvaluator:
    """Closed-loop evaluator for the memory-model subtask pipeline."""

    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: Optional[str] = None,
        nav_model: Optional[NavigationModel] = None,
        epoch: int = 0,
        args: Optional[argparse.Namespace] = None,
    ):
        """Initialize evaluator configuration and runtime state."""
        self.args = args
        requested_device = getattr(args, "device", "cuda") if args else "cuda"
        self.device = torch.device(requested_device)
        self.split = split
        self.env_num = env_num
        self.output_path = output_path or "."
        self.epoch = epoch
        self.config_path = config_path
        self.sample_rate = float(getattr(args, "sample_rate", 1.0)) if args else 1.0
        self.target_episode_key = str(getattr(args, "target_episode_key", "") or "").strip()
        self.enable_step_debug = bool(getattr(args, "enable_step_debug", False))
        self.step_debug_format = str(getattr(args, "step_debug_format", "none"))
        self.config = get_habitat_config(config_path)

        with habitat.config.read_write(self.config):
            self.config.habitat.dataset.split = self.split
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        self.nav_model = nav_model

    def config_env(self) -> Env:
        """Build one Habitat environment for current split and measurement settings."""
        start = time.time()
        logger.info(
            "[rank=%d] Creating Habitat Env (config=%s split=%s)",
            get_rank(),
            self.config_path,
            self.split,
        )
        env = Env(config=self.config)
        logger.info(
            "[rank=%d] Habitat Env ready: episodes=%d elapsed=%.2fs",
            get_rank(),
            len(env.episodes),
            time.time() - start,
        )
        return env

    def prepare_model_image(self, rgb: np.ndarray, info: Dict[str, Any]) -> Image.Image:
        """Create model input image from RGB only."""
        del info
        return Image.fromarray(rgb.astype(np.uint8))

    @staticmethod
    def _episode_instruction(config_path: str, episode: Any) -> str:
        """Return episode instruction string (ObjectNav uses object category)."""
        if "objectnav" in config_path:
            return episode.object_category
        return episode.instruction.instruction_text

    @staticmethod
    def _build_scene_episode_dict(env: Env) -> Dict[str, List[Any]]:
        """Group episodes by scene id."""
        scene_episode_dict: Dict[str, List[Any]] = {}
        for episode in env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)
        return scene_episode_dict

    def _iter_assigned_episodes(self, env: Env, idx: int):
        """Yield sampled episodes assigned to this rank via round-robin sharding."""
        scene_episode_dict = self._build_scene_episode_dict(env)
        sampled_episodes: List[Tuple[str, Any]] = []
        target_key = self.target_episode_key
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = extract_scene_id(scene)
            for episode in episodes:
                episode_key = build_episode_key(scene_id, episode.episode_id)
                if target_key:
                    if episode_key != target_key:
                        continue
                elif not should_sample_episode(scene_id, episode.episode_id, self.sample_rate):
                    continue
                sampled_episodes.append((scene_id, episode))

        for scene_id, episode in shard_items_round_robin(
            sampled_episodes,
            rank=idx,
            world_size=self.env_num,
        ):
            yield scene_id, episode

    @staticmethod
    def _normalize_actions_for_replay(actions: List[Any], episode_key: str) -> Tuple[List[int], Dict[str, Any]]:
        if not isinstance(actions, list):
            return [], {
                "episode_key": episode_key,
                "leading_sentinel_stripped": False,
                "original_len": 0,
                "normalized_len": 0,
            }

        leading = actions[0] if actions else None
        should_strip = False
        if isinstance(leading, (int, np.integer)):
            should_strip = int(leading) == -1
        elif isinstance(leading, str):
            should_strip = leading.strip() == "-1"

        normalized_raw = actions[1:] if should_strip else actions
        normalized = [normalize_action(action) for action in normalized_raw]
        if should_strip:
            logger.debug(
                "[subtask] replay sentinel stripped for episode_key=%s original_len=%d normalized_len=%d",
                episode_key,
                len(actions),
                len(normalized),
            )
        return normalized, {
            "episode_key": episode_key,
            "leading_sentinel_stripped": bool(should_strip),
            "original_len": int(len(actions)),
            "normalized_len": int(len(normalized)),
        }

    @staticmethod
    def _current_position(env: Env) -> np.ndarray:
        """Read current agent position from simulator state."""
        return np.array(env.sim.get_agent_state().position, dtype=np.float32)

    @staticmethod
    def _safe_geodesic_distance(env: Env, current_pos: np.ndarray, goal_pos: np.ndarray) -> float:
        """Compute robust geodesic distance with Euclidean fallback on failure."""
        try:
            distance = env.sim.geodesic_distance(current_pos.tolist(), goal_pos.tolist())
            if distance is None:
                return float("inf")
            distance = float(distance)
            if np.isnan(distance) or np.isinf(distance):
                return float("inf")
            return distance
        except Exception:
            return float(np.linalg.norm(current_pos - goal_pos))

    def _replay_gt_positions(self, env: Env, episode: Any, actions: List[Any]) -> List[np.ndarray]:
        """Replay full GT action sequence and collect per-step positions."""
        env.current_episode = episode
        env.reset()
        positions = [self._current_position(env)]
        for action in actions:
            if env.episode_over:
                break
            env.step(normalize_action(action))
            positions.append(self._current_position(env))
        return positions

    def _replay_to_frame(
        self,
        env: Env,
        episode: Any,
        actions: List[Any],
        target_frame: int,
    ):
        """Replay to a target frame and return resulting observations."""
        env.current_episode = episode
        observations = env.reset()
        replay_steps = min(max(int(target_frame), 0), len(actions))
        for step_idx in range(replay_steps):
            if env.episode_over:
                break
            observations = env.step(normalize_action(actions[step_idx]))
        return observations

    @staticmethod
    def _subtask_idx_at_frame(subtask_sequence: List[Any], frame_idx: int) -> int:
        if not subtask_sequence:
            return 1
        idx = max(0, min(int(frame_idx), len(subtask_sequence) - 1))
        return _normalize_subtask_idx(subtask_sequence[idx])

    def _replay_to_frame_with_memory(
        self,
        env: Env,
        episode: Any,
        actions: List[Any],
        subtask_sequence: List[Any],
        target_frame: int,
        episode_key: str,
    ):
        """Replay to target frame and prime model memory with frames [0, target_frame)."""
        if not supports_memory_bank_replay(self.nav_model):
            return self._replay_to_frame(env, episode, actions, target_frame)

        self.nav_model.reset_episode_state(episode_key=episode_key)
        env.current_episode = episode
        observations = env.reset()
        replay_steps = min(max(int(target_frame), 0), len(actions))
        for step_idx in range(replay_steps):
            if env.episode_over:
                break
            info = env.get_metrics()
            image = self.prepare_model_image(observations["rgb"], info)
            self.nav_model.record_memory_observation(
                observation=image,
                subtask_id=self._subtask_idx_at_frame(subtask_sequence, step_idx),
                episode_key=episode_key,
            )
            observations = env.step(normalize_action(actions[step_idx]))
        return observations

    def _warmup_subtask_one_with_memory(
        self,
        env: Env,
        episode: Any,
        episode_key: str,
        subtask_id: int,
    ):
        env.current_episode = episode
        observations = env.reset()
        if not supports_memory_bank_replay(self.nav_model) or int(subtask_id) != 1:
            return observations

        self.nav_model.reset_episode_state(episode_key=episode_key)
        warmup_actions = 24
        keep_every = max(1, warmup_actions // 6)
        for step_idx in range(warmup_actions):
            if env.episode_over:
                break
            observations = env.step(3)
            if (step_idx + 1) % keep_every == 0:
                info = env.get_metrics()
                image = self.prepare_model_image(observations["rgb"], info)
                self.nav_model.record_memory_observation(
                    observation=image,
                    subtask_id=1,
                    episode_key=episode_key,
                )
        return observations

    def eval_subtask_closed_loop(self, idx: int, summary_full: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate subtask-by-subtask closed-loop success for actor-capable nav models.

        Workflow per episode:
        1. Read episode metadata from `summary_full` (actions, subtask sequence, plan).
        2. Build contiguous subtask spans and GT subgoal positions via GT action replay.
        3. For each subtask, replay to its start frame and roll out policy actions online.
        4. Mark success when geodesic distance to subgoal position is below threshold.
        5. Accumulate scalar statistics and write detailed JSONL records.

        Returns:
            Scalar stats dict suitable for distributed `all_reduce`.
        """
        if not supports_progress_done_actor_model(self.nav_model):
            raise ValueError(
                "Subtask ladder mode requires a navigation model that implements "
                "predict_action_with_progress_and_done(...)."
            )

        env = self.config_env()
        log_interval = max(0, int(getattr(self.args, "debug_log_interval", 0)))
        stats = {
            "episodes_total": 0.0,
            "episodes_evaluated": 0.0,
            "episodes_missing_meta": 0.0,
            "episodes_malformed": 0.0,
            "subtasks_total": 0.0,
            "subtasks_success": 0.0,
            "steps_success_sum": 0.0,
            "steps_success_count": 0.0,
            "progress_abs_error_sum": 0.0,
            "progress_count": 0.0,
        }

        detail_path = os.path.join(self.output_path, f"subtask_closed_loop_rank{idx}.jsonl")
        logger.info(
            "[subtask][rank=%d] Start closed-loop subtask eval. detail_path=%s episodes=%d",
            idx,
            detail_path,
            len(env.episodes),
        )
        with open(detail_path, "w", encoding="utf-8") as detail_file:
            for scene_id, episode in self._iter_assigned_episodes(env, idx):
                stats["episodes_total"] += 1.0
                episode_key = build_episode_key(scene_id, episode.episode_id)
                episode_instruction = self._episode_instruction(self.config_path, episode)
                logger.info(
                    "[subtask][rank=%d] Episode start key=%s scene=%s episode_id=%s",
                    idx,
                    episode_key,
                    scene_id,
                    episode.episode_id,
                )

                meta = summary_full.get(episode_key)
                if meta is None:
                    stats["episodes_missing_meta"] += 1.0
                    logger.warning("[subtask][rank=%d] missing summary_full for episode_key=%s", idx, episode_key)
                    continue

                debugger: Optional[EpisodeStepDebugger] = None
                debugger_finalized = False
                try:
                    actions = meta.get("actions")
                    subtask_sequence = meta.get("subtask_sequence")
                    plan_steps = parse_plan_steps(meta.get("plan", []))
                    if not isinstance(actions, list) or not isinstance(subtask_sequence, list):
                        raise ValueError("actions/subtask_sequence must be list")
                    if not actions or not subtask_sequence or not plan_steps:
                        raise ValueError("empty actions/subtask_sequence/plan")
                    replay_actions, replay_meta = self._normalize_actions_for_replay(actions, episode_key)
                    if not replay_actions:
                        raise ValueError("empty replay actions after normalization")

                    spans = build_subtask_spans(subtask_sequence)
                    if not spans:
                        raise ValueError("subtask_sequence has no valid spans")

                    gt_positions = self._replay_gt_positions(env, episode, replay_actions)
                    if not gt_positions:
                        raise ValueError("failed to precompute GT positions")

                    stats["episodes_evaluated"] += 1.0
                    self.nav_model.reset_episode_state(episode_key=episode_key)
                    logger.info(
                        "[subtask][rank=%d] Episode parsed key=%s spans=%d plan_steps=%d actions=%d",
                        idx,
                        episode_key,
                        len(spans),
                        len(plan_steps),
                        len(replay_actions),
                    )
                    debug_episode = self.enable_step_debug and (
                        not self.target_episode_key or self.target_episode_key == episode_key
                    )
                    debugger = EpisodeStepDebugger(
                        output_path=self.output_path,
                        episode_key=episode_key,
                        enable_step_debug=debug_episode,
                        step_debug_format=self.step_debug_format,
                    )

                    for subtask_idx, start_frame, end_frame in spans:
                        gt_subtask_steps = max(0, end_frame - start_frame + 1)
                        step_budget = compute_step_budget(
                            gt_subtask_steps,
                            float(self.args.subtask_step_budget_factor),
                        )

                        if int(subtask_idx) == 1:
                            observations = self._warmup_subtask_one_with_memory(
                                env=env,
                                episode=episode,
                                episode_key=episode_key,
                                subtask_id=subtask_idx,
                            )
                        else:
                            observations = self._replay_to_frame_with_memory(
                                env=env,
                                episode=episode,
                                actions=replay_actions,
                                subtask_sequence=subtask_sequence,
                                target_frame=start_frame,
                                episode_key=episode_key,
                            )
                        replay_memory_count = len(getattr(self.nav_model, "memory_bank_images", []))
                        goal_pos = gt_positions[min(end_frame, len(gt_positions) - 1)]
                        plan_idx = min(max(subtask_idx - 1, 0), len(plan_steps) - 1)
                        subgoal_text = plan_steps[plan_idx]
                        logger.info(
                            "[subtask][rank=%d] key=%s subtask=%d frame=[%d,%d] gt_steps=%d budget=%d subgoal=%s",
                            idx,
                            episode_key,
                            subtask_idx,
                            start_frame,
                            end_frame,
                            gt_subtask_steps,
                            step_budget,
                            subgoal_text,
                        )

                        rollout_steps = 0
                        previous_pred_progress: Optional[float] = None
                        current_pos = self._current_position(env)
                        final_distance = self._safe_geodesic_distance(env, current_pos, goal_pos)
                        success = final_distance <= float(self.args.subgoal_success_distance)
                        fail_reason = "already_at_subgoal" if success else "step_budget"

                        while (not success) and (not env.episode_over) and rollout_steps < step_budget:
                            info = env.get_metrics()
                            image = self.prepare_model_image(observations["rgb"], info)
                            action, pred_progress, _ = self.nav_model.predict_action_with_progress_and_done(
                                observation=image,
                                instruction=episode_instruction,
                                subgoal=subgoal_text,
                                episode_key=episode_key,
                                subtask_id=subtask_idx,
                                forbidden_actions=[0],
                            )
                            snapshot = self.nav_model.get_last_debug_snapshot()

                            target_progress = timeline_progress(rollout_steps, gt_subtask_steps)
                            stats["progress_abs_error_sum"] += abs(pred_progress - target_progress)
                            stats["progress_count"] += 1.0
                            prev_progress_in = None
                            if snapshot is not None:
                                prev_progress_in = snapshot.get("prev_progress_input")

                            if previous_pred_progress is None:
                                expected_prev_progress = 0.0
                            else:
                                expected_prev_progress = float(previous_pred_progress)
                            if prev_progress_in is None:
                                progress_pass_through_delta = None
                                progress_pass_through_ok = False
                            else:
                                progress_pass_through_delta = abs(float(prev_progress_in) - expected_prev_progress)
                                progress_pass_through_ok = progress_pass_through_delta <= 1e-5
                            previous_pred_progress = float(pred_progress)

                            expected_memory_bank_size = replay_memory_count + rollout_steps + 1
                            memory_bank_size = int(
                                snapshot.get("memory_bank_size", len(getattr(self.nav_model, "memory_bank_images", [])))
                                if snapshot is not None
                                else len(getattr(self.nav_model, "memory_bank_images", []))
                            )
                            memory_frame_count_ok = memory_bank_size == expected_memory_bank_size

                            if snapshot is not None:
                                memory_bank_subtasks = snapshot.get("memory_bank_subtasks", [])
                            else:
                                memory_bank_subtasks = list(getattr(self.nav_model, "memory_bank_subtasks", []))
                            if subtask_idx > 1:
                                memory_includes_past_subtask = any(
                                    int(bank_subtask) < int(subtask_idx)
                                    for bank_subtask in memory_bank_subtasks
                                )
                            else:
                                memory_includes_past_subtask = True

                            if debugger is not None:
                                debug_images = snapshot.get("selected_images", [image]) if snapshot is not None else [image]
                                debug_row = {
                                    "scene_id": scene_id,
                                    "episode_id": episode.episode_id,
                                    "episode_key": episode_key,
                                    "subtask_idx": int(subtask_idx),
                                    "rollout_step": int(rollout_steps),
                                    "frame_start": int(start_frame),
                                    "frame_end": int(end_frame),
                                    "prompt": snapshot.get("prompt", "") if snapshot is not None else "",
                                    "query_token_ids": snapshot.get("query_token_ids", []) if snapshot is not None else [],
                                    "selected_indices": snapshot.get("selected_indices", []) if snapshot is not None else [],
                                    "selected_subtask_ids": snapshot.get("selected_subtask_ids", []) if snapshot is not None else [],
                                    "prev_progress_in": prev_progress_in,
                                    "pred_progress": float(pred_progress),
                                    "target_progress": float(target_progress),
                                    "progress_pass_through_delta": progress_pass_through_delta,
                                    "memory_bank_size": int(memory_bank_size),
                                    "expected_memory_bank_size": int(expected_memory_bank_size),
                                    "checks": {
                                        "memory_frame_count_ok": bool(memory_frame_count_ok),
                                        "memory_includes_past_subtask": bool(memory_includes_past_subtask),
                                        "progress_pass_through_ok": bool(progress_pass_through_ok),
                                    },
                                    "replay_leading_sentinel_stripped": bool(
                                        replay_meta["leading_sentinel_stripped"]
                                    ),
                                }
                                debugger.record_step(debug_row, images=debug_images)

                            observations = env.step(action)
                            rollout_steps += 1

                            current_pos = self._current_position(env)
                            final_distance = self._safe_geodesic_distance(env, current_pos, goal_pos)
                            if log_interval > 0 and (rollout_steps % log_interval == 0):
                                logger.info(
                                    "[subtask][rank=%d] key=%s subtask=%d rollout=%d/%d distance=%.3f pred_progress=%.3f target_progress=%.3f",
                                    idx,
                                    episode_key,
                                    subtask_idx,
                                    rollout_steps,
                                    step_budget,
                                    final_distance,
                                    pred_progress,
                                    target_progress,
                                )
                            if final_distance <= float(self.args.subgoal_success_distance):
                                success = True
                                fail_reason = ""
                                break

                        if not success and env.episode_over:
                            fail_reason = "episode_over"

                        stats["subtasks_total"] += 1.0
                        if success:
                            stats["subtasks_success"] += 1.0
                            stats["steps_success_sum"] += float(rollout_steps)
                            stats["steps_success_count"] += 1.0

                        detail = {
                            "scene_id": scene_id,
                            "episode_id": episode.episode_id,
                            "episode_key": episode_key,
                            "subtask_idx": subtask_idx,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "gt_subtask_steps": gt_subtask_steps,
                            "step_budget": step_budget,
                            "steps_to_subgoal": rollout_steps,
                            "success": bool(success),
                            "final_distance": float(final_distance),
                            "subgoal_text": subgoal_text,
                            "fail_reason": fail_reason,
                            "replay_leading_sentinel_stripped": bool(
                                replay_meta["leading_sentinel_stripped"]
                            ),
                            "replay_actions_original_len": int(replay_meta["original_len"]),
                            "replay_actions_normalized_len": int(replay_meta["normalized_len"]),
                        }
                        write_jsonl_record(detail_file, detail)
                        logger.info(
                            "[subtask][rank=%d] key=%s subtask=%d done success=%s steps=%d final_distance=%.3f fail_reason=%s",
                            idx,
                            episode_key,
                            subtask_idx,
                            bool(success),
                            rollout_steps,
                            float(final_distance),
                            fail_reason,
                        )
                    if debugger is not None:
                        debug_summary = debugger.finalize()
                        debugger_finalized = True
                        logger.info(
                            "[subtask][rank=%d] debug finalized episode=%s steps=%d",
                            idx,
                            episode_key,
                            int(debug_summary.get("steps_total", 0)),
                        )
                except Exception as exc:
                    stats["episodes_malformed"] += 1.0
                    logger.exception("[subtask][rank=%d] malformed episode %s: %s", idx, episode_key, exc)
                    continue
                finally:
                    if debugger is not None and not debugger_finalized:
                        debug_summary = debugger.finalize()
                        logger.info(
                            "[subtask][rank=%d] debug finalized (finally) episode=%s steps=%d",
                            idx,
                            episode_key,
                            int(debug_summary.get("steps_total", 0)),
                        )

        if self.target_episode_key and int(stats["episodes_total"]) == 0:
            logger.warning(
                "[subtask][rank=%d] target_episode_key=%s not found in assigned split=%s",
                idx,
                self.target_episode_key,
                self.split,
            )
        env.close()
        logger.info(
            "[subtask][rank=%d] Finished. episodes_total=%d evaluated=%d missing_meta=%d malformed=%d subtasks=%d success=%d",
            idx,
            int(stats["episodes_total"]),
            int(stats["episodes_evaluated"]),
            int(stats["episodes_missing_meta"]),
            int(stats["episodes_malformed"]),
            int(stats["subtasks_total"]),
            int(stats["subtasks_success"]),
        )
        return stats

__all__ = ["VLNEvaluator", "supports_progress_done_actor_model", "supports_memory_bank_replay"]
