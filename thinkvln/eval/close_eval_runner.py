import argparse
import json
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
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations import maps
from habitat_baselines.config.default import get_config as get_habitat_config
from PIL import Image
from tqdm import tqdm

from thinkvln.habitat_extensions import measures  # noqa: F401
from thinkvln.models.navigation_model import (
    NavigationModel,
    StreamVLNNavigationModel,
    ThinkVLNActorNavigationModel,
    ThinkVLNNavigationModel,
)

from thinkvln.eval.close_eval_dist import get_rank
from thinkvln.eval.close_eval_utils import (
    build_episode_key,
    build_subtask_spans,
    compute_step_budget,
    extract_scene_id,
    normalize_action,
    oracle_subtask_at_step,
    parse_plan_steps,
    should_sample_episode,
    timeline_progress,
    write_jsonl_record,
)


logger = logging.getLogger(__name__)


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: Optional[str] = None,
        nav_model: Optional[NavigationModel] = None,
        model: Any = None,
        processor: Any = None,
        epoch: int = 0,
        args: Optional[argparse.Namespace] = None,
    ):
        self.args = args
        requested_device = getattr(args, "device", "cuda") if args else "cuda"
        self.device = torch.device(requested_device)
        self.split = split
        self.env_num = env_num
        self.output_path = output_path or "."
        self.epoch = epoch
        self.config_path = config_path
        self.sample_rate = float(getattr(args, "sample_rate", 1.0)) if args else 1.0
        self.config = get_habitat_config(config_path)
        self.agent_config = get_agent_config(self.config.habitat.simulator)
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors
        self.model_type = getattr(args, "model_type", "thinkvln") if args else "thinkvln"

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

        if nav_model is not None:
            self.nav_model = nav_model
        elif model is not None and processor is not None:
            self.nav_model = ThinkVLNNavigationModel(
                model=model,
                processor=processor,
                device=str(self.device),
                max_new_tokens=getattr(args, "model_max_length", 4096) if args else 1024,
            )
        else:
            self.nav_model = None

    def config_env(self) -> Env:
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

    def prepare_image_with_map(self, rgb: np.ndarray, info: Dict[str, Any]) -> Image.Image:
        rgb_image = rgb.astype(np.uint8)
        if info.get("top_down_map") is not None:
            top_down_map = info["top_down_map"]
            top_down_map_vis = maps.colorize_draw_agent_and_fit_to_height(
                top_down_map, rgb_image.shape[0]
            )
        else:
            top_down_map_vis = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), dtype=np.uint8)
        concatenated = np.concatenate((rgb_image, top_down_map_vis), axis=1)
        return Image.fromarray(concatenated)

    @staticmethod
    def _episode_instruction(config_path: str, episode: Any) -> str:
        if "objectnav" in config_path:
            return episode.object_category
        return episode.instruction.instruction_text

    @staticmethod
    def _build_scene_episode_dict(env: Env) -> Dict[str, List[Any]]:
        scene_episode_dict: Dict[str, List[Any]] = {}
        for episode in env.episodes:
            scene_episode_dict.setdefault(episode.scene_id, []).append(episode)
        return scene_episode_dict

    def _iter_assigned_episodes(self, env: Env, idx: int):
        scene_episode_dict = self._build_scene_episode_dict(env)
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = extract_scene_id(scene)
            for episode in episodes[idx::self.env_num]:
                if should_sample_episode(scene_id, episode.episode_id, self.sample_rate):
                    yield scene_id, episode

    @staticmethod
    def _current_position(env: Env) -> np.ndarray:
        return np.array(env.sim.get_agent_state().position, dtype=np.float32)

    @staticmethod
    def _safe_geodesic_distance(env: Env, current_pos: np.ndarray, goal_pos: np.ndarray) -> float:
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
        env.current_episode = episode
        observations = env.reset()
        replay_steps = min(max(int(target_frame), 0), len(actions))
        for step_idx in range(replay_steps):
            if env.episode_over:
                break
            observations = env.step(normalize_action(actions[step_idx]))
        return observations

    def eval_action(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        env = self.config_env()
        scene_episode_dict = self._build_scene_episode_dict(env)

        sucs, spls, oss, ones = [], [], [], []
        done_res = []

        result_file = os.path.join(self.output_path, "result.json")
        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    try:
                        res = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not all(k in res for k in ["scene_id", "episode_id", "episode_instruction"]):
                        continue
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res.get("success", 0.0))
                        spls.append(res.get("spl", 0.0))
                        oss.append(res.get("os", 0.0))
                        ones.append(res.get("ne", 0.0))

        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = extract_scene_id(scene)
            print(f"scene_id = {scene_id}")

            assigned_episodes = [
                episode
                for episode in episodes[idx::self.env_num]
                if should_sample_episode(scene_id, episode.episode_id, self.sample_rate)
            ]
            process_bar = tqdm(range(len(assigned_episodes)), desc=f"scene {scene_id}")
            for episode in assigned_episodes:
                episode_instruction = self._episode_instruction(self.config_path, episode)
                episode_id = episode.episode_id

                if [scene_id, episode_id, episode_instruction] in done_res:
                    process_bar.update(1)
                    continue

                env.current_episode = episode
                observations = env.reset()

                os.makedirs(os.path.join(self.output_path, f"check_sim_{self.epoch}"), exist_ok=True)
                Image.fromarray(observations["rgb"]).save(
                    os.path.join(self.output_path, f"check_sim_{self.epoch}", f"rgb_{idx}.jpg")
                )

                step_id = 0
                prev_subtask = None

                if self.model_type == "streamvln" and isinstance(self.nav_model, StreamVLNNavigationModel):
                    self.nav_model.rgb_list = []
                    self.nav_model.depth_list = []
                    self.nav_model.pose_list = []
                    self.nav_model.intrinsic_list = []
                    self.nav_model.time_ids = []
                    self.nav_model.action_seq = []
                    self.nav_model.past_key_values = None
                    self.nav_model.output_ids = None
                    self.nav_model.step_count = 0
                    initial_height = env.sim.get_agent_state().position[1]

                while not env.episode_over:
                    if self.nav_model is None:
                        raise ValueError("Navigation model not initialized")

                    self.nav_model.eval()
                    rgb = observations["rgb"]
                    info = env.get_metrics()

                    if self.model_type == "streamvln" and isinstance(self.nav_model, StreamVLNNavigationModel):
                        observation_input = {
                            "rgb": rgb,
                            "depth": observations.get("depth"),
                            "gps": observations.get("gps", [0, 0]),
                            "compass": observations.get("compass", [0]),
                            "env": env,
                            "sensor_config": self.sim_sensors_config,
                            "initial_height": initial_height,
                            "camera_height": self.sim_sensors_config.rgb_sensor.position[1],
                            "min_depth": self.sim_sensors_config.depth_sensor.min_depth,
                            "max_depth": self.sim_sensors_config.depth_sensor.max_depth,
                        }
                    else:
                        observation_input = self.prepare_image_with_map(rgb, info)

                    plan = "1. Navigate to the goal."
                    if hasattr(episode, "reference_path"):
                        plan = "1. Follow the reference path."

                    action, prev_subtask = self.nav_model.predict_action(
                        observation=observation_input,
                        instruction=episode_instruction,
                        plan=plan,
                        prev_subtask=prev_subtask,
                    )

                    observations = env.step(action)
                    step_id += 1

                metrics = env.get_metrics()
                sucs.append(metrics["success"])
                spls.append(metrics["spl"])
                oss.append(metrics["oracle_success"])
                ones.append(metrics["distance_to_goal"])

                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics["oracle_success"],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                }
                with open(result_file, "a", encoding="utf-8") as f:
                    write_jsonl_record(f, result)
                process_bar.update(1)

        env.close()
        return (
            torch.tensor(sucs, device=self.device),
            torch.tensor(spls, device=self.device),
            torch.tensor(oss, device=self.device),
            torch.tensor(ones, device=self.device),
            torch.tensor(len(sucs), device=self.device),
        )

    def eval_subtask_closed_loop(self, idx: int, summary_full: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        if not isinstance(self.nav_model, ThinkVLNActorNavigationModel):
            raise ValueError("Subtask ladder mode currently supports ThinkVLNActorNavigationModel only.")

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

                try:
                    actions = meta.get("actions")
                    subtask_sequence = meta.get("subtask_sequence")
                    plan_steps = parse_plan_steps(meta.get("plan", []))
                    if not isinstance(actions, list) or not isinstance(subtask_sequence, list):
                        raise ValueError("actions/subtask_sequence must be list")
                    if not actions or not subtask_sequence or not plan_steps:
                        raise ValueError("empty actions/subtask_sequence/plan")

                    spans = build_subtask_spans(subtask_sequence)
                    if not spans:
                        raise ValueError("subtask_sequence has no valid spans")

                    gt_positions = self._replay_gt_positions(env, episode, actions)
                    if not gt_positions:
                        raise ValueError("failed to precompute GT positions")

                    stats["episodes_evaluated"] += 1.0
                    logger.info(
                        "[subtask][rank=%d] Episode parsed key=%s spans=%d plan_steps=%d actions=%d",
                        idx,
                        episode_key,
                        len(spans),
                        len(plan_steps),
                        len(actions),
                    )

                    for subtask_idx, start_frame, end_frame in spans:
                        gt_subtask_steps = max(0, end_frame - start_frame + 1)
                        step_budget = compute_step_budget(
                            gt_subtask_steps,
                            float(self.args.subtask_step_budget_factor),
                        )

                        observations = self._replay_to_frame(env, episode, actions, start_frame)
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
                        current_pos = self._current_position(env)
                        final_distance = self._safe_geodesic_distance(env, current_pos, goal_pos)
                        success = final_distance <= float(self.args.subgoal_success_distance)
                        fail_reason = "already_at_subgoal" if success else "step_budget"

                        while (not success) and (not env.episode_over) and rollout_steps < step_budget:
                            info = env.get_metrics()
                            image = self.prepare_image_with_map(observations["rgb"], info)
                            action, pred_progress = self.nav_model.predict_action_with_progress(
                                observation=image,
                                instruction=episode_instruction,
                                subgoal=subgoal_text,
                            )

                            target_progress = timeline_progress(rollout_steps, gt_subtask_steps)
                            stats["progress_abs_error_sum"] += abs(pred_progress - target_progress)
                            stats["progress_count"] += 1.0

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
                except Exception as exc:
                    stats["episodes_malformed"] += 1.0
                    logger.exception("[subtask][rank=%d] malformed episode %s: %s", idx, episode_key, exc)
                    continue

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

    def eval_oracle_switch_closed_loop(
        self,
        idx: int,
        summary_full: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        if not isinstance(self.nav_model, ThinkVLNActorNavigationModel):
            raise ValueError("Oracle-switch ladder mode currently supports ThinkVLNActorNavigationModel only.")

        env = self.config_env()
        stats = {
            "episodes_total": 0.0,
            "episodes_evaluated": 0.0,
            "episodes_missing_meta": 0.0,
            "episodes_malformed": 0.0,
            "sr_sum": 0.0,
            "spl_sum": 0.0,
        }

        detail_path = os.path.join(self.output_path, f"oracle_switch_rank{idx}.jsonl")
        with open(detail_path, "w", encoding="utf-8") as detail_file:
            for scene_id, episode in self._iter_assigned_episodes(env, idx):
                stats["episodes_total"] += 1.0
                episode_key = build_episode_key(scene_id, episode.episode_id)
                episode_instruction = self._episode_instruction(self.config_path, episode)

                meta = summary_full.get(episode_key)
                if meta is None:
                    stats["episodes_missing_meta"] += 1.0
                    print(f"[oracle] missing summary_full for episode_key={episode_key}")
                    continue

                try:
                    subtask_sequence = meta.get("subtask_sequence")
                    plan_steps = parse_plan_steps(meta.get("plan", []))
                    if not isinstance(subtask_sequence, list):
                        raise ValueError("subtask_sequence must be list")
                    if not subtask_sequence or not plan_steps:
                        raise ValueError("empty subtask_sequence/plan")

                    spans = build_subtask_spans(subtask_sequence)
                    if not spans:
                        raise ValueError("subtask_sequence has no valid spans")

                    env.current_episode = episode
                    observations = env.reset()
                    step_idx = 0

                    while not env.episode_over:
                        oracle_subtask = oracle_subtask_at_step(step_idx, spans)
                        plan_idx = min(max(oracle_subtask - 1, 0), len(plan_steps) - 1)
                        subgoal_text = plan_steps[plan_idx]

                        info = env.get_metrics()
                        image = self.prepare_image_with_map(observations["rgb"], info)
                        action, _ = self.nav_model.predict_action_with_progress(
                            observation=image,
                            instruction=episode_instruction,
                            subgoal=subgoal_text,
                        )

                        observations = env.step(action)
                        step_idx += 1

                    metrics = env.get_metrics()
                    sr = float(metrics.get("success", 0.0))
                    spl = float(metrics.get("spl", 0.0))
                    stats["episodes_evaluated"] += 1.0
                    stats["sr_sum"] += sr
                    stats["spl_sum"] += spl

                    detail = {
                        "scene_id": scene_id,
                        "episode_id": episode.episode_id,
                        "episode_key": episode_key,
                        "steps": step_idx,
                        "success": sr,
                        "spl": spl,
                    }
                    write_jsonl_record(detail_file, detail)
                except Exception as exc:
                    stats["episodes_malformed"] += 1.0
                    print(f"[oracle] malformed episode {episode_key}: {exc}")
                    continue

        env.close()
        return stats


__all__ = ["VLNEvaluator"]
