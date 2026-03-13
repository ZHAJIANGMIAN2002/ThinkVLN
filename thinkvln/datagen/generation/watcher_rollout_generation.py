from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from random import Random
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from thinkvln.datagen.generation.watcher_utils import (
    action_id_to_name,
    append_jsonl,
    build_pivot_image_relpath,
    build_rollout_image_relpath,
    build_sample_id,
    derive_seed,
    ensure_parent,
    load_jsonl,
    select_pivots,
)
from thinkvln.eval.close_eval_models import build_nav_model
from thinkvln.eval.close_eval_runner import VLNEvaluator
from thinkvln.eval.close_eval_utils import (
    _normalize_subtask_idx,
    build_episode_key,
    build_subtask_spans,
    extract_scene_id,
    normalize_action,
    parse_plan_steps,
)
from habitat.utils.geometry_utils import quaternion_to_list


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate watcher rollout bundles.")
    parser.add_argument("--summary_full_path", type=Path, required=True)
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--bundle_root", type=Path, required=True)
    parser.add_argument("--manifest_file", type=Path, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--num_pivots", type=int, default=4)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--min_rollout_steps", type=int, default=6)
    parser.add_argument("--max_rollout_steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_nav_args(args: argparse.Namespace, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_type="thinkvln_actor",
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        model_max_length=4096,
        memory_num_history_images=6,
        done_threshold=0.85,
        device=device,
        use_memory=False,
    )


def build_eval_args(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        sample_rate=1.0,
        target_episode_key="",
        enable_step_debug=False,
        step_debug_format="none",
        debug_log_interval=0,
    )


def resolve_manifest_file(bundle_root: Path, manifest_file: Optional[Path]) -> Path:
    if manifest_file is not None:
        return manifest_file
    return bundle_root / "manifest" / "watcher_rollout_manifest.jsonl"


def build_episode_lookup(env: Any) -> Dict[str, Any]:
    lookup: Dict[str, Any] = {}
    for episode in env.episodes:
        scene_id = extract_scene_id(episode.scene_id)
        episode_key = build_episode_key(scene_id, episode.episode_id)
        lookup[episode_key] = episode
    return lookup


def load_existing_sample_ids(manifest_file: Path) -> set[str]:
    if not manifest_file.exists():
        return set()
    return {
        str(row["sample_id"])
        for row in load_jsonl(manifest_file)
        if isinstance(row, dict) and isinstance(row.get("sample_id"), str)
    }


def strip_leading_sentinel(actions: List[Any]) -> List[int]:
    if not actions:
        return []
    first = actions[0]
    try:
        if int(first) == -1:
            actions = actions[1:]
    except (TypeError, ValueError):
        pass
    return [normalize_action(action) for action in actions]


def resolve_subtask_text(plan_steps: List[str], subtask_id: int) -> str:
    if not plan_steps:
        return ""
    index = min(max(int(subtask_id) - 1, 0), len(plan_steps) - 1)
    return str(plan_steps[index])


def save_rgb_image(rgb_array: Any, output_path: Path) -> None:
    ensure_parent(output_path)
    Image.fromarray(rgb_array).save(output_path, format="JPEG", quality=95)


def serialize_rotation(rotation: Any) -> np.ndarray:
    if isinstance(rotation, np.ndarray):
        return np.array(rotation, dtype=np.float32, copy=True)
    if isinstance(rotation, (list, tuple)):
        return np.array(rotation, dtype=np.float32)
    return np.array(quaternion_to_list(rotation), dtype=np.float32)


def flush_episode_outputs(
    manifest_file: Path,
    pending_rows: List[Dict[str, Any]],
    pending_images: List[Tuple[Any, Path]],
) -> None:
    if pending_rows:
        ensure_parent(manifest_file)
        with open(manifest_file, "a", encoding="utf-8") as handle:
            for row in pending_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for rgb_array, output_path in pending_images:
        save_rgb_image(rgb_array, output_path)


def collect_episode_cache(
    env: Any,
    episode: Any,
    actions: List[int],
) -> Dict[str, List[Any]]:
    env.current_episode = episode
    observations = env.reset()
    rgb_frames: List[Any] = [np.array(observations["rgb"], copy=True)]
    positions: List[np.ndarray] = [VLNEvaluator._current_position(env)]
    rotations: List[np.ndarray] = [serialize_rotation(env.sim.get_agent_state().rotation)]
    for action in actions:
        if env.episode_over:
            break
        observations = env.step(normalize_action(action))
        rgb_frames.append(np.array(observations["rgb"], copy=True))
        positions.append(VLNEvaluator._current_position(env))
        rotations.append(serialize_rotation(env.sim.get_agent_state().rotation))
    return {
        "rgb_frames": rgb_frames,
        "positions": positions,
        "rotations": rotations,
    }


def restore_episode_frame(
    env: Any,
    episode: Any,
    episode_cache: Dict[str, List[Any]],
    target_frame: int,
) -> Any:
    frame_idx = max(0, min(int(target_frame), len(episode_cache.get("positions", [])) - 1))
    env.current_episode = episode
    env.reset()
    sim = getattr(env, "sim", None)
    positions = episode_cache.get("positions", [])
    rotations = episode_cache.get("rotations", [])
    if (
        sim is not None
        and hasattr(sim, "set_agent_state")
        and hasattr(sim, "get_observations_at")
        and frame_idx < len(positions)
        and frame_idx < len(rotations)
    ):
        position = np.array(positions[frame_idx], dtype=np.float32)
        rotation = np.array(rotations[frame_idx], dtype=np.float32)
        sim.set_agent_state(position, rotation, reset_sensors=False)
        observations = sim.get_observations_at(
            position=position,
            rotation=rotation,
            keep_agent_at_new_pose=True,
        )
        if observations is not None:
            return observations
    return {"rgb": np.array(episode_cache["rgb_frames"][frame_idx], copy=True)}


def nearest_path_distance(point: np.ndarray, path_positions: List[np.ndarray]) -> float:
    if not path_positions:
        return float("inf")
    point_arr = np.array(point, dtype=np.float32)
    return min(
        float(np.linalg.norm(point_arr - np.array(path_point, dtype=np.float32)))
        for path_point in path_positions
    )


def classify_sim_label(
    rollout_positions: List[np.ndarray],
    gt_path_positions: List[np.ndarray],
    goal_pos: np.ndarray,
    goal_reached_dist: float = 0.75,
    off_path_max_dist: float = 1.5,
) -> str:
    if not rollout_positions:
        return "FAIL"
    goal_pos_arr = np.array(goal_pos, dtype=np.float32)
    for position in rollout_positions:
        if float(np.linalg.norm(np.array(position, dtype=np.float32) - goal_pos_arr)) <= float(goal_reached_dist):
            return "PROCEED"
    max_distance = max(
        nearest_path_distance(np.array(position, dtype=np.float32), gt_path_positions)
        for position in rollout_positions
    )
    if max_distance > float(off_path_max_dist):
        return "FAIL"
    return "RESUME"


def find_span_end_frame(spans: List[Tuple[int, int, int]], subtask_id: int, pivot_frame: int) -> int:
    for span_subtask_id, start, end in spans:
        if int(span_subtask_id) == int(subtask_id) and int(start) <= int(pivot_frame) <= int(end):
            return int(end)
    return int(pivot_frame)


def copy_pivot_rgb(
    summary_root: Path,
    meta: Dict[str, Any],
    episode_key: str,
    pivot_frame: int,
    pivot_rgb: Any,
    bundle_root: Path,
) -> str:
    relpath = build_pivot_image_relpath(episode_key, pivot_frame)
    destination = bundle_root / "images" / relpath
    if destination.exists():
        return relpath

    source_rel = meta.get("video", "")
    source_name = Path(source_rel).name if source_rel else ""
    source_path = summary_root / "r2r" / source_name / f"{int(pivot_frame):06d}_rgb.jpg"
    ensure_parent(destination)
    if source_path.exists():
        shutil.copy2(source_path, destination)
    else:
        save_rgb_image(pivot_rgb, destination)
    return relpath


def run_rollout(
    evaluator: VLNEvaluator,
    env: Any,
    episode: Any,
    episode_cache: Dict[str, List[Any]],
    replay_actions: List[int],
    episode_key: str,
    pivot_frame: int,
    instruction: str,
    subtask_id: int,
    subtask_text: str,
    rollout_id: int,
    max_rollout_steps: int,
    seed: int,
) -> Tuple[List[str], List[Any], List[np.ndarray]]:
    observations = restore_episode_frame(
        env=env,
        episode=episode,
        episode_cache=episode_cache,
        target_frame=pivot_frame,
    )
    if evaluator.nav_model is None:
        raise ValueError("navigation model is not initialized")

    action_generator = torch.Generator(device="cpu")
    action_generator.manual_seed(int(seed))
    action_names: List[str] = []
    rollout_rgbs: List[Any] = []
    rollout_positions: List[np.ndarray] = [evaluator._current_position(env)]

    for step_idx in range(max_rollout_steps):
        if env.episode_over:
            break

        info = env.get_metrics()
        model_observation = evaluator.prepare_model_image(observations["rgb"], info)
        action, _, _ = evaluator.nav_model.predict_action_with_progress_and_done(
            observation=model_observation,
            instruction=instruction,
            subgoal=subtask_text,
            episode_key=episode_key,
            subtask_id=subtask_id,
            sample_action=True,
            action_generator=action_generator,
        )

        action_names.append(action_id_to_name(action))
        observations = env.step(int(action))
        rollout_positions.append(evaluator._current_position(env))
        rollout_rgbs.append(np.array(observations["rgb"], copy=True))

        if int(action) == 0 or env.episode_over:
            break

    return action_names, rollout_rgbs, rollout_positions


def generate_bundle(args: argparse.Namespace) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle_root = args.bundle_root.resolve()
    manifest_file = resolve_manifest_file(bundle_root, args.manifest_file)
    summary_root = args.summary_full_path.resolve().parent
    existing_sample_ids = load_existing_sample_ids(manifest_file) if args.resume else set()

    nav_model = build_nav_model(build_nav_args(args, device), device, rank=0, world_size=1)
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split="train",
        env_num=1,
        output_path=str(bundle_root),
        nav_model=nav_model,
        args=build_eval_args(device),
    )
    env = evaluator.config_env()
    episode_lookup = build_episode_lookup(env)
    summary_records = load_jsonl(args.summary_full_path)

    bundle_root.mkdir(parents=True, exist_ok=True)
    processed_samples = 0
    processed_episodes = 0
    total_records = len(summary_records)
    logger.info(
        "watcher rollout generation start: records=%d num_pivots=%d num_rollouts=%d steps=%d..%d",
        total_records,
        args.num_pivots,
        args.num_rollouts,
        args.min_rollout_steps,
        args.max_rollout_steps,
    )

    for meta in summary_records:
        episode_start = time.perf_counter()
        episode_rows: List[Dict[str, Any]] = []
        episode_images: List[Tuple[Any, Path]] = []
        episode_samples = 0
        episode_key = str(meta.get("episode_key", "")).strip()
        if not episode_key:
            continue
        episode = episode_lookup.get(episode_key)
        if episode is None:
            continue

        plan_steps = parse_plan_steps(meta.get("plan", []))
        subtask_sequence = meta.get("subtask_sequence", [])
        actions = meta.get("actions", [])
        if not isinstance(subtask_sequence, list) or not isinstance(actions, list) or not plan_steps:
            continue

        replay_actions = strip_leading_sentinel(actions)
        if not replay_actions:
            continue

        spans = build_subtask_spans(subtask_sequence)
        if not spans:
            continue
        episode_cache = collect_episode_cache(env, episode, replay_actions)
        gt_positions = [np.array(position, dtype=np.float32) for position in episode_cache["positions"]]
        rgb_frames = episode_cache["rgb_frames"]

        episode_rng = Random(derive_seed(args.seed, episode_key, "pivot"))
        pivots = select_pivots(spans, args.num_pivots, episode_rng)
        if not pivots:
            continue

        instruction = str(meta.get("instruction", "") or "")
        logger.info(
            "episode %d/%d key=%s pivots=%d",
            processed_episodes + 1,
            total_records,
            episode_key,
            len(pivots),
        )
        for pivot in pivots:
            pivot_frame = int(pivot["pivot_frame"])
            subtask_id = _normalize_subtask_idx(pivot["subtask_id"])
            subtask_text = resolve_subtask_text(plan_steps, subtask_id)
            end_frame = find_span_end_frame(spans, subtask_id, pivot_frame)
            gt_path_positions = [
                np.array(position, dtype=np.float32)
                for position in gt_positions[pivot_frame:min(end_frame, len(gt_positions) - 1) + 1]
            ]
            goal_pos = np.array(gt_positions[min(end_frame, len(gt_positions) - 1)], dtype=np.float32)

            pivot_relpath = copy_pivot_rgb(
                summary_root=summary_root,
                meta=meta,
                episode_key=episode_key,
                pivot_frame=pivot_frame,
                pivot_rgb=rgb_frames[min(pivot_frame, len(rgb_frames) - 1)],
                bundle_root=bundle_root,
            )

            for rollout_id in range(1, args.num_rollouts + 1):
                sample_id = build_sample_id(episode_key, pivot_frame, rollout_id)
                if sample_id in existing_sample_ids:
                    continue

                row_seed = derive_seed(args.seed, episode_key, pivot_frame, rollout_id)
                rollout_steps = Random(derive_seed(row_seed, "rollout_steps")).randint(
                    min(args.min_rollout_steps, args.max_rollout_steps),
                    max(args.min_rollout_steps, args.max_rollout_steps),
                )
                action_names, rollout_relpaths, rollout_positions = run_rollout(
                    evaluator=evaluator,
                    env=env,
                    episode=episode,
                    episode_cache=episode_cache,
                    replay_actions=replay_actions,
                    episode_key=episode_key,
                    pivot_frame=pivot_frame,
                    instruction=instruction,
                    subtask_id=subtask_id,
                    subtask_text=subtask_text,
                    rollout_id=rollout_id,
                    max_rollout_steps=rollout_steps,
                    seed=row_seed,
                )
                rollout_relpath_list: List[str] = []
                for step_idx, rgb in enumerate(rollout_relpaths):
                    relpath = build_rollout_image_relpath(episode_key, pivot_frame, rollout_id, step_idx)
                    rollout_relpath_list.append(relpath)
                    episode_images.append((rgb, bundle_root / "images" / relpath))
                sim_label = classify_sim_label(
                    rollout_positions=rollout_positions,
                    gt_path_positions=gt_path_positions,
                    goal_pos=goal_pos,
                )
                row = {
                    "sample_id": sample_id,
                    "episode_key": episode_key,
                    "episode_id": int(meta.get("episode_id", meta.get("id", 0))),
                    "scene_id": str(meta.get("scene_id", "")),
                    "pivot_frame": pivot_frame,
                    "rollout_id": int(rollout_id),
                    "subtask_id": int(subtask_id),
                    "instruction": instruction,
                    "subtask_text": subtask_text,
                    "base_image_path": "images",
                    "pivot_image_relpath": pivot_relpath,
                    "rollout_image_relpaths": rollout_relpath_list,
                    "actions": action_names,
                    "sim_label": sim_label,
                    "label_image_stride": 1,
                    "seed": int(row_seed),
                }
                episode_rows.append(row)
                existing_sample_ids.add(sample_id)
                processed_samples += 1
                episode_samples += 1

        flush_episode_outputs(manifest_file, episode_rows, episode_images)
        processed_episodes += 1
        logger.info(
            "episode done key=%s elapsed=%.2fs samples=%d pending_images=%d",
            episode_key,
            time.perf_counter() - episode_start,
            episode_samples,
            len(episode_images),
        )
        if args.max_episodes is not None and processed_episodes >= args.max_episodes:
            break

    env.close()
    return processed_samples


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    processed_samples = generate_bundle(args)
    print(f"watcher rollout samples written: {processed_samples}")


if __name__ == "__main__":
    main()
