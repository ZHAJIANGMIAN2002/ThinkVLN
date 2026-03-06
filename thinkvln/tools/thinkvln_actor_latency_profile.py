#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from habitat import Env
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat_baselines.config.default import get_config as get_habitat_config

from thinkvln.eval.close_eval_models import load_thinkvln_actor_model
from thinkvln.habitat_extensions import measures as _measures  # noqa: F401


PROMPT_TEMPLATE = (
    "Instruction: {instruction}\n"
    "Current subgoal: {subgoal}\n"
    "Predict the next 4 actions and subgoal progress."
)


def resolve_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Refusing implicit CPU fallback because loading ThinkVLNActor "
            "on CPU is extremely slow and may look stuck. Enable GPU (check CUDA_VISIBLE_DEVICES), "
            "or pass --cpu explicitly."
        )
    return torch.device("cuda")


def log_phase(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def get_instruction_text(config_path: str, episode: Any) -> str:
    if "objectnav" in config_path and hasattr(episode, "object_category"):
        return str(episode.object_category)
    if hasattr(episode, "instruction") and hasattr(episode.instruction, "instruction_text"):
        return episode.instruction.instruction_text
    if hasattr(episode, "object_category"):
        return str(episode.object_category)
    return ""


def next_shortest_path_action(
    env: Any,
    follower: Any,
    ref_path: List[Any],
    waypoint_id: int,
) -> Tuple[Optional[int], int, Any]:
    if waypoint_id >= len(ref_path):
        return None, waypoint_id, follower

    action = follower.get_next_action(ref_path[waypoint_id])
    while action == 0:
        waypoint_id += 1
        if waypoint_id == len(ref_path) - 1:
            follower = ShortestPathFollower(sim=env.sim, goal_radius=0.25, return_one_hot=False)
        if waypoint_id >= len(ref_path):
            return None, waypoint_id, follower
        action = follower.get_next_action(ref_path[waypoint_id])
    return int(action), waypoint_id, follower


def prepare_image_with_map(rgb: np.ndarray, info: Dict[str, Any]) -> Image.Image:
    rgb_image = rgb.astype(np.uint8)
    if info.get("top_down_map") is not None:
        top_down_map_vis = maps.colorize_draw_agent_and_fit_to_height(
            info["top_down_map"], rgb_image.shape[0]
        )
    else:
        top_down_map_vis = np.zeros_like(rgb_image)
    return Image.fromarray(np.concatenate((rgb_image, top_down_map_vis), axis=1))


def build_query_tokens(model: Any, device: torch.device) -> Tuple[torch.Tensor, int]:
    num_query_tokens = int(getattr(model.config, "num_query_tokens", 4))
    action_query_token_id = int(getattr(model.config, "action_query_token_id", 151700))
    progress_query_token_id = int(getattr(model.config, "progress_query_token_id", 151701))

    tokens: List[int] = []
    for _ in range(num_query_tokens):
        tokens.append(action_query_token_id)
        tokens.append(progress_query_token_id)
    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0), num_query_tokens


def build_actor_inputs(
    model: Any,
    processor: Any,
    observation: Image.Image,
    prompt: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": observation},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[observation],
        return_tensors="pt",
        padding=False,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    query_tokens, num_query_tokens = build_query_tokens(model, device)
    input_ids = torch.cat([input_ids, query_tokens], dim=1)
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((1, query_tokens.shape[1]), dtype=torch.long, device=device),
        ],
        dim=1,
    )

    model_inputs: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "action_labels": torch.zeros((1, num_query_tokens), dtype=torch.long, device=device),
        "progress_labels": torch.zeros((1,), dtype=torch.float32, device=device),
        "done_labels": torch.zeros((1,), dtype=torch.float32, device=device),
    }
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        model_inputs["pixel_values"] = inputs["pixel_values"].to(device)
    if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
        model_inputs["image_grid_thw"] = inputs["image_grid_thw"].to(device)
    return model_inputs


def get_output_tensor(outputs: Any, key: str) -> Optional[torch.Tensor]:
    if isinstance(outputs, dict):
        return outputs.get(key)
    if hasattr(outputs, "get"):
        try:
            return outputs.get(key)
        except Exception:
            pass
    return getattr(outputs, key, None)


def predict_action_with_timing(
    model: Any,
    processor: Any,
    instruction: str,
    subgoal: str,
    observation: Image.Image,
    device: torch.device,
) -> Tuple[int, float, float]:
    prompt = PROMPT_TEMPLATE.format(instruction=instruction, subgoal=subgoal)
    model_inputs = build_actor_inputs(
        model=model,
        processor=processor,
        observation=observation,
        prompt=prompt,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**model_inputs, return_dict=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    action_logits = get_output_tensor(outputs, "action_logits")
    progress_preds = get_output_tensor(outputs, "progress_preds")

    action = 0
    if action_logits is not None and action_logits.shape[1] > 0:
        action = int(torch.argmax(action_logits[0, 0], dim=-1).item())

    progress = 0.0
    if progress_preds is not None:
        if progress_preds.ndim == 1 and progress_preds.shape[0] > 0:
            progress = float(progress_preds[0].item())
        elif progress_preds.ndim > 1 and progress_preds.shape[1] > 0:
            progress = float(progress_preds[0, 0].item())
    progress = max(0.0, min(1.0, progress))
    return action, progress, latency_ms


def summarize_latencies(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {
            "num_steps": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
        }
    arr = np.asarray(latencies, dtype=np.float64)
    return {
        "num_steps": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def plot_latency(step_records: List[Dict[str, Any]], output_path: Path, episode_label: str) -> None:
    steps = [rec["step"] for rec in step_records]
    lat_ms = [rec["latency_ms"] for rec in step_records]
    plt.figure(figsize=(10, 4.8))
    if steps:
        plt.plot(steps, lat_ms, marker="o", linewidth=1.2, markersize=3)
    plt.xlabel("Step")
    plt.ylabel("Latency (ms)")
    plt.title(f"ThinkVLNActor Per-step Inference Latency ({episode_label})")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_env_config(args: argparse.Namespace) -> Any:
    cfg = get_habitat_config(args.cfg)
    with read_write(cfg):
        cfg.habitat.dataset.split = args.split
        if args.scenes_dir:
            cfg.habitat.dataset.scenes_dir = args.scenes_dir
        if args.data_path:
            cfg.habitat.dataset.data_path = args.data_path
        cfg.habitat.task.measurements.update(
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
    return cfg


def evaluate_one_episode(args: argparse.Namespace) -> Dict[str, Any]:
    device = resolve_device(args.cpu)
    if device.type == "cpu":
        log_phase("Running on CPU: startup may take several minutes for large checkpoints.")
    log_phase(f"Loading model from {args.model_path}")
    load_start = time.perf_counter()
    model, processor = load_thinkvln_actor_model(
        model_path=args.model_path,
        device=str(device),
        base_model_path=args.base_model_path,
    )
    log_phase(f"Model+processor loaded in {time.perf_counter() - load_start:.2f}s")

    log_phase("Building Habitat config")
    cfg = build_env_config(args)
    log_phase("Creating Habitat environment")
    env_start = time.perf_counter()
    env = Env(config=cfg)
    log_phase(f"Habitat environment ready in {time.perf_counter() - env_start:.2f}s")
    try:
        episodes = [
            ep
            for ep in env.episodes
            if hasattr(ep, "reference_path") and ep.reference_path is not None and len(ep.reference_path) > 1
        ]
        if not episodes:
            raise RuntimeError("No episodes with valid reference_path found.")

        episodes = sorted(episodes, key=lambda ep: (getattr(ep, "scene_id", ""), str(ep.episode_id)))
        if args.episode is not None:
            selected = [ep for ep in episodes if str(ep.episode_id) == str(args.episode)]
            if not selected:
                raise ValueError(f"Episode id {args.episode} not found in split={args.split}.")
            episode = selected[0]
        else:
            rng = random.Random(args.seed)
            episode = rng.choice(episodes)

        env.current_episode = episode
        obs = env.reset()

        instruction = get_instruction_text(args.cfg, episode)
        follower = ShortestPathFollower(sim=env.sim, goal_radius=0.5, return_one_hot=False)
        ref_path = episode.reference_path
        waypoint_id = 1

        step = 0
        step_records: List[Dict[str, Any]] = []

        while (not env.episode_over) and (step < args.steps):
            gt_action, waypoint_id, follower = next_shortest_path_action(env, follower, ref_path, waypoint_id)
            if gt_action is None:
                break

            if args.use_map:
                info = env.get_metrics()
                observation = prepare_image_with_map(obs["rgb"], info)
            else:
                observation = Image.fromarray(obs["rgb"]).convert("RGB")

            pred_action, pred_progress, latency_ms = predict_action_with_timing(
                model=model,
                processor=processor,
                instruction=instruction,
                subgoal=args.subgoal,
                observation=observation,
                device=device,
            )

            step_records.append(
                {
                    "step": step,
                    "latency_ms": float(latency_ms),
                    "gt_action": int(gt_action),
                    "pred_action": int(pred_action),
                    "pred_progress": float(pred_progress),
                }
            )
            if args.log_interval > 0 and ((step + 1) % args.log_interval == 0):
                log_phase(
                    f"step={step + 1}/{args.steps} latency_ms={latency_ms:.2f} "
                    f"pred_action={pred_action} gt_action={gt_action} progress={pred_progress:.3f}"
                )

            obs = env.step(gt_action)
            step += 1

        latencies = [rec["latency_ms"] for rec in step_records]
        scene_id = episode.scene_id.split("/")[-2] if hasattr(episode, "scene_id") else "unknown_scene"
        payload = {
            "config": {
                "model_path": args.model_path,
                "base_model_path": args.base_model_path,
                "cfg": args.cfg,
                "split": args.split,
                "data_path": args.data_path,
                "scenes_dir": args.scenes_dir,
                "seed": args.seed,
                "steps": args.steps,
                "episode": args.episode,
                "subgoal": args.subgoal,
                "use_map": bool(args.use_map),
                "device": str(device),
            },
            "episode": {
                "scene_id": scene_id,
                "episode_id": str(episode.episode_id),
                "instruction": instruction,
            },
            "summary": summarize_latencies(latencies),
            "steps": step_records,
        }
        return payload
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile ThinkVLNActor per-step forward latency on one episode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", type=str, help="ThinkVLNActor checkpoint path")
    parser.add_argument("--base-model-path", type=str, default=None, help="Base model path for LoRA checkpoints")
    parser.add_argument("-o", "--out", type=str, default="results/thinkvln_actor_latency", help="Output directory")
    parser.add_argument("-s", "--split", type=str, default="val_seen", help="Dataset split")
    parser.add_argument("-n", "--steps", type=int, default=500, help="Max rollout steps")
    parser.add_argument("-e", "--episode", type=str, default=None, help="Episode id (default: random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subgoal", type=str, default="Navigate to the goal.", help="Subgoal text prompt")
    parser.add_argument("--no-map", action="store_false", dest="use_map", help="Disable map concatenation")
    parser.set_defaults(use_map=True)
    parser.add_argument("--cpu", action="store_true", help="Use CPU")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print per-step progress every N rollout steps (<=0 disables)",
    )

    parser.add_argument("--cfg", type=str, default="config/vln_r2r.yaml", help="Habitat config path")
    parser.add_argument("--data-path", type=str, default=None, help="Dataset path override")
    parser.add_argument("--scenes-dir", type=str, default=None, help="Scenes dir override")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = evaluate_one_episode(args)

    json_path = output_dir / "latency_profile.json"
    png_path = output_dir / "latency_step_curve.png"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    episode = payload["episode"]
    episode_label = f"{episode['scene_id']}:{episode['episode_id']}"
    plot_latency(payload["steps"], png_path, episode_label)

    summary = payload["summary"]
    print("=" * 60)
    print("ThinkVLNActor Latency Profile")
    print("=" * 60)
    print(f"episode: {episode_label}")
    print(f"num_steps: {summary['num_steps']}")
    print(f"mean_ms: {summary['mean_ms']:.3f}")
    print(f"p95_ms: {summary['p95_ms']:.3f}")
    print(f"json: {json_path}")
    print(f"plot: {png_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
