#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import copy
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
import transformers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils import DEFAULT_IMAGE_TOKEN, DEFAULT_MEMORY_TOKEN, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX

try:
    from depth_camera_filtering import filter_depth
except ImportError:
    def filter_depth(depth, blur_type=None):
        return depth


ACTIONS2IDX = {
    "STOP": 0,
    "↑": 1,
    "←": 2,
    "→": 3,
}
ACTION_PATTERN = re.compile("|".join(re.escape(a) for a in ACTIONS2IDX))


def parse_actions(text: str) -> List[int]:
    return [ACTIONS2IDX[token] for token in ACTION_PATTERN.findall(text)]


def build_prompt(instruction: str, use_memory: bool) -> str:
    prompt = (
        f"You are an autonomous navigation assistant. Your task is to {instruction}. "
        "Devise an action sequence to follow the instruction using the four actions: "
        "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
        "MOVE FORWARD (↑) by 25 centimeters, or STOP."
    )
    if use_memory:
        prompt += f" You have visited these areas {DEFAULT_MEMORY_TOKEN}."
    prompt += f" you can see {DEFAULT_IMAGE_TOKEN}."
    return prompt


def preprocess_qwen(prompt: str, tokenizer, add_system: bool = True) -> torch.Tensor:
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.chat_template = chat_template

    source = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]
    input_ids = []
    if add_system:
        input_ids += tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}])
    for conv in source:
        input_ids += tokenizer.apply_chat_template([conv])

    img_tok_id = tokenizer.convert_tokens_to_ids("<image>")
    mem_tok_id = tokenizer.convert_tokens_to_ids("<memory>")
    for i, tok in enumerate(input_ids):
        if tok == img_tok_id:
            input_ids[i] = IMAGE_TOKEN_INDEX
        elif tok == mem_tok_id:
            input_ids[i] = MEMORY_TOKEN_INDEX

    return torch.tensor([input_ids], dtype=torch.long)


def get_axis_align_matrix() -> torch.Tensor:
    return torch.tensor(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=torch.double,
    )


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    x, y, z = xyz
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0, x],
            [np.sin(yaw), np.cos(yaw), 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def get_intrinsic_matrix(sensor_cfg) -> np.ndarray:
    width = sensor_cfg.width
    height = sensor_cfg.height
    hfov = sensor_cfg.hfov
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.array(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def preprocess_intrinsic(intrinsic: np.ndarray, ori_size: Tuple[int, int], target_size: Tuple[int, int]) -> np.ndarray:
    intrinsic = np.array(intrinsic, copy=True)
    if intrinsic.ndim == 2:
        intrinsic = intrinsic[None, :, :]
    intrinsic[:, 0] /= ori_size[0] / target_size[0]
    intrinsic[:, 1] /= ori_size[1] / target_size[1]
    intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2
    return intrinsic.squeeze(0)


def preprocess_depth(
    depth_obs: np.ndarray,
    image_processor: Any,
    min_depth: float,
    max_depth: float,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    target_h = image_processor.crop_size["height"]
    target_w = image_processor.crop_size["width"]

    depth = depth_obs.reshape(depth_obs.shape[:2])
    depth = filter_depth(depth, blur_type=None)
    depth = depth * (max_depth - min_depth) + min_depth
    depth = depth * 1000.0

    resized = Image.fromarray(depth.astype(np.uint16), mode="I;16").resize((target_w, target_h), Image.NEAREST)
    depth_np = np.asarray(resized, dtype=np.float32) / 1000.0
    return depth_np, (target_w, target_h)


def get_instruction_text(episode: Any) -> str:
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
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

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


def resolve_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(args, device: torch.device) -> Tuple[Any, Any, torch.dtype]:
    from model.stream_video_vln import StreamVLNForCausalLM

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=args.model_max_length, padding_side="right"
    )
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.add_tokens(["<memory>"], special_tokens=True)

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model_kwargs = {
        "dtype": model_dtype,
        "config": config,
        "low_cpu_mem_usage": False,
    }
    attn_impl = args.attn
    if device.type != "cuda" and attn_impl == "flash_attention_2":
        attn_impl = None
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = StreamVLNForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    model.reset(1)
    return model, tokenizer, model_dtype


def predict_action_with_timing(
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    instruction: str,
    image_tensor: torch.Tensor,
    depth_tensor: torch.Tensor,
    pose_tensor: torch.Tensor,
    intrinsic_tensor: torch.Tensor,
    use_memory: bool,
    memory_window: int,
    image_history: List[torch.Tensor],
    depth_history: List[torch.Tensor],
    pose_history: List[torch.Tensor],
    intrinsic_history: List[torch.Tensor],
    step_id: int,
    device: torch.device,
    model_dtype: torch.dtype,
    max_new_tokens: int,
) -> Tuple[int, float, int]:
    model.reset_for_env(0)

    if use_memory and len(image_history) >= memory_window + 1:
        images = image_history[-(memory_window + 1):]
        depths = depth_history[-(memory_window + 1):]
        poses = pose_history[-(memory_window + 1):]
        intrinsics = intrinsic_history[-(memory_window + 1):]
        time_ids = [[max(0, step_id - memory_window) + i for i in range(memory_window + 1)]]
    else:
        images = [image_tensor]
        depths = [depth_tensor]
        poses = [pose_tensor]
        intrinsics = [intrinsic_tensor]
        time_ids = [[0]]

    prompt = build_prompt(instruction, use_memory=(len(images) > 1))
    input_ids = preprocess_qwen(prompt, tokenizer, add_system=True)
    input_dict = {
        "images": torch.stack(images).unsqueeze(0).to(device, dtype=model_dtype),
        "depths": torch.stack(depths).unsqueeze(0).to(device, dtype=model_dtype),
        "poses": torch.stack(poses).unsqueeze(0).to(device, dtype=model_dtype),
        "intrinsics": torch.stack(intrinsics).unsqueeze(0).to(device, dtype=model_dtype),
        "inputs": input_ids.to(device),
        "env_id": 0,
        "time_ids": time_ids,
        "task_type": [0],
    }

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **input_dict,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0].strip()
    parsed = parse_actions(decoded)
    pred_action = parsed[0] if parsed else 0
    return pred_action, latency_ms, len(images)


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
    plt.title(f"StreamVLN Per-step Inference Latency ({episode_label})")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_one_episode(args) -> Dict[str, Any]:
    from habitat import Env
    from habitat.config import read_write
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from habitat_baselines.config.default import get_config as get_habitat_config
    from habitat_extensions import measures as _measures  # noqa: F401

    device = resolve_device(args.cpu)
    model, tokenizer, model_dtype = load_model_and_tokenizer(args, device)
    image_processor = model.get_vision_tower().image_processor

    cfg = get_habitat_config(args.cfg)
    with read_write(cfg):
        cfg.habitat.dataset.split = args.split
        if args.scenes_dir:
            cfg.habitat.dataset.scenes_dir = args.scenes_dir
        if args.data_path:
            cfg.habitat.dataset.data_path = args.data_path

    env = Env(config=cfg)
    try:
        episodes = [
            ep for ep in env.episodes
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

        sim_sensors = cfg.habitat.simulator.agents.main_agent.sim_sensors
        camera_height = sim_sensors.rgb_sensor.position[1]
        min_depth = sim_sensors.depth_sensor.min_depth
        max_depth = sim_sensors.depth_sensor.max_depth
        intrinsic_matrix = get_intrinsic_matrix(sim_sensors.rgb_sensor)
        axis_align_matrix = get_axis_align_matrix()

        initial_height = env.sim.get_agent_state().position[1]
        instruction = get_instruction_text(episode)
        follower = ShortestPathFollower(sim=env.sim, goal_radius=0.5, return_one_hot=False)
        ref_path = episode.reference_path
        waypoint_id = 1

        step = 0
        step_records: List[Dict[str, Any]] = []
        image_history: List[torch.Tensor] = []
        depth_history: List[torch.Tensor] = []
        pose_history: List[torch.Tensor] = []
        intrinsic_history: List[torch.Tensor] = []

        while (not env.episode_over) and (step < args.steps):
            gt_action, waypoint_id, follower = next_shortest_path_action(env, follower, ref_path, waypoint_id)
            if gt_action is None:
                break

            image = Image.fromarray(obs["rgb"]).convert("RGB")
            image_size = image.size
            image_tensor = image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]

            depth_np, resized_depth = preprocess_depth(obs["depth"], image_processor, min_depth, max_depth)
            depth_tensor = torch.from_numpy(depth_np).float().unsqueeze(-1)

            x, y = obs["gps"]
            yaw = float(obs["compass"][0])
            state = env.sim.get_agent_state()
            height = state.position[1] - initial_height
            camera_position = np.array([x, -y, camera_height + height], dtype=np.float32)
            pose = torch.from_numpy(xyz_yaw_to_tf_matrix(camera_position, yaw)).double()
            pose = (pose @ axis_align_matrix).float()

            intrinsic = preprocess_intrinsic(intrinsic_matrix, image_size, resized_depth)
            intrinsic_tensor = torch.from_numpy(intrinsic).float()

            image_history.append(image_tensor)
            depth_history.append(depth_tensor)
            pose_history.append(pose)
            intrinsic_history.append(intrinsic_tensor)

            pred_action, latency_ms, memory_frames_used = predict_action_with_timing(
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                instruction=instruction,
                image_tensor=image_tensor,
                depth_tensor=depth_tensor,
                pose_tensor=pose,
                intrinsic_tensor=intrinsic_tensor,
                use_memory=args.use_memory,
                memory_window=args.memory_window,
                image_history=image_history,
                depth_history=depth_history,
                pose_history=pose_history,
                intrinsic_history=intrinsic_history,
                step_id=step,
                device=device,
                model_dtype=model_dtype,
                max_new_tokens=args.max_new_tokens,
            )

            step_records.append(
                {
                    "step": step,
                    "latency_ms": float(latency_ms),
                    "gt_action": int(gt_action),
                    "pred_action": int(pred_action),
                    "memory_frames_used": int(memory_frames_used),
                }
            )

            obs = env.step(gt_action)
            step += 1

        latencies = [rec["latency_ms"] for rec in step_records]
        scene_id = episode.scene_id.split("/")[-2] if hasattr(episode, "scene_id") else "unknown_scene"
        payload = {
            "config": {
                "model_path": args.model_path,
                "cfg": args.cfg,
                "split": args.split,
                "data_path": args.data_path,
                "scenes_dir": args.scenes_dir,
                "seed": args.seed,
                "steps": args.steps,
                "episode": args.episode,
                "use_memory": args.use_memory,
                "memory_window": args.memory_window,
                "device": str(device),
                "max_new_tokens": args.max_new_tokens,
                "num_history": args.num_history,
                "attn": args.attn,
                "model_dtype": str(model_dtype),
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
        description="Profile StreamVLN per-step generate latency on one episode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", type=str, help="StreamVLN checkpoint path")
    parser.add_argument("-o", "--out", type=str, default="results/streamvln_latency", help="Output directory")
    parser.add_argument("-s", "--split", type=str, default="val_seen", help="Dataset split")
    parser.add_argument("-n", "--steps", type=int, default=500, help="Max rollout steps")
    parser.add_argument("-e", "--episode", type=str, default=None, help="Episode id (default: random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-memory", action="store_false", dest="use_memory", help="Disable memory context")
    parser.set_defaults(use_memory=True)
    parser.add_argument("-w", "--memory-window", type=int, default=8, help="Memory window size")
    parser.add_argument("--cpu", action="store_true", help="Use CPU")

    parser.add_argument("--cfg", type=str, default="config/vln_r2r.yaml", help="Habitat config path")
    parser.add_argument("--data-path", type=str, default=None, help="Dataset path override")
    parser.add_argument("--scenes-dir", type=str, default=None, help="Scenes dir override")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generate max new tokens")
    parser.add_argument("--num-history", type=int, default=8, help="Model num_history")
    parser.add_argument("--attn", type=str, default="flash_attention_2", help="Attention implementation")
    parser.add_argument("--model-max-length", type=int, default=4096, help="Tokenizer model max length")
    args = parser.parse_args()

    if args.memory_window < 1:
        parser.error("--memory-window must be >= 1")
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
    print("StreamVLN Latency Profile")
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
