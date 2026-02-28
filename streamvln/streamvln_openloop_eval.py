#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import copy
import json
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple

# Quiet Habitat-Sim C++ logs unless user explicitly overrides outside.
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np
import torch
import transformers
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import habitat
from habitat import Env
from habitat.config import read_write
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat_extensions import measures  # noqa: F401

from model.stream_video_vln import StreamVLNForCausalLM
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


def preprocess_depth(depth_obs: np.ndarray, image_processor, min_depth: float, max_depth: float) -> Tuple[np.ndarray, Tuple[int, int]]:
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
    env: Env,
    follower: ShortestPathFollower,
    ref_path: List[Any],
    waypoint_id: int,
) -> Tuple[Optional[int], int, ShortestPathFollower]:
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


def load_model_and_tokenizer(args) -> Tuple[Any, Any]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=args.model_max_length, padding_side="right"
    )
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.add_tokens(["<memory>"], special_tokens=True)

    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "config": config,
        "low_cpu_mem_usage": False,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = StreamVLNForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to(args.device)
    model.eval()
    model.reset(1)
    return model, tokenizer


def predict_action(
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
    device: str,
    max_new_tokens: int,
) -> Tuple[int, str]:
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
        "images": torch.stack(images).unsqueeze(0).to(device, dtype=torch.bfloat16),
        "depths": torch.stack(depths).unsqueeze(0).to(device, dtype=torch.bfloat16),
        "poses": torch.stack(poses).unsqueeze(0).to(device, dtype=torch.bfloat16),
        "intrinsics": torch.stack(intrinsics).unsqueeze(0).to(device, dtype=torch.bfloat16),
        "inputs": input_ids.to(device),
        "env_id": 0,
        "time_ids": time_ids,
        "task_type": [0],
    }

    with torch.no_grad():
        outputs = model.generate(
            **input_dict,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )

    decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0].strip()
    parsed = parse_actions(decoded)
    pred_action = parsed[0] if parsed else 0
    return pred_action, decoded


def evaluate(args) -> Dict[str, Any]:
    model, tokenizer = load_model_and_tokenizer(args)
    image_processor = model.get_vision_tower().image_processor

    cfg = get_habitat_config(args.habitat_config_path)
    with read_write(cfg):
        cfg.habitat.dataset.split = args.eval_split
        if args.scenes_dir:
            cfg.habitat.dataset.scenes_dir = args.scenes_dir
        if args.data_path:
            cfg.habitat.dataset.data_path = args.data_path

    env = Env(config=cfg)
    episodes = list(env.episodes)

    if args.max_episodes is not None and args.max_episodes > 0 and len(episodes) > args.max_episodes:
        random.seed(args.seed)
        episodes = random.sample(episodes, args.max_episodes)
    episodes = sorted(episodes, key=lambda ep: getattr(ep, "scene_id", ""))

    sim_sensors = cfg.habitat.simulator.agents.main_agent.sim_sensors
    camera_height = sim_sensors.rgb_sensor.position[1]
    min_depth = sim_sensors.depth_sensor.min_depth
    max_depth = sim_sensors.depth_sensor.max_depth
    intrinsic_matrix = get_intrinsic_matrix(sim_sensors.rgb_sensor)
    axis_align_matrix = get_axis_align_matrix()

    episode_results: List[Dict[str, Any]] = []
    total_steps = 0
    total_correct = 0
    total_episodes = 0

    pbar = tqdm(episodes, desc="Open-loop episodes")
    for episode in pbar:
        if not hasattr(episode, "reference_path") or episode.reference_path is None:
            continue

        env.current_episode = episode
        obs = env.reset()
        initial_height = env.sim.get_agent_state().position[1]
        instruction = get_instruction_text(episode)

        follower = ShortestPathFollower(sim=env.sim, goal_radius=0.5, return_one_hot=False)
        ref_path = episode.reference_path
        waypoint_id = 1

        gt_actions: List[int] = []
        pred_actions: List[int] = []
        decode_traces: List[str] = []
        step = 0
        image_history: List[torch.Tensor] = []
        depth_history: List[torch.Tensor] = []
        pose_history: List[torch.Tensor] = []
        intrinsic_history: List[torch.Tensor] = []

        while (not env.episode_over) and (step < args.max_steps):
            gt_action, waypoint_id, follower = next_shortest_path_action(env, follower, ref_path, waypoint_id)
            if gt_action is None:
                break

            rgb = obs["rgb"]
            image = Image.fromarray(rgb).convert("RGB")
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

            try:
                pred_action, decoded = predict_action(
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
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                pred_action = 0
                decoded = f"[prediction_error] {exc}"

            gt_actions.append(gt_action)
            pred_actions.append(pred_action)
            if args.save_decoded:
                decode_traces.append(decoded)

            obs = env.step(gt_action)
            step += 1

        if len(gt_actions) == 0:
            continue

        step_correct = sum(int(p == g) for p, g in zip(pred_actions, gt_actions))
        action_acc = step_correct / len(gt_actions)

        total_steps += len(gt_actions)
        total_correct += step_correct
        total_episodes += 1

        scene_id = episode.scene_id.split("/")[-2] if hasattr(episode, "scene_id") else "unknown_scene"
        episode_results.append(
            {
                "scene_id": scene_id,
                "episode_id": str(episode.episode_id),
                "instruction": instruction,
                "num_steps": len(gt_actions),
                "step_correct": step_correct,
                "action_accuracy": action_acc,
                "gt_actions": gt_actions,
                "pred_actions": pred_actions,
                "decoded_outputs": decode_traces if args.save_decoded else None,
            }
        )

        pbar.set_postfix(
            step_sr=f"{(total_correct / max(total_steps, 1)):.3f}",
        )

    env.close()

    metrics = {
        "num_episodes": total_episodes,
        "num_steps": total_steps,
        "step_wise_sr": (total_correct / total_steps) if total_steps > 0 else 0.0,
    }

    payload = {
        "config": {
            "model_path": args.model_path,
            "habitat_config_path": args.habitat_config_path,
            "eval_split": args.eval_split,
            "max_episodes": args.max_episodes,
            "max_steps": args.max_steps,
            "use_memory": args.use_memory,
            "memory_window": args.memory_window,
            "seed": args.seed,
        },
        "metrics": metrics,
        "episodes": episode_results,
    }

    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return payload


def main():
    parser = argparse.ArgumentParser(description="StreamVLN Open-loop Success Evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--eval_split", type=str, default="val_seen")
    parser.add_argument("--data_path", type=str, default=None, help="Optional override for habitat dataset data_path")
    parser.add_argument("--scenes_dir", type=str, default=None, help="Optional override for habitat scenes_dir")

    parser.add_argument("--max_episodes", type=int, default=20, help="Evaluate a small subset for quick open-loop check")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--use_memory", action="store_true", default=False)
    parser.add_argument("--no_memory", action="store_false", dest="use_memory")
    parser.add_argument("--memory_window", type=int, default=8, help="Number of historical frames for <memory> context")

    parser.add_argument("--save_decoded", action="store_true")
    parser.add_argument("--output_path", type=str, default="results/streamvln_openloop_eval.json")
    args = parser.parse_args()

    result = evaluate(args)
    metrics = result["metrics"]
    print("=" * 60)
    print("StreamVLN Open-loop Evaluation")
    print("=" * 60)
    print(f"num_episodes: {metrics['num_episodes']}")
    print(f"num_steps: {metrics['num_steps']}")
    print(f"step_wise_sr: {metrics['step_wise_sr']:.4f}")
    print("=" * 60)
    if args.output_path:
        print(f"saved: {args.output_path}")


if __name__ == "__main__":
    main()
