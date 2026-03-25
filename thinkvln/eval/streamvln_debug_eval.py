#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from thinkvln.datagen.generation.watcher_utils import action_id_to_name
from thinkvln.eval.close_eval_runner import VLNEvaluator as HabitatEvaluator
from thinkvln.eval.close_eval_utils import build_episode_key, extract_scene_id, load_summary_full, write_jsonl_record
from thinkvln.eval.two_system_eval import (
    TwoSystemEpisodeRunner,
    _sample_debug_candidates,
    _trace_without_images,
    _write_debug_episode_artifacts,
    load_config as load_two_system_eval_config,
)


DEFAULT_SAMPLE_CONFIG = "config/two_system_eval.local_debug20.yaml"
DEFAULT_STREAMVLN_MODEL_PATH = "/mnt/swx/ThinkVLN/model_weights/streamvln"
DEFAULT_OUTPUT_PATH = "/mnt/swx/ThinkVLN/results/streamvln_eval"
logger = logging.getLogger(__name__)


def _load_sample_config(path: str) -> Dict[str, Any]:
    return load_two_system_eval_config(path)


def _setup_logging(level_name: str) -> None:
    import logging

    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)


class StreamVLNDebugPolicy:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "cuda",
        num_frames: int = 32,
        num_future_steps: int = 4,
        num_history: int = 8,
        env_id: int = 0,
        max_new_tokens: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_frames = int(num_frames)
        self.num_future_steps = int(num_future_steps)
        self.num_history = int(num_history)
        self.env_id = int(env_id)
        self.max_new_tokens = int(max_new_tokens)
        self.image_processor = model.get_vision_tower().image_processor
        self.actions2idx = {
            "STOP": 0,
            "↑": 1,
            "←": 2,
            "→": 3,
        }
        self.conversation = [
            {
                "from": "human",
                "value": (
                    "<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. "
                    "Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) "
                    "or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
                ),
            },
            {"from": "gpt", "value": ""},
        ]
        self.conjunctions = [
            "you can see ",
            "in front of you is ",
            "there is ",
            "you can spot ",
            "you are toward the ",
            "ahead of you is ",
            "in your sight is ",
        ]
        self.model.eval()
        self.model.reset(1)
        self.reset_episode_state()

    def reset_episode_state(self, episode_key: str = "") -> None:
        self.episode_key = str(episode_key)
        self.rgb_list: List[torch.Tensor] = []
        self.depth_list: List[torch.Tensor] = []
        self.pose_list: List[torch.Tensor] = []
        self.intrinsic_list: List[torch.Tensor] = []
        self.time_ids: List[int] = []
        self.action_seq: List[int] = []
        self.output_ids = None
        self.past_key_values = None
        self.step_id = 0
        self.last_raw_output = ""
        self.last_generated_prompt = ""
        self.last_generated_actions: List[int] = []
        self._last_debug_snapshot: Optional[Dict[str, Any]] = None
        self.model.reset_for_env(self.env_id)

    def get_last_debug_snapshot(self) -> Optional[Dict[str, Any]]:
        return None if self._last_debug_snapshot is None else dict(self._last_debug_snapshot)

    def after_env_step(self) -> None:
        self.step_id += 1
        if self.step_id > 0 and self.step_id % self.num_frames == 0:
            self.model.reset_for_env(self.env_id)
            self.output_ids = None
            self.past_key_values = None
            self.time_ids = []

    def preprocess_depth_image(self, depth_image, do_depth_scale: bool = True, depth_scale: int = 1000):
        from transformers.image_utils import to_numpy_array

        target_height = self.image_processor.crop_size["height"]
        target_width = self.image_processor.crop_size["width"]
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
        return img, (target_width, target_height)

    @staticmethod
    def get_intrinsic_matrix(sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        return np.array(
            [
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def preprocess_intrinsic(intrinsic, ori_size, target_size):
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]
        intrinsic[:, 0] /= ori_size[0] / target_size[0]
        intrinsic[:, 1] /= ori_size[1] / target_size[1]
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2
        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)
        return intrinsic

    @staticmethod
    def get_axis_align_matrix():
        return torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()

    @staticmethod
    def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    def parse_actions(self, output: str) -> List[int]:
        action_patterns = "|".join(re.escape(action) for action in self.actions2idx)
        matches = re.compile(action_patterns).findall(output)
        return [self.actions2idx[match] for match in matches]

    def preprocess_qwen(
        self,
        sources,
        has_image: bool = False,
        system_message: str = "You are a helpful assistant.",
        add_system: bool = False,
    ):
        import transformers

        del transformers
        from streamvln.utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX

        roles = {"human": "user", "gpt": "assistant"}
        tokenizer = copy.deepcopy(self.tokenizer)
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
            "'<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
        tokenizer.chat_template = chat_template

        conversations = []
        input_ids = []
        for source in sources:
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]

            input_id = []
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])

            for conv in source:
                role = roles.get(conv.get("role", conv.get("from")), conv.get("role", conv.get("from")))
                content = conv.get("content", conv.get("value", ""))
                conversations.append(content)
                input_id += tokenizer.apply_chat_template([{"role": role, "content": content}])

            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
            input_ids.append(input_id)
        return torch.tensor(input_ids, dtype=torch.long), conversations

    def predict_action(
        self,
        observations: Dict[str, Any],
        env: Any,
        instruction: str,
        sensor_config: Any,
        initial_height: float,
    ) -> int:
        try:
            from depth_camera_filtering import filter_depth
        except ImportError:
            def filter_depth(depth, blur_type=None):
                del blur_type
                return depth

        from streamvln.utils.utils import DEFAULT_MEMORY_TOKEN, DEFAULT_VIDEO_TOKEN, dict_to_cuda

        self.time_ids.append(self.step_id)
        rgb = observations["rgb"]
        depth = observations["depth"]
        x, y = observations["gps"]
        camera_yaw = observations["compass"][0]
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        depth = depth * (sensor_config.depth_sensor.max_depth - sensor_config.depth_sensor.min_depth) + sensor_config.depth_sensor.min_depth
        depth = depth * 1000

        agent_state = env.sim.get_agent_state()
        height = agent_state.position[1] - initial_height
        camera_position = np.array([x, -y, sensor_config.rgb_sensor.position[1] + height])
        tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        image = Image.fromarray(rgb).convert("RGB")
        image_size = image.size
        image_tensor = self.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]
        depth_image, resize_shape = self.preprocess_depth_image(
            Image.fromarray(depth.astype(np.uint16), mode="I;16"),
            do_depth_scale=True,
        )
        intrinsic_matrix = self.get_intrinsic_matrix(sensor_config.rgb_sensor)
        intrinsic = self.preprocess_intrinsic(intrinsic_matrix, image_size, resize_shape)
        intrinsic = torch.from_numpy(intrinsic).float()

        self.rgb_list.append(image_tensor)
        self.depth_list.append(torch.from_numpy(depth_image).float())
        self.pose_list.append(torch.from_numpy(tf_camera_to_episodic) @ self.get_axis_align_matrix())
        self.intrinsic_list.append(intrinsic)

        generated_this_step = False
        used_cached_output = self.output_ids is not None
        selected_frame_count = 0
        history_frame_count = 0

        if len(self.action_seq) == 0:
            generated_this_step = True
            if self.output_ids is None:
                sources = copy.deepcopy(self.conversation)
                sources[0]["value"] = sources[0]["value"].replace(
                    " Where should you go next to stay on track?",
                    " Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.",
                )
                if self.step_id != 0:
                    sources[0]["value"] += f" These are your historical observations {DEFAULT_MEMORY_TOKEN}."
                sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN + "\n", "")
                sources[0]["value"] = sources[0]["value"].replace("<instruction>.", instruction)
                add_system = True
            else:
                sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                add_system = False

            input_ids, _ = self.preprocess_qwen([sources], has_image=True, add_system=add_system)
            if self.output_ids is not None:
                input_ids = torch.cat([self.output_ids, input_ids.to(self.output_ids.device)], dim=1)

            images = self.rgb_list[-1:]
            depths = self.depth_list[-1:]
            poses = self.pose_list[-1:]
            intrinsics = self.intrinsic_list[-1:]
            if self.step_id != 0 and self.step_id % self.num_frames == 0:
                if self.num_history is None:
                    history_ids = slice(0, self.time_ids[0], self.num_future_steps)
                else:
                    stride = max(1, self.time_ids[0] // self.num_history)
                    history_ids = slice(0, self.time_ids[0], stride)
                images = self.rgb_list[history_ids] + images
                depths = self.depth_list[history_ids] + depths
                poses = self.pose_list[history_ids] + poses
                intrinsics = self.intrinsic_list[history_ids] + intrinsics

            selected_frame_count = len(images)
            history_frame_count = max(0, selected_frame_count - 1)
            input_dict = {
                "images": torch.stack(images).unsqueeze(0),
                "depths": torch.stack(depths).unsqueeze(0),
                "poses": torch.stack(poses).unsqueeze(0),
                "intrinsics": torch.stack(intrinsics).unsqueeze(0),
                "inputs": input_ids,
                "env_id": self.env_id,
                "time_ids": [list(self.time_ids)],
                "task_type": [0],
            }
            input_dict = dict_to_cuda(input_dict, self.device)
            for key, value in input_dict.items():
                if key in {"images", "depths", "poses", "intrinsics"}:
                    input_dict[key] = value.to(torch.bfloat16)

            try:
                t_generate = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **input_dict,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=True,
                        return_dict_in_generate=True,
                        past_key_values=self.past_key_values,
                    )
                logger.info(
                    "[%s] step=%d streamvln.generate done in %.2fs",
                    self.episode_key or "unknown_episode",
                    self.step_id,
                    time.time() - t_generate,
                )
                self.output_ids = outputs.sequences
                self.past_key_values = outputs.past_key_values
                self.last_raw_output = self.tokenizer.batch_decode(self.output_ids, skip_special_tokens=False)[0].strip()
                self.last_generated_actions = self.parse_actions(self.last_raw_output)
                self.action_seq = list(self.last_generated_actions) if self.last_generated_actions else [0]
            except Exception as exc:
                self.output_ids = None
                self.past_key_values = None
                self.last_raw_output = f"<generation_error>\n{exc}"
                self.last_generated_actions = []
                self.action_seq = [0]

            self.last_generated_prompt = sources[0]["value"]

        buffer_before = list(self.action_seq)
        action = int(self.action_seq.pop(0)) if self.action_seq else 0
        self._last_debug_snapshot = {
            "step_id": int(self.step_id),
            "instruction": instruction,
            "generated_this_step": bool(generated_this_step),
            "used_cached_output": bool(used_cached_output),
            "time_ids": list(self.time_ids),
            "selected_frame_count": int(selected_frame_count),
            "history_frame_count": int(history_frame_count),
            "model_prompt": self.last_generated_prompt,
            "raw_output": self.last_raw_output,
            "generated_actions": list(self.last_generated_actions),
            "action_buffer_before": list(buffer_before),
            "returned_action": int(action),
            "action_buffer_after": list(self.action_seq),
        }
        return action


def _format_buffer(actions: Sequence[int]) -> str:
    return "[" + ", ".join(action_id_to_name(action) for action in actions) + "]"


def _build_debug_prompt(snapshot: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "Model Input:",
            f"Instruction: {snapshot.get('instruction', '')}",
            "RGB: current frame shown on the left",
            "",
            "Model Output:",
            f"Returned Action: {action_id_to_name(snapshot.get('returned_action', 0))}",
            "Model Prompt:",
            str(snapshot.get("model_prompt", "")),
            "",
            "Decoded Output:",
            str(snapshot.get("raw_output", "")),
            "",
            f"Parsed Generated Actions: {_format_buffer(snapshot.get('generated_actions', []))}",
        ]
    )


def _agent_state(env: Any) -> Dict[str, List[float]]:
    try:
        state = env.sim.get_agent_state()
        position = np.asarray(state.position, dtype=np.float32).tolist()
        rotation = np.asarray(state.rotation, dtype=np.float32).tolist()
        return {"position": position, "rotation": rotation}
    except Exception:
        return {"position": [], "rotation": []}


def _build_step_record(
    observation_image: Image.Image,
    instruction: str,
    step_index: int,
    action: int,
    env: Any,
    snapshot: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    state = _agent_state(env)
    return {
        "step_index": int(step_index),
        "env_step_index": int(step_index),
        "instruction": instruction,
        "image": observation_image,
        "action_id": int(action),
        "action": action_id_to_name(action),
        "position": state["position"],
        "rotation": state["rotation"],
        "actor_progress": 1.0 if int(action) == 0 else 0.0,
        "actor_done": bool(int(action) == 0),
        "active_plan_step": "",
        "watcher_hint": "",
        "watcher_subtask": "",
        "map_agent_coord": TwoSystemEpisodeRunner._current_map_agent_coord(env),
        "rollout_index": int(step_index),
        "step_in_rollout": 0,
        "is_rollout_start": True,
        "is_rollout_end": True,
        "actor_prompt": _build_debug_prompt(snapshot or {"instruction": instruction, "returned_action": action}),
        "predicted_waypoint": None,
    }


def _sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(metrics or {}).items()
        if str(key) != "top_down_map"
    }


def _episode_summary(episode_result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(episode_result.get("metrics", {}) or {})
    return {
        "episode_key": episode_result["episode_key"],
        "nav_success": bool(episode_result.get("nav_success", False)),
        "failed": bool(episode_result.get("failed", False)),
        "failure_reason": str(episode_result.get("failure_reason", "") or ""),
        "error": str(episode_result.get("error", "") or ""),
        "steps_total": int(episode_result.get("steps_total", 0)),
        "metrics": _sanitize_metrics(metrics),
    }


def _episode_payload(
    args: argparse.Namespace,
    sample_config: Dict[str, Any],
    episode_result: Dict[str, Any],
    instruction: str,
) -> Dict[str, Any]:
    return {
        "episode_key": episode_result["episode_key"],
        "instruction": instruction,
        "nav_success": bool(episode_result.get("nav_success", False)),
        "failed": bool(episode_result.get("failed", False)),
        "failure_reason": str(episode_result.get("failure_reason", "") or ""),
        "error": str(episode_result.get("error", "") or ""),
        "steps_total": int(episode_result.get("steps_total", 0)),
        "metrics": _sanitize_metrics(episode_result.get("metrics", {})),
        "trace": _trace_without_images(episode_result.get("trace", {"steps": [], "watcher_events": []})),
        "config": {
            "sample_config_path": str(args.sample_config_path),
            "model_path": str(args.model_path),
            "device": str(args.device),
            "num_frames": int(args.num_frames),
            "num_future_steps": int(args.num_future_steps),
            "num_history": int(args.num_history),
            "sample_seed": int(sample_config["debug"].get("sample_seed", 0) or 0),
            "sample_limit": int(sample_config["debug"].get("sample_limit", 1) or 1),
        },
    }


def _batch_summary(episode_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = len(episode_results)
    denom = max(evaluated, 1)
    return {
        "episodes_total": int(evaluated),
        "episodes_evaluated": int(evaluated),
        "success_rate": sum(float(bool(item.get("nav_success", False))) for item in episode_results) / denom,
        "avg_steps_total": sum(float(item.get("steps_total", 0)) for item in episode_results) / denom,
        "avg_spl": sum(float(item.get("metrics", {}).get("spl", 0.0)) for item in episode_results) / denom,
        "avg_oracle_success": sum(float(item.get("metrics", {}).get("oracle_success", 0.0)) for item in episode_results) / denom,
        "avg_distance_to_goal": sum(float(item.get("metrics", {}).get("distance_to_goal", 0.0)) for item in episode_results) / denom,
    }


def _result_row(scene_id: str, episode: Any, instruction: str, episode_result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(episode_result.get("metrics", {}) or {})
    return {
        "scene_id": scene_id,
        "episode_id": episode.episode_id,
        "success": float(metrics.get("success", 0.0)),
        "spl": float(metrics.get("spl", 0.0)),
        "os": float(metrics.get("oracle_success", 0.0)),
        "ne": float(metrics.get("distance_to_goal", 0.0)),
        "steps": int(episode_result.get("steps_total", 0)),
        "episode_instruction": instruction,
    }


def _append_result_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _build_env_evaluator(args: argparse.Namespace, sample_config: Dict[str, Any]) -> HabitatEvaluator:
    env_cfg = sample_config["env"]
    evaluator_args = SimpleNamespace(
        device=str(args.device),
        sample_rate=float(env_cfg.get("sample_rate", 1.0)),
        target_episode_key="",
    )
    return HabitatEvaluator(
        config_path=env_cfg["habitat_config_path"],
        split=env_cfg["eval_split"],
        env_num=1,
        output_path=str(args.output_path),
        nav_model=None,
        epoch=0,
        args=evaluator_args,
    )


def _collect_debug_candidates(
    evaluator: HabitatEvaluator,
    env: Any,
    rank: int,
    summary_full: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, Any, Dict[str, Any]]]:
    candidates: List[Tuple[str, Any, Dict[str, Any]]] = []
    for scene_id, episode in evaluator._iter_assigned_episodes(env, rank):
        episode_key = build_episode_key(scene_id, episode.episode_id)
        record = summary_full.get(episode_key)
        if record is not None:
            candidates.append((episode_key, episode, record))
    return candidates


def _select_debug_candidates(
    candidates: Sequence[Tuple[str, Any, Dict[str, Any]]],
    config: Dict[str, Any],
) -> List[Tuple[str, Any, Dict[str, Any]]]:
    return _sample_debug_candidates(
        candidates,
        sample_limit=int(config["debug"].get("sample_limit", 1) or 1),
        sample_seed=int(config["debug"].get("sample_seed", 0) or 0),
    )


def _ensure_llava_next_available() -> None:
    thinkvln_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    possible_llava_paths = [
        os.path.join(thinkvln_root, "third_party", "LLaVA-NeXT"),
        os.path.join(thinkvln_root, "LLaVA-NeXT"),
        os.path.expanduser("~/LLaVA-NeXT"),
    ]
    if thinkvln_root not in sys.path:
        sys.path.insert(0, thinkvln_root)

    for llava_path in possible_llava_paths:
        if os.path.exists(llava_path) and os.path.exists(os.path.join(llava_path, "llava")):
            if llava_path not in sys.path:
                sys.path.insert(0, llava_path)
            compat_patch_path = os.path.join(llava_path, "llava", "compat_patch.py")
            if os.path.exists(compat_patch_path):
                import importlib.util

                spec = importlib.util.spec_from_file_location("llava.compat_patch", compat_patch_path)
                compat_patch = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(compat_patch)
            return
    raise ImportError("LLaVA-NeXT not found. StreamVLN requires LLaVA-NeXT codebase.")


def _load_streamvln_policy(
    args: argparse.Namespace,
    device: str,
    rank: int,
    world_size: int,
) -> StreamVLNDebugPolicy:
    import transformers

    _ensure_llava_next_available()
    from streamvln.model.stream_video_vln import StreamVLNForCausalLM

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    if not hasattr(config, "layer_types") or config.layer_types is None:
        num_layers = getattr(config, "num_hidden_layers", 32)
        sliding_window = getattr(config, "sliding_window", None)
        max_window_layers = getattr(config, "max_window_layers", num_layers)
        if sliding_window is not None:
            config.layer_types = [
                "sliding_attention" if i >= max_window_layers else "full_attention"
                for i in range(num_layers)
            ]
        else:
            config.layer_types = ["full_attention"] * num_layers

    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )
    model.model.num_history = int(args.num_history)
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    model.reset(world_size)
    return StreamVLNDebugPolicy(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_frames=args.num_frames,
        num_future_steps=args.num_future_steps,
        num_history=args.num_history,
        env_id=rank,
        max_new_tokens=args.max_new_tokens,
    )


def run_streamvln_episode(
    policy: StreamVLNDebugPolicy,
    evaluator: HabitatEvaluator,
    env: Any,
    episode: Any,
    episode_key: str,
    instruction: str,
    episode_step_cap: Optional[int] = None,
) -> Dict[str, Any]:
    env.current_episode = episode
    observations = env.reset()
    initial_height = env.sim.get_agent_state().position[1]
    policy.reset_episode_state(episode_key=episode_key)

    trace: Dict[str, Any] = {"steps": [], "watcher_events": []}
    step_cap = None if episode_step_cap is None else max(1, int(episode_step_cap))
    failed = False
    failure_reason = ""
    error = ""

    try:
        while not env.episode_over:
            if step_cap is not None and len(trace["steps"]) >= step_cap:
                failed = True
                failure_reason = "episode_step_cap_exceeded"
                break

            observation_image = Image.fromarray(np.asarray(observations["rgb"], dtype=np.uint8)).convert("RGB")
            action = policy.predict_action(
                observations=observations,
                env=env,
                instruction=instruction,
                sensor_config=evaluator.config.habitat.simulator.agents.main_agent.sim_sensors,
                initial_height=initial_height,
            )
            snapshot = policy.get_last_debug_snapshot()
            step_record = _build_step_record(
                observation_image=observation_image,
                instruction=instruction,
                step_index=len(trace["steps"]),
                action=action,
                env=env,
                snapshot=snapshot,
            )
            trace["steps"].append(step_record)
            observations = env.step(int(action))
            policy.after_env_step()
    except Exception as exc:
        failed = True
        failure_reason = "episode_runtime_error"
        error = str(exc)

    metrics = _sanitize_metrics(dict(env.get_metrics()))
    return {
        "episode_key": episode_key,
        "nav_success": bool(float(metrics.get("success", 0.0)) > 0.0),
        "failed": bool(failed),
        "failure_reason": str(failure_reason),
        "error": str(error),
        "steps_total": int(len(trace["steps"])),
        "trace": trace,
        "debug_top_down_map": TwoSystemEpisodeRunner._current_top_down_map_info(env),
        "debug_reference_path_map_coords": TwoSystemEpisodeRunner._reference_path_map_coords(env),
        "metrics": metrics,
    }


def evaluate_debug(args: argparse.Namespace) -> Dict[str, Any]:
    sample_config = _load_sample_config(str(args.sample_config_path))
    summary_full = load_summary_full(sample_config["env"]["summary_full_path"])
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rank = 0
    world_size = 1
    device = str(args.device)
    evaluator = _build_env_evaluator(args, sample_config)
    env = evaluator.config_env()
    policy = _load_streamvln_policy(args, device=device, rank=rank, world_size=world_size)
    candidates = _collect_debug_candidates(
        evaluator=evaluator,
        env=env,
        rank=rank,
        summary_full=summary_full,
    )
    selected = _select_debug_candidates(candidates, sample_config)
    if not selected:
        env.close()
        raise ValueError("No debug episodes were selected.")

    episode_summaries: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []
    result_path = output_dir / "result.json"
    episodes_path = output_dir / "episodes.jsonl"
    if result_path.exists():
        result_path.unlink()
    if episodes_path.exists():
        episodes_path.unlink()

    for episode_key, episode, _ in selected:
        instruction = evaluator._episode_instruction(sample_config["env"]["habitat_config_path"], episode)
        episode_result = run_streamvln_episode(
            policy=policy,
            evaluator=evaluator,
            env=env,
            episode=episode,
            episode_key=episode_key,
            instruction=instruction,
            episode_step_cap=getattr(args, "episode_step_cap", None),
        )
        summary_payload = _episode_summary(episode_result)
        episode_payload = _episode_payload(args, sample_config, episode_result, instruction)
        _write_debug_episode_artifacts(
            output_dir=output_dir,
            episode_result=episode_result,
            episode_payload=episode_payload,
            summary_payload=summary_payload,
            fps=int(args.debug_video_fps),
            single_episode=False,
        )
        manifest_row = {"episode_key": episode_key, "summary": summary_payload}
        episode_summaries.append(manifest_row)
        with open(episodes_path, "a", encoding="utf-8") as handle:
            write_jsonl_record(handle=handle, payload=manifest_row, sync_to_disk=False)
        scene_id = extract_scene_id(str(episode.scene_id))
        result_row = _result_row(scene_id, episode, instruction, episode_result)
        result_rows.append(result_row)
        _append_result_json(result_path, result_row)

    env.close()

    summary = _batch_summary([item["summary"] for item in episode_summaries])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    aggregate_result = {
        "sucs_all": sum(float(item["success"]) for item in result_rows) / max(len(result_rows), 1),
        "spls_all": sum(float(item["spl"]) for item in result_rows) / max(len(result_rows), 1),
        "oss_all": sum(float(item["os"]) for item in result_rows) / max(len(result_rows), 1),
        "ones_all": sum(float(item["ne"]) for item in result_rows) / max(len(result_rows), 1),
        "length": len(result_rows),
    }
    _append_result_json(result_path, aggregate_result)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StreamVLN debug eval on the same sampled episodes as two_system_eval.")
    parser.add_argument("--sample-config-path", type=str, default=DEFAULT_SAMPLE_CONFIG)
    parser.add_argument("--model-path", type=str, default=DEFAULT_STREAMVLN_MODEL_PATH)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--num-future-steps", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--debug-video-fps", type=int, default=2)
    parser.add_argument("--episode-step-cap", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)
    return evaluate_debug(args)


if __name__ == "__main__":
    main()
