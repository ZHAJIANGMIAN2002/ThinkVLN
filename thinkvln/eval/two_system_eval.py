#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import html as html_lib
import io
import json
import logging
import os
import random
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from thinkvln.datagen.generation.watcher_utils import action_id_to_name, extract_json_object
from thinkvln.dataset.watcher_sft_dataset import (
    ROLLOUT_SYSTEM_PROMPT,
    build_rollout_prompt as build_watcher_rollout_prompt,
)
from thinkvln.eval.close_eval_utils import load_summary_full, parse_plan_steps, write_jsonl_record


logger = logging.getLogger(__name__)

DEFAULT_STAGNATION_DISTANCE = 0.05
MAX_ACTOR_CALLS_PER_EPISODE = 128
DATAGEN_WATCHER_MODEL_NAME = "qwen/qwen3.5-397b-a17b"
DATAGEN_WATCHER_API_BASE_URL = "https://openrouter.ai/api/v1"
DATAGEN_WATCHER_API_KEY_ENV = "OPENROUTER_API_KEY"
DATAGEN_ROLLOUT_SYSTEM_PROMPT = ROLLOUT_SYSTEM_PROMPT


REQUIRED_CONFIG_KEYS = {
    "actor": {
        "model_type",
        "model_path",
        "base_model_path",
        "device",
        "memory_num_history_images",
        "done_threshold",
    },
    "watcher": {
        "backend",
        "model_name",
        "model_path",
        "base_model_path",
        "image_stride",
        "api_base_url",
        "api_key_env",
        "reasoning_effort",
        "request_timeout",
        "max_retries",
    },
    "env": {
        "habitat_config_path",
        "summary_full_path",
        "eval_split",
        "sample_rate",
        "target_episode_key",
    },
    "rollout": {
        "max_steps_per_wakeup",
        "progress_threshold",
        "max_watcher_wakeups",
        "episode_step_cap",
        "forbidden_actions",
    },
    "output": {
        "output_dir",
        "save_trace_jsonl",
        "save_summary_json",
        "save_step_debug_html",
        "save_debug_video",
        "debug_video_fps",
    },
    "debug": {
        "enabled",
        "episode_key",
        "output_dir",
        "video_fps",
    },
    "runtime": {
        "log_level",
        "world_size",
        "dist_url",
        "dist_timeout_minutes",
    },
}


@dataclass
class WatcherPrompt:
    system_prompt: str
    user_text: str
    images: List[Image.Image]


@dataclass
class WatcherTodoState:
    done_steps: List[str]
    active_step: str
    pending_steps: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "done_steps": list(self.done_steps),
            "active_step": self.active_step,
            "pending_steps": list(self.pending_steps),
        }


@dataclass
class WatcherDecision:
    memory: str
    done: bool
    subtask: str
    raw_response: Dict[str, Any]
    wakeup_reason: str


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to two-system eval YAML config")
    parser.add_argument("--local_rank", default=0, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--local-rank", dest="local_rank", default=0, type=int, help=argparse.SUPPRESS)
    return parser


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    missing: List[str] = []
    for section, keys in REQUIRED_CONFIG_KEYS.items():
        section_payload = config.get(section)
        if not isinstance(section_payload, dict):
            missing.append(section)
            continue
        for key in sorted(keys):
            if key not in section_payload:
                missing.append(f"{section}.{key}")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required config keys: {joined}")
    return config


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return _validate_config(_load_yaml(config_path))


def _plan_lines(plan_steps: Sequence[str]) -> str:
    return "\n".join(f"- {step}" for step in plan_steps) if plan_steps else "- (empty)"


def _plan_progress_text(todo_state: WatcherTodoState) -> str:
    total = len(todo_state.done_steps) + len(todo_state.pending_steps) + (1 if todo_state.active_step else 0)
    completed = len(todo_state.done_steps)
    return f"{completed}/{max(total, 1)} completed"


def _format_plan_section(plan_steps: Sequence[str]) -> str:
    steps = [str(step).strip() for step in plan_steps if str(step).strip()]
    if not steps:
        return "- None."
    return "\n".join(f"- {step}" for step in steps)


def _sample_rollout_images(
    rollout_images: Sequence[Image.Image],
    image_stride: int,
) -> List[Image.Image]:
    images = list(rollout_images)
    if not images:
        return []
    stride = max(1, int(image_stride))
    sampled = images[::stride]
    if sampled[-1] is not images[-1]:
        sampled.append(images[-1])
    return sampled


def _watcher_image_stride(image_stride: Any) -> int:
    stride = int(image_stride)
    if stride < 1:
        raise ValueError("watcher.image_stride must be >= 1")
    return stride


def build_init_prompt(
    instruction: str,
    plan_steps: Sequence[str],
    first_observation: Image.Image,
) -> WatcherPrompt:
    done_steps: List[str] = []
    active_step = str(plan_steps[0]).strip() if plan_steps else ""
    pending_steps = [str(step).strip() for step in plan_steps[1:] if str(step).strip()]
    prompt = build_watcher_rollout_prompt(
        instruction=instruction,
        plan_steps=plan_steps,
        done_steps=done_steps,
        active_step=active_step,
        pending_steps=pending_steps,
        memory_start="",
        rollout_actions=[],
        rollout_images=[first_observation],
    )
    return WatcherPrompt(
        system_prompt=prompt.system_prompt,
        user_text=prompt.user_text,
        images=prompt.images,
    )


def build_update_prompt(
    instruction: str,
    todo_state: WatcherTodoState,
    memory_text: str,
    rollout_images: Sequence[Image.Image],
    rollout_actions: Sequence[str],
    image_stride: int = 1,
) -> WatcherPrompt:
    sampled_images = _sample_rollout_images(rollout_images, image_stride=image_stride)
    action_lines = "\n".join(f"- {action}" for action in rollout_actions) if rollout_actions else "- (none)"
    system_prompt = (
        "You are the watcher for a two-system VLN evaluator.\n"
        "Respond with one JSON object only.\n"
        'Use this schema exactly: {"memory":"...","done":true/false,"subtask":"..."}\n'
        "Judge whether the active plan step has reached a natural handoff.\n"
        "Keep `memory` cumulative, concise, and useful for the actor.\n"
        "If `done` is false, `subtask` should continue refining the current active step.\n"
        "If `done` is true, `subtask` should be the next actor-facing text after the handoff."
    )
    user_text = (
        f"Instruction:\n{instruction}\n\n"
        "Plan State:\n"
        f"Plan Progress:\n{_plan_progress_text(todo_state)}\n\n"
        f"Done:\n{_plan_lines(todo_state.done_steps)}\n\n"
        f"Active:\n- {todo_state.active_step or '(none)'}\n\n"
        f"Pending:\n{_plan_lines(todo_state.pending_steps)}\n\n"
        f"Prior memory:\n{memory_text or '(empty)'}\n\n"
        f"Rollout actions:\n{action_lines}\n\n"
        "Use the rollout images and actions to update the memory and decide whether to hand off."
    )
    return WatcherPrompt(
        system_prompt=system_prompt,
        user_text=user_text,
        images=sampled_images,
    )


def build_api_update_prompt(
    todo_state: WatcherTodoState,
    memory_text: str,
    rollout_images: Sequence[Image.Image],
    rollout_actions: Sequence[str],
    image_stride: int = 1,
) -> WatcherPrompt:
    sampled_images = _sample_rollout_images(rollout_images, image_stride=image_stride)
    prompt = build_watcher_rollout_prompt(
        instruction="",
        plan_steps=[],
        done_steps=todo_state.done_steps,
        active_step=todo_state.active_step,
        pending_steps=todo_state.pending_steps,
        memory_start=memory_text or "",
        rollout_actions=rollout_actions,
        rollout_images=sampled_images,
    )
    return WatcherPrompt(
        system_prompt=prompt.system_prompt,
        user_text=prompt.user_text,
        images=prompt.images,
    )


def _encode_image_content(image: Image.Image) -> Dict[str, Any]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sanitize_episode_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(metrics or {}).items()
        if str(key) != "top_down_map"
    }


def _episode_nav_success(episode_result: Dict[str, Any]) -> bool:
    if "nav_success" in episode_result:
        return bool(episode_result.get("nav_success", False))
    metrics = dict(episode_result.get("metrics", {}) or {})
    try:
        return bool(float(metrics.get("success", 0.0)) > 0.0)
    except Exception:
        return False


def _trace_without_images(trace: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(trace or {})
    sanitized["steps"] = [
        {str(key): value for key, value in dict(step).items() if str(key) != "image"}
        for step in trace.get("steps", [])
    ]
    sanitized["watcher_events"] = list(trace.get("watcher_events", []))
    return sanitized


def _debug_episode_key(config: Dict[str, Any]) -> str:
    return str(config["debug"]["episode_key"] or "").strip()


def _debug_sample_limit(config: Dict[str, Any]) -> int:
    limit = int(config["debug"].get("sample_limit", 1) or 1)
    if limit < 1:
        raise ValueError("debug.sample_limit must be >= 1")
    return limit


def _debug_sample_seed(config: Dict[str, Any]) -> int:
    return int(config["debug"].get("sample_seed", 0) or 0)


def _debug_output_dir(config: Dict[str, Any], episode_key: str = "") -> Path:
    configured = str(config["debug"]["output_dir"] or "").strip()
    if configured:
        return Path(configured)
    if episode_key:
        return Path(config["output"]["output_dir"]) / f"debug_{episode_key}"
    return Path(config["output"]["output_dir"]) / f"debug_sample_{_debug_sample_limit(config)}"


def _debug_summary(episode_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "episode_key": episode_result["episode_key"],
        "episodes_total": 1,
        "episodes_evaluated": 1,
        "nav_success": _episode_nav_success(episode_result),
        "watcher_complete": bool(episode_result["watcher_complete"]),
        "failed": bool(episode_result.get("failed", False)),
        "failure_reason": str(episode_result.get("failure_reason", "") or ""),
        "error": str(episode_result.get("error", "") or ""),
        "steps_total": int(episode_result["steps_total"]),
        "watcher_wakeups": int(episode_result["watcher_wakeups"]),
        "done_steps": list(episode_result["done_steps"]),
        "metrics": _sanitize_episode_metrics(episode_result["metrics"]),
    }


def _debug_episode_payload(
    config: Dict[str, Any],
    episode_result: Dict[str, Any],
    instruction: str,
    plan_steps: Sequence[str],
    output_dir: Path,
) -> Dict[str, Any]:
    return {
        "episode_key": episode_result["episode_key"],
        "instruction": instruction,
        "plan_steps": list(plan_steps),
        "nav_success": _episode_nav_success(episode_result),
        "watcher_complete": episode_result["watcher_complete"],
        "failed": episode_result.get("failed", False),
        "failure_reason": episode_result.get("failure_reason", ""),
        "error": episode_result.get("error", ""),
        "steps_total": episode_result["steps_total"],
        "watcher_wakeups": episode_result["watcher_wakeups"],
        "done_steps": list(episode_result["done_steps"]),
        "final_memory": episode_result.get("final_memory", ""),
        "metrics": _sanitize_episode_metrics(episode_result["metrics"]),
        "trace": _trace_without_images(episode_result["trace"]),
        "config": {
            "actor": {
                "model_type": config["actor"]["model_type"],
                "model_path": config["actor"]["model_path"],
            },
            "watcher": {
                "backend": config["watcher"]["backend"],
                "model_name": config["watcher"]["model_name"],
                "model_path": config["watcher"]["model_path"],
            },
            "rollout": dict(config["rollout"]),
            "debug": {
                "enabled": bool(config["debug"]["enabled"]),
                "episode_key": config["debug"]["episode_key"],
                "sample_limit": int(config["debug"].get("sample_limit", 1) or 1),
                "sample_seed": int(config["debug"].get("sample_seed", 0) or 0),
                "output_dir": str(output_dir),
                "video_fps": int(config["debug"]["video_fps"]),
            },
        },
    }


def _debug_batch_summary(episode_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated = int(len(episode_results))
    denom = max(evaluated, 1)
    return {
        "episodes_total": evaluated,
        "episodes_evaluated": evaluated,
        "nav_success_rate": sum(float(bool(item.get("nav_success", False))) for item in episode_results) / denom,
        "watcher_complete_rate": sum(float(bool(item["watcher_complete"])) for item in episode_results) / denom,
        "avg_steps_total": sum(float(item["steps_total"]) for item in episode_results) / denom,
        "avg_watcher_wakeups": sum(float(item["watcher_wakeups"]) for item in episode_results) / denom,
        "avg_success": sum(float(item["metrics"].get("success", 0.0)) for item in episode_results) / denom,
    }


def _sample_debug_candidates(
    candidates: Sequence[tuple[str, Any, Dict[str, Any]]],
    sample_limit: int,
    sample_seed: int,
) -> List[tuple[str, Any, Dict[str, Any]]]:
    items = list(candidates)
    if len(items) <= int(sample_limit):
        return items
    return random.Random(int(sample_seed)).sample(items, int(sample_limit))


def _write_debug_episode_artifacts(
    output_dir: Path,
    episode_result: Dict[str, Any],
    episode_payload: Dict[str, Any],
    summary_payload: Dict[str, Any],
    fps: int,
    single_episode: bool,
) -> None:
    if single_episode:
        (output_dir / "episode.json").write_text(json.dumps(episode_payload, indent=2), encoding="utf-8")
        (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    else:
        episode_dir = output_dir / "episodes"
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / f"{episode_result['episode_key']}.json").write_text(
            json.dumps(episode_payload, indent=2),
            encoding="utf-8",
        )
    _write_debug_html(output_dir / "debug" / f"{episode_result['episode_key']}.html", episode_result)
    _write_debug_video(
        output_dir / "debug_video" / f"{episode_result['episode_key']}.mp4",
        episode_result,
        fps=fps,
    )


def _build_reasoning_config(reasoning_effort: str) -> Dict[str, Any]:
    effort = str(reasoning_effort or "none").strip().lower()
    if effort == "none":
        return {"effort": "none"}
    return {"effort": effort, "exclude": True}


def _build_api_messages(prompt: WatcherPrompt) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt.user_text}]
    for image in prompt.images:
        content.append(_encode_image_content(image))
    return [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": content},
    ]


def _normalize_watcher_payload(payload: Dict[str, Any], wakeup_reason: str) -> WatcherDecision:
    memory = str(payload.get("memory", "") or "").strip()
    subtask = str(payload.get("subtask", "") or "").strip()
    if not isinstance(payload.get("done"), bool):
        raise ValueError("Watcher response must contain boolean `done`.")
    return WatcherDecision(
        memory=memory,
        done=bool(payload["done"]),
        subtask=subtask,
        raw_response=dict(payload),
        wakeup_reason=wakeup_reason,
    )


def _normalize_api_update_payload(payload: Dict[str, Any], wakeup_reason: str) -> WatcherDecision:
    memory = str(payload.get("memory_end", "") or "").strip()
    subtask = str(payload.get("next_subtask", "") or "").strip()
    if not isinstance(payload.get("done"), bool):
        raise ValueError("Watcher response must contain boolean `done`.")
    return WatcherDecision(
        memory=memory,
        done=bool(payload["done"]),
        subtask=subtask,
        raw_response=dict(payload),
        wakeup_reason=wakeup_reason,
    )


class ApiWatcherBackend:
    def __init__(
        self,
        model_name: str,
        api_base_url: Optional[str] = None,
        image_stride: int = 1,
        api_key_env: str = "OPENAI_API_KEY",
        reasoning_effort: str = "none",
        request_timeout: float = 180.0,
        max_retries: int = 3,
        client: Any = None,
    ):
        self.model_name = model_name
        self.api_base_url = api_base_url
        self.image_stride = _watcher_image_stride(image_stride)
        self.api_key_env = api_key_env
        self.reasoning_effort = reasoning_effort
        self.request_timeout = float(request_timeout)
        self.max_retries = max(1, int(max_retries))
        self.client = client

    def _get_client(self):
        if self.client is not None:
            return self.client
        from openai import OpenAI

        api_key = os.environ.get(self.api_key_env, "EMPTY")
        self.client = OpenAI(
            base_url=self.api_base_url or None,
            api_key=api_key,
        )
        return self.client

    def _request_json(self, prompt: WatcherPrompt) -> Dict[str, Any]:
        client = self._get_client()
        last_error: Optional[Exception] = None
        messages = _build_api_messages(prompt)
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=self.request_timeout,
                    extra_body={"reasoning": _build_reasoning_config(self.reasoning_effort)},
                )
                content = response.choices[0].message.content or ""
                return extract_json_object(content)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "watcher api request failed attempt %d/%d: %s: %s",
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    exc,
                )
        raise RuntimeError(f"Watcher API request failed after retries: {last_error}")

    def initialize(
        self,
        instruction: str,
        plan_steps: Sequence[str],
        first_observation: Image.Image,
        episode_key: str,
    ) -> WatcherDecision:
        del episode_key
        prompt = build_init_prompt(
            instruction=instruction,
            plan_steps=plan_steps,
            first_observation=first_observation,
        )
        return _normalize_api_update_payload(self._request_json(prompt), wakeup_reason="init")

    def update(
        self,
        instruction: str,
        todo_state: WatcherTodoState,
        memory_text: str,
        rollout_slice: Sequence[Dict[str, Any]],
        episode_key: str,
    ) -> WatcherDecision:
        del instruction, episode_key
        rollout_images = [item["image"] for item in rollout_slice if isinstance(item.get("image"), Image.Image)]
        rollout_actions = [str(item.get("action", "")) for item in rollout_slice]
        prompt = build_api_update_prompt(
            todo_state=todo_state,
            memory_text=memory_text,
            rollout_images=rollout_images,
            rollout_actions=rollout_actions,
            image_stride=self.image_stride,
        )
        return _normalize_api_update_payload(self._request_json(prompt), wakeup_reason="")


class LocalWatcherBackend:
    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        device: str = "cuda",
        image_stride: int = 1,
        max_new_tokens: int = 256,
    ):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = device
        self.image_stride = _watcher_image_stride(image_stride)
        self.max_new_tokens = int(max_new_tokens)
        self.model = None
        self.processor = None

    def _resolve_base_model_path(self) -> Optional[str]:
        if self.base_model_path:
            return self.base_model_path
        adapter_config_path = Path(self.model_path) / "adapter_config.json"
        if not adapter_config_path.is_file():
            return None
        with open(adapter_config_path, "r", encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        return adapter_config.get("base_model_name_or_path")

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        base_model_path = self._resolve_base_model_path()
        load_path = base_model_path or self.model_path
        dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            load_path,
            dtype=dtype,
            device_map="cpu" if base_model_path else None,
            trust_remote_code=True,
        )
        if base_model_path:
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _generate_json(self, prompt: WatcherPrompt) -> Dict[str, Any]:
        import torch

        self._ensure_loaded()
        content = [{"type": "text", "text": prompt.user_text}]
        content.extend({"type": "image", "image": image} for image in prompt.images)
        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[prompt.images] if prompt.images else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return extract_json_object(generated_text)

    def initialize(
        self,
        instruction: str,
        plan_steps: Sequence[str],
        first_observation: Image.Image,
        episode_key: str,
    ) -> WatcherDecision:
        del episode_key
        prompt = build_init_prompt(
            instruction=instruction,
            plan_steps=plan_steps,
            first_observation=first_observation,
        )
        return _normalize_api_update_payload(self._generate_json(prompt), wakeup_reason="init")

    def update(
        self,
        instruction: str,
        todo_state: WatcherTodoState,
        memory_text: str,
        rollout_slice: Sequence[Dict[str, Any]],
        episode_key: str,
    ) -> WatcherDecision:
        del episode_key
        prompt = build_update_prompt(
            instruction=instruction,
            todo_state=todo_state,
            memory_text=memory_text,
            rollout_images=[item["image"] for item in rollout_slice if isinstance(item.get("image"), Image.Image)],
            rollout_actions=[str(item.get("action", "")) for item in rollout_slice],
            image_stride=self.image_stride,
        )
        return _normalize_watcher_payload(self._generate_json(prompt), wakeup_reason="")


class TwoSystemEpisodeRunner:
    def __init__(
        self,
        nav_model: Any,
        watcher_backend: Any,
        max_steps_per_wakeup: int,
        progress_threshold: float,
        episode_step_cap: int,
        max_watcher_wakeups: Optional[int] = None,
        forbidden_actions: Optional[Sequence[int]] = None,
        max_actor_calls_per_episode: int = MAX_ACTOR_CALLS_PER_EPISODE,
    ):
        self.nav_model = nav_model
        self.watcher_backend = watcher_backend
        self.max_steps_per_wakeup = max(1, int(max_steps_per_wakeup))
        self.progress_threshold = float(progress_threshold)
        self.max_watcher_wakeups = None if max_watcher_wakeups is None else max(1, int(max_watcher_wakeups))
        self.episode_step_cap = max(1, int(episode_step_cap))
        self.forbidden_actions = [int(action) for action in (forbidden_actions or [])]
        self.max_actor_calls_per_episode = max(1, int(max_actor_calls_per_episode))
        self.stagnation_distance_threshold = DEFAULT_STAGNATION_DISTANCE

    @staticmethod
    def _observation_to_image(observation: Any) -> Image.Image:
        if isinstance(observation, Image.Image):
            return observation.convert("RGB")
        if isinstance(observation, dict):
            image = observation.get("rgb")
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            if image is not None:
                return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        return Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    @staticmethod
    def _agent_state(env: Any) -> Dict[str, List[float]]:
        try:
            state = env.sim.get_agent_state()
            position = np.asarray(state.position, dtype=np.float32).tolist()
            rotation = np.asarray(state.rotation, dtype=np.float32).tolist()
            return {"position": position, "rotation": rotation}
        except Exception:
            return {"position": [], "rotation": []}

    def _step_record(
        self,
        observation_image: Image.Image,
        instruction: str,
        episode_key: str,
        step_index: int,
        env_step_index: Optional[int],
        rollout_index: int,
        step_in_rollout: int,
        action: int,
        progress: Optional[float],
        actor_done: bool,
        active_step: str,
        watcher_hint: str,
        watcher_subtask: str,
        env: Any,
    ) -> Dict[str, Any]:
        debug_snapshot = None
        if hasattr(self.nav_model, "get_last_debug_snapshot"):
            debug_snapshot = self.nav_model.get_last_debug_snapshot()
        state = self._agent_state(env)
        map_agent_coord = self._current_map_agent_coord(env)
        return {
            "step_index": int(step_index),
            "env_step_index": None if env_step_index is None else int(env_step_index),
            "instruction": instruction,
            "image_ref": f"{episode_key}:step:{int(step_index):06d}",
            "image": observation_image,
            "action_id": int(action),
            "action": action_id_to_name(action),
            "position": state["position"],
            "rotation": state["rotation"],
            "actor_progress": None if progress is None else float(progress),
            "actor_done": bool(actor_done),
            "active_plan_step": active_step,
            "watcher_hint": watcher_hint,
            "watcher_subtask": watcher_subtask,
            "map_agent_coord": map_agent_coord,
            "rollout_index": int(rollout_index),
            "step_in_rollout": int(step_in_rollout),
            "is_rollout_start": bool(step_in_rollout == 0),
            "is_rollout_end": False,
            "actor_prompt": None
            if not debug_snapshot
            else (debug_snapshot.get("prompt") or debug_snapshot.get("instruction_text")),
            "predicted_waypoint": None if not debug_snapshot else debug_snapshot.get("waypoint_first"),
            "actor_history_frame_indices": []
            if not debug_snapshot
            else list(debug_snapshot.get("history_frame_indices") or []),
            "actor_history_frame_count": 0
            if not debug_snapshot
            else int(debug_snapshot.get("history_frame_count") or 0),
            "actor_history_frame_target": None
            if not debug_snapshot
            else debug_snapshot.get("history_frame_target"),
            "actor_history_frame_count_ok": True
            if not debug_snapshot
            else bool(debug_snapshot.get("history_frame_count_ok", True)),
        }

    @staticmethod
    def _log_actor_decision(episode_key: str, step_record: Dict[str, Any]) -> None:
        logger.info(
            "[%s] actor decision=%d env_step=%s active_step=%s subtask=%s hint=%s output=%s",
            episode_key,
            int(step_record["step_index"]),
            "-" if step_record.get("env_step_index") is None else int(step_record["env_step_index"]),
            step_record.get("active_plan_step", ""),
            step_record.get("watcher_subtask", ""),
            step_record.get("watcher_hint", ""),
            json.dumps(
                {
                    "action": step_record.get("action"),
                    "progress": step_record.get("actor_progress"),
                    "done": step_record.get("actor_done"),
                    "predicted_waypoint": step_record.get("predicted_waypoint"),
                },
                ensure_ascii=False,
            ),
        )

    @staticmethod
    def _log_watcher_decision(
        episode_key: str,
        event_type: str,
        decision: WatcherDecision,
        todo_state: WatcherTodoState,
        rollout_slice: Sequence[Dict[str, Any]],
        previous_memory: str,
    ) -> None:
        logger.info(
            "[%s] watcher %s rollout=%s todo=%s previous_memory=%s output=%s",
            episode_key,
            event_type,
            json.dumps([item.get("action") for item in rollout_slice], ensure_ascii=False),
            json.dumps(todo_state.as_dict(), ensure_ascii=False),
            previous_memory,
            json.dumps(
                {
                    "done": decision.done,
                    "subtask": decision.subtask,
                    "memory": decision.memory,
                    "wakeup_reason": decision.wakeup_reason,
                    "raw_response": _json_safe(decision.raw_response),
                },
                ensure_ascii=False,
            ),
        )

    def _wakeup_reason(
        self,
        action: int,
        progress: Optional[float],
        actor_done: bool,
        fresh_actor_metadata: bool,
        env: Any,
        rollout_len: int,
    ) -> Optional[str]:
        del actor_done
        if int(action) == 0:
            return "stop_action"
        if fresh_actor_metadata and progress is not None and float(progress) >= self.progress_threshold:
            return "progress_threshold"
        if bool(getattr(env, "episode_over", False)):
            return "env_episode_over"
        if int(rollout_len) >= self.max_steps_per_wakeup:
            return "max_steps_per_wakeup"
        return None

    def _trace_watcher_event(
        self,
        decision: WatcherDecision,
        todo_state: WatcherTodoState,
        rollout_slice: Sequence[Dict[str, Any]],
        trace: Dict[str, Any],
        previous_memory: str,
        event_type: str,
    ) -> None:
        trace["watcher_events"].append(
            {
                "type": event_type,
                "memory": decision.memory,
                "done": decision.done,
                "subtask": decision.subtask,
                "raw_response": dict(decision.raw_response),
                "wakeup_reason": decision.wakeup_reason,
                "todo_state": todo_state.as_dict(),
                "previous_memory": previous_memory,
                "rollout_step_indices": [int(item["step_index"]) for item in rollout_slice],
            }
        )

    @staticmethod
    def _actor_subtask_signature(todo_state: WatcherTodoState, decision: WatcherDecision) -> tuple[str, str]:
        return (
            str(todo_state.active_step or "").strip(),
            str(decision.subtask or todo_state.active_step or "").strip(),
        )

    def _watcher_wakeup_cap_result(
        self,
        episode_key: str,
        trace: Dict[str, Any],
        rollout_slice: Sequence[Dict[str, Any]],
        wakeup_reason: str,
        done_steps: Sequence[str],
        steps_total: int,
        watcher_wakeups: int,
        final_memory: str,
        env: Any,
    ) -> Dict[str, Any]:
        trace["watcher_events"].append(
            {
                "type": "error",
                "stage": "watcher_wakeup_cap",
                "error": "watcher wakeup cap exceeded",
                "wakeup_reason": wakeup_reason,
                "rollout_step_indices": [int(item["step_index"]) for item in rollout_slice],
            }
        )
        return self._failure_result(
            episode_key=episode_key,
            trace=trace,
            done_steps=done_steps,
            steps_total=steps_total,
            watcher_wakeups=watcher_wakeups,
            final_memory=final_memory,
            env=env,
            failure_reason="watcher_wakeup_cap_exceeded",
            error=f"Watcher wakeups reached cap {self.max_watcher_wakeups}.",
        )

    @staticmethod
    def _current_top_down_map_info(env: Any) -> Optional[Dict[str, Any]]:
        if not hasattr(env, "get_metrics"):
            return None
        try:
            payload = dict(env.get_metrics()).get("top_down_map")
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        map_data = payload.get("map")
        if map_data is None:
            return None
        copied = {
            "map": np.array(map_data, copy=True),
        }
        fog = payload.get("fog_of_war_mask")
        if fog is not None:
            copied["fog_of_war_mask"] = np.array(fog, copy=True)
        if "agent_map_coord" in payload:
            copied["agent_map_coord"] = [list(item) for item in payload.get("agent_map_coord", [])]
        if "agent_angle" in payload:
            copied["agent_angle"] = [float(item) for item in payload.get("agent_angle", [])]
        return copied

    @staticmethod
    def _current_map_agent_coord(env: Any) -> List[int]:
        top_down_map = TwoSystemEpisodeRunner._current_top_down_map_info(env)
        if not isinstance(top_down_map, dict):
            return []
        coords = top_down_map.get("agent_map_coord") or []
        if not coords:
            return []
        coord = coords[0]
        if not isinstance(coord, (list, tuple)) or len(coord) < 2:
            return []
        return [int(coord[0]), int(coord[1])]

    @staticmethod
    def _reference_path_map_coords(env: Any) -> List[List[int]]:
        episode = getattr(env, "current_episode", None)
        top_down_map = TwoSystemEpisodeRunner._current_top_down_map_info(env)
        if episode is None or not isinstance(top_down_map, dict):
            return []
        reference_path = getattr(episode, "reference_path", None)
        map_data = top_down_map.get("map")
        if not isinstance(reference_path, list) or map_data is None:
            return []
        from habitat.utils.visualizations import maps as habitat_maps

        map_shape = np.asarray(map_data).shape[0:2]
        coords: List[List[int]] = []
        for point in reference_path:
            if not isinstance(point, (list, tuple)) or len(point) < 3:
                continue
            grid_x, grid_y = habitat_maps.to_grid(
                point[2],
                point[0],
                map_shape,
                sim=env.sim,
            )
            coords.append([int(grid_x), int(grid_y)])
        return coords

    @staticmethod
    def _current_metrics(env: Any) -> Dict[str, Any]:
        if hasattr(env, "get_metrics"):
            try:
                return dict(env.get_metrics())
            except Exception:
                return {}
        return {}

    @staticmethod
    def _current_nav_success(env: Any) -> bool:
        metrics = TwoSystemEpisodeRunner._current_metrics(env)
        try:
            return bool(float(metrics.get("success", 0.0)) > 0.0)
        except Exception:
            return False

    def _failure_result(
        self,
        episode_key: str,
        trace: Dict[str, Any],
        done_steps: Sequence[str],
        steps_total: int,
        watcher_wakeups: int,
        final_memory: str,
        env: Any,
        failure_reason: str,
        error: str,
    ) -> Dict[str, Any]:
        nav_success = self._current_nav_success(env)
        return {
            "episode_key": episode_key,
            "nav_success": nav_success,
            "watcher_complete": False,
            "failed": True,
            "failure_reason": failure_reason,
            "error": error,
            "steps_total": int(steps_total),
            "watcher_wakeups": int(watcher_wakeups),
            "done_steps": list(done_steps),
            "final_memory": final_memory,
            "trace": trace,
            "debug_top_down_map": self._current_top_down_map_info(env),
            "debug_reference_path_map_coords": self._reference_path_map_coords(env),
            "metrics": _sanitize_episode_metrics(self._current_metrics(env)),
        }

    def run_episode(
        self,
        env: Any,
        episode_key: str,
        instruction: str,
        plan_steps: Sequence[str],
    ) -> Dict[str, Any]:
        normalized_plan = parse_plan_steps(plan_steps)
        if not normalized_plan:
            normalized_plan = [instruction.strip() or "Navigate to the goal."]

        done_steps: List[str] = []
        active_step = normalized_plan[0]
        pending_steps = list(normalized_plan[1:])
        trace: Dict[str, Any] = {"steps": [], "watcher_events": []}

        if hasattr(self.nav_model, "eval"):
            self.nav_model.eval()
        if hasattr(self.nav_model, "reset_episode_state"):
            self.nav_model.reset_episode_state(episode_key=episode_key)

        first_observation = env.reset()
        first_image = self._observation_to_image(first_observation)
        try:
            init_decision = self.watcher_backend.initialize(
                instruction=instruction,
                plan_steps=normalized_plan,
                first_observation=first_image,
                episode_key=episode_key,
            )
        except Exception as exc:
            trace["watcher_events"].append(
                {
                    "type": "error",
                    "stage": "watcher_initialize",
                    "error": str(exc),
                    "rollout_step_indices": [],
                }
            )
            return self._failure_result(
                episode_key=episode_key,
                trace=trace,
                done_steps=done_steps,
                steps_total=0,
                watcher_wakeups=0,
                final_memory="",
                env=env,
                failure_reason="watcher_initialize_error",
                error=str(exc),
            )
        todo_state = WatcherTodoState(
            done_steps=list(done_steps),
            active_step=active_step,
            pending_steps=list(pending_steps),
        )
        self._trace_watcher_event(
            decision=init_decision,
            todo_state=todo_state,
            rollout_slice=[],
            trace=trace,
            previous_memory="",
            event_type="init",
        )
        self._log_watcher_decision(
            episode_key=episode_key,
            event_type="init",
            decision=init_decision,
            todo_state=todo_state,
            rollout_slice=[],
            previous_memory="",
        )

        current_observation = first_observation
        current_decision = init_decision
        watcher_wakeups = 0
        steps_total = 0
        decision_index = 0
        watcher_complete = False
        episode_over = bool(getattr(env, "episode_over", False))
        nav_success = self._current_nav_success(env)
        rollout_index = 0
        last_active_step_end_position: Optional[np.ndarray] = None
        actor_calls_total = 0
        actor_subtask_id = 1
        actor_subtask_signature = self._actor_subtask_signature(todo_state, current_decision)

        while steps_total < self.episode_step_cap and not watcher_complete and not episode_over and not nav_success:
            rollout_slice: List[Dict[str, Any]] = []
            while steps_total < self.episode_step_cap and not watcher_complete and not episode_over and not nav_success:
                pending_cached_actions = len(getattr(self.nav_model, "action_seq", []) or [])
                if actor_calls_total >= self.max_actor_calls_per_episode and pending_cached_actions <= 0:
                    trace["watcher_events"].append(
                        {
                            "type": "error",
                            "stage": "actor_call_cap",
                            "error": "actor call cap exceeded",
                            "rollout_step_indices": [int(item["step_index"]) for item in rollout_slice],
                        }
                    )
                    return self._failure_result(
                        episode_key=episode_key,
                        trace=trace,
                        done_steps=done_steps,
                        steps_total=steps_total,
                        watcher_wakeups=watcher_wakeups,
                        final_memory=current_decision.memory,
                        env=env,
                        failure_reason="actor_call_cap_exceeded",
                        error=f"Actor was called {actor_calls_total} times, reached cap {self.max_actor_calls_per_episode}.",
                    )
                observation_image = self._observation_to_image(current_observation)
                watcher_hint = current_decision.memory
                watcher_subtask = current_decision.subtask or todo_state.active_step
                action, progress, actor_done = self.nav_model.predict_action_with_progress_and_done(
                    observation=observation_image,
                    instruction=instruction,
                    subgoal=watcher_subtask,
                    episode_key=episode_key,
                    subtask_id=actor_subtask_id,
                    hint=watcher_hint or None,
                    forbidden_actions=self.forbidden_actions,
                )
                debug_snapshot = None
                if hasattr(self.nav_model, "get_last_debug_snapshot"):
                    debug_snapshot = self.nav_model.get_last_debug_snapshot()
                fresh_actor_metadata = bool(
                    True if debug_snapshot is None else debug_snapshot.get("fresh_actor_metadata", True)
                )
                if fresh_actor_metadata:
                    actor_calls_total += 1
                step_progress = float(progress) if fresh_actor_metadata else None
                step_actor_done = bool(actor_done) if fresh_actor_metadata else False

                if int(action) == 0:
                    step_record = self._step_record(
                        observation_image=observation_image,
                        instruction=instruction,
                        episode_key=episode_key,
                        step_index=decision_index,
                        env_step_index=None,
                        rollout_index=rollout_index,
                        step_in_rollout=len(rollout_slice),
                        action=int(action),
                        progress=step_progress,
                        actor_done=step_actor_done,
                        active_step=todo_state.active_step,
                        watcher_hint=watcher_hint,
                        watcher_subtask=watcher_subtask,
                        env=env,
                    )
                    step_record["is_rollout_end"] = True
                    rollout_slice.append(step_record)
                    trace["steps"].append(dict(step_record))
                    self._log_actor_decision(episode_key=episode_key, step_record=step_record)
                    decision_index += 1
                    previous_memory = current_decision.memory
                    if self.max_watcher_wakeups is not None and watcher_wakeups >= self.max_watcher_wakeups:
                        return self._watcher_wakeup_cap_result(
                            episode_key=episode_key,
                            trace=trace,
                            rollout_slice=rollout_slice,
                            wakeup_reason="stop_action",
                            done_steps=done_steps,
                            steps_total=steps_total,
                            watcher_wakeups=watcher_wakeups,
                            final_memory=previous_memory,
                            env=env,
                        )
                    try:
                        watcher_decision = self.watcher_backend.update(
                            instruction=instruction,
                            todo_state=todo_state,
                            memory_text=previous_memory,
                            rollout_slice=rollout_slice,
                            episode_key=episode_key,
                        )
                    except Exception as exc:
                        trace["watcher_events"].append(
                            {
                                "type": "error",
                                "stage": "watcher_update",
                                "error": str(exc),
                                "wakeup_reason": "stop_action",
                                "rollout_step_indices": [int(item["step_index"]) for item in rollout_slice],
                            }
                        )
                        return self._failure_result(
                            episode_key=episode_key,
                            trace=trace,
                            done_steps=done_steps,
                            steps_total=steps_total,
                            watcher_wakeups=watcher_wakeups,
                            final_memory=current_decision.memory,
                            env=env,
                            failure_reason="watcher_update_error",
                            error=str(exc),
                        )
                    if not watcher_decision.wakeup_reason:
                        watcher_decision = replace(watcher_decision, wakeup_reason="stop_action")
                    watcher_wakeups += 1

                    if watcher_decision.done:
                        done_steps.append(todo_state.active_step)
                        if pending_steps:
                            active_step = pending_steps.pop(0)
                        else:
                            watcher_complete = True
                        last_active_step_end_position = None
                    else:
                        active_step = todo_state.active_step

                    todo_state = WatcherTodoState(
                        done_steps=list(done_steps),
                        active_step=active_step if not watcher_complete else "",
                        pending_steps=list(pending_steps),
                    )
                    self._trace_watcher_event(
                        decision=watcher_decision,
                        todo_state=todo_state,
                        rollout_slice=rollout_slice,
                        trace=trace,
                        previous_memory=previous_memory,
                        event_type="update",
                    )
                    self._log_watcher_decision(
                        episode_key=episode_key,
                        event_type="update",
                        decision=watcher_decision,
                        todo_state=todo_state,
                        rollout_slice=rollout_slice,
                        previous_memory=previous_memory,
                    )
                    current_decision = watcher_decision
                    new_signature = self._actor_subtask_signature(todo_state, current_decision)
                    if new_signature != actor_subtask_signature:
                        actor_subtask_id += 1
                        actor_subtask_signature = new_signature
                    rollout_index += 1
                    break

                current_observation = env.step(int(action))
                episode_over = bool(getattr(env, "episode_over", False))
                step_record = self._step_record(
                    observation_image=observation_image,
                    instruction=instruction,
                    episode_key=episode_key,
                    step_index=decision_index,
                    env_step_index=steps_total,
                    rollout_index=rollout_index,
                    step_in_rollout=len(rollout_slice),
                    action=int(action),
                    progress=step_progress,
                    actor_done=step_actor_done,
                    active_step=todo_state.active_step,
                    watcher_hint=watcher_hint,
                    watcher_subtask=watcher_subtask,
                    env=env,
                )
                rollout_slice.append(step_record)
                trace["steps"].append(dict(step_record))
                self._log_actor_decision(episode_key=episode_key, step_record=step_record)
                decision_index += 1
                steps_total += 1
                nav_success = self._current_nav_success(env)
                if nav_success:
                    rollout_slice[-1]["is_rollout_end"] = True
                    break

                wakeup_reason = self._wakeup_reason(
                    action=int(action),
                    progress=step_progress,
                    actor_done=step_actor_done,
                    fresh_actor_metadata=fresh_actor_metadata,
                    env=env,
                    rollout_len=len(rollout_slice),
                )
                if wakeup_reason:
                    rollout_slice[-1]["is_rollout_end"] = True
                    previous_memory = current_decision.memory
                    if self.max_watcher_wakeups is not None and watcher_wakeups >= self.max_watcher_wakeups:
                        return self._watcher_wakeup_cap_result(
                            episode_key=episode_key,
                            trace=trace,
                            rollout_slice=rollout_slice,
                            wakeup_reason=wakeup_reason,
                            done_steps=done_steps,
                            steps_total=steps_total,
                            watcher_wakeups=watcher_wakeups,
                            final_memory=previous_memory,
                            env=env,
                        )
                    try:
                        watcher_decision = self.watcher_backend.update(
                            instruction=instruction,
                            todo_state=todo_state,
                            memory_text=previous_memory,
                            rollout_slice=rollout_slice,
                            episode_key=episode_key,
                        )
                    except Exception as exc:
                        trace["watcher_events"].append(
                            {
                                "type": "error",
                                "stage": "watcher_update",
                                "error": str(exc),
                                "wakeup_reason": wakeup_reason,
                                "rollout_step_indices": [int(item["step_index"]) for item in rollout_slice],
                            }
                        )
                        return self._failure_result(
                            episode_key=episode_key,
                            trace=trace,
                            done_steps=done_steps,
                            steps_total=steps_total,
                            watcher_wakeups=watcher_wakeups,
                            final_memory=current_decision.memory,
                            env=env,
                            failure_reason="watcher_update_error",
                            error=str(exc),
                        )
                    if not watcher_decision.wakeup_reason:
                        watcher_decision = replace(watcher_decision, wakeup_reason=wakeup_reason)
                    watcher_wakeups += 1
                    current_end_position = np.asarray(rollout_slice[-1].get("position") or [], dtype=np.float32)

                    if watcher_decision.done:
                        done_steps.append(todo_state.active_step)
                        if pending_steps:
                            active_step = pending_steps.pop(0)
                        else:
                            watcher_complete = True
                        last_active_step_end_position = None
                    else:
                        active_step = todo_state.active_step
                        if current_end_position.size and last_active_step_end_position is not None:
                            travel = float(np.linalg.norm(current_end_position - last_active_step_end_position))
                            if travel <= self.stagnation_distance_threshold:
                                trace["watcher_events"].append(
                                    {
                                        "type": "error",
                                        "stage": "stagnation",
                                        "error": "stagnation detected",
                                        "wakeup_reason": watcher_decision.wakeup_reason,
                                        "rollout_step_indices": [int(item["step_index"]) for item in rollout_slice],
                                    }
                                )
                                return self._failure_result(
                                    episode_key=episode_key,
                                    trace=trace,
                                    done_steps=done_steps,
                                    steps_total=steps_total,
                                    watcher_wakeups=watcher_wakeups,
                                    final_memory=watcher_decision.memory,
                                    env=env,
                                    failure_reason="stagnation",
                                    error="Agent position did not change across watcher wakeups for the same subtask.",
                                )
                        if current_end_position.size:
                            last_active_step_end_position = current_end_position

                    todo_state = WatcherTodoState(
                        done_steps=list(done_steps),
                        active_step=active_step if not watcher_complete else "",
                        pending_steps=list(pending_steps),
                    )
                    self._trace_watcher_event(
                        decision=watcher_decision,
                        todo_state=todo_state,
                        rollout_slice=rollout_slice,
                        trace=trace,
                        previous_memory=previous_memory,
                        event_type="update",
                    )
                    self._log_watcher_decision(
                        episode_key=episode_key,
                        event_type="update",
                        decision=watcher_decision,
                        todo_state=todo_state,
                        rollout_slice=rollout_slice,
                        previous_memory=previous_memory,
                    )
                    current_decision = watcher_decision
                    new_signature = self._actor_subtask_signature(todo_state, current_decision)
                    if new_signature != actor_subtask_signature:
                        actor_subtask_id += 1
                        actor_subtask_signature = new_signature
                    rollout_index += 1
                    break

            if watcher_complete or nav_success:
                break

        final_metrics = _sanitize_episode_metrics(self._current_metrics(env))
        return {
            "episode_key": episode_key,
            "nav_success": bool(float(final_metrics.get("success", 0.0)) > 0.0),
            "watcher_complete": watcher_complete,
            "failed": False,
            "failure_reason": "",
            "error": "",
            "steps_total": int(steps_total),
            "watcher_wakeups": int(watcher_wakeups),
            "done_steps": list(done_steps),
            "final_memory": current_decision.memory,
            "trace": trace,
            "debug_top_down_map": self._current_top_down_map_info(env),
            "debug_reference_path_map_coords": self._reference_path_map_coords(env),
            "metrics": final_metrics,
        }


def _build_actor_args(config: Dict[str, Any], device: str) -> SimpleNamespace:
    actor_cfg = config["actor"]
    return SimpleNamespace(
        model_type=actor_cfg["model_type"],
        model_path=actor_cfg["model_path"],
        base_model_path=actor_cfg.get("base_model_path"),
        memory_num_history_images=actor_cfg["memory_num_history_images"],
        done_threshold=actor_cfg["done_threshold"],
        use_memory=False,
        device=device,
        model_max_length=int(actor_cfg.get("model_max_length", 4096)),
        num_frames=int(actor_cfg.get("num_frames", 32)),
        num_future_steps=int(actor_cfg.get("num_future_steps", 4)),
        num_history=int(actor_cfg.get("num_history", 8)),
    )


def _build_watcher_backend(config: Dict[str, Any], device: str):
    watcher_cfg = config["watcher"]
    backend = str(watcher_cfg["backend"]).strip().lower()
    watcher_device = str(watcher_cfg.get("device", device) or device)
    if backend == "api":
        return ApiWatcherBackend(
            model_name=watcher_cfg["model_name"],
            api_base_url=watcher_cfg["api_base_url"],
            image_stride=watcher_cfg["image_stride"],
            api_key_env=watcher_cfg["api_key_env"],
            reasoning_effort=watcher_cfg["reasoning_effort"],
            request_timeout=watcher_cfg["request_timeout"],
            max_retries=watcher_cfg["max_retries"],
        )
    if backend == "local":
        return LocalWatcherBackend(
            model_path=watcher_cfg["model_path"],
            base_model_path=watcher_cfg["base_model_path"],
            device=watcher_device,
            image_stride=watcher_cfg["image_stride"],
        )
    raise ValueError(f"Unsupported watcher backend: {backend}")


def _write_debug_html(path: Path, episode_result: Dict[str, Any]) -> None:
    step_rows = []
    for step in episode_result["trace"].get("steps", []):
        actor_progress = step.get("actor_progress")
        actor_progress_text = "" if actor_progress is None else f"{float(actor_progress):.3f}"
        step_rows.append(
            "<tr>"
            f"<td>{int(step.get('step_index', 0))}</td>"
            f"<td>{'' if step.get('env_step_index') is None else int(step['env_step_index'])}</td>"
            f"<td>{html_lib.escape(str(step.get('action', '')))}</td>"
            f"<td>{actor_progress_text}</td>"
            f"<td>{html_lib.escape(str(step.get('actor_done', False)))}</td>"
            f"<td><pre>{html_lib.escape(str(step.get('actor_prompt', '')))}</pre></td>"
            "</tr>"
        )
    watcher_rows = []
    for event in episode_result["trace"].get("watcher_events", []):
        watcher_rows.append(
            "<tr>"
            f"<td>{html_lib.escape(str(event.get('type', '')))}</td>"
            f"<td>{html_lib.escape(str(event.get('wakeup_reason', '')))}</td>"
            f"<td>{html_lib.escape(str(event.get('done', '')))}</td>"
            f"<td>{html_lib.escape(str(event.get('subtask', '')))}</td>"
            f"<td><pre>{html_lib.escape(str(event.get('previous_memory', '')))}</pre></td>"
            f"<td><pre>{html_lib.escape(str(event.get('memory', '')))}</pre></td>"
            f"<td><pre>{html_lib.escape(json.dumps(_json_safe(event.get('raw_response', {})), ensure_ascii=False, indent=2))}</pre></td>"
            f"<td>{html_lib.escape(str(event.get('rollout_step_indices', [])))}</td>"
            "</tr>"
        )
    html = (
        "<html><body>"
        f"<h1>{html_lib.escape(str(episode_result['episode_key']))}</h1>"
        "<h2>Actor Steps</h2>"
        "<table border='1'>"
        "<tr><th>Decision</th><th>Env Step</th><th>Action</th><th>Progress</th><th>Done</th><th>Actor Prompt</th></tr>"
        f"{''.join(step_rows)}"
        "</table>"
        "<h2>Watcher Events</h2>"
        "<table border='1'>"
        "<tr><th>Type</th><th>Wakeup</th><th>Done</th><th>Subtask</th><th>Previous Memory</th><th>Memory</th><th>Raw Response</th><th>Rollout Steps</th></tr>"
        f"{''.join(watcher_rows)}"
        "</table>"
        "</body></html>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _event_by_last_step(trace: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for event in trace.get("watcher_events", []):
        step_indices = [int(idx) for idx in event.get("rollout_step_indices", [])]
        if step_indices:
            lookup[step_indices[-1]] = dict(event)
    return lookup


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_chars: int = 48,
    line_height: int = 14,
) -> int:
    x, y = xy
    for paragraph in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(paragraph, width=max_chars) or [""]
        for line in wrapped:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
    return y


def _build_debug_video_sections(
    episode_key: str,
    step: Dict[str, Any],
    watcher_event: Optional[Dict[str, Any]] = None,
) -> List[tuple[str, List[tuple[str, Any]]]]:
    actor_progress = step.get("actor_progress")
    actor_progress_text = "" if actor_progress is None else f"{float(actor_progress):.3f}"
    sections: List[tuple[str, List[tuple[str, Any]]]] = [
        (
            "Episode",
            [
                ("Episode", episode_key),
                ("Decision", step.get("step_index", "")),
                ("Env Step", step.get("env_step_index", "-")),
            ],
        ),
        (
            "Actor Input",
            [
                ("Prompt", step.get("actor_prompt", "")),
            ],
        ),
        (
            "Actor Output",
            [
                ("Action", step.get("action", "")),
                ("Progress", actor_progress_text),
                ("Done", step.get("actor_done", False)),
            ],
        ),
    ]
    if watcher_event is not None:
        sections.append(
            (
                "Watcher Update",
                [
                    ("Wakeup", watcher_event.get("wakeup_reason", "")),
                    ("Done", watcher_event.get("done", "")),
                    ("Subtask", watcher_event.get("subtask", "")),
                    ("Memory", watcher_event.get("memory", "")),
                ],
            )
        )
    return sections


def _estimate_debug_video_height(
    sections: Sequence[tuple[str, Sequence[tuple[str, Any]]]],
    max_chars: int = 44,
    line_height: int = 14,
) -> int:
    height = 12
    for idx, (_, fields) in enumerate(sections):
        if idx > 0:
            height += 6
        height += line_height + 2
        for label, value in fields:
            text = f"{label}: {value}"
            lines = 0
            for paragraph in str(text).splitlines() or [""]:
                lines += max(1, len(textwrap.wrap(paragraph, width=max_chars)))
            height += lines * line_height + 2
    return height + 12


def _render_debug_video_frame(
    episode_key: str,
    step: Dict[str, Any],
    watcher_event: Optional[Dict[str, Any]] = None,
    top_down_map_panel: Optional[Image.Image] = None,
) -> Image.Image:
    base_image = step.get("image")
    if not isinstance(base_image, Image.Image):
        raise ValueError("Debug video frame requires PIL image in step record.")
    rgb = base_image.convert("RGB")
    panel_width = 480
    sections = _build_debug_video_sections(episode_key, step, watcher_event)
    canvas_height = max(int(rgb.height), _estimate_debug_video_height(sections), 360)
    map_width = int(top_down_map_panel.width) if isinstance(top_down_map_panel, Image.Image) else 0
    canvas = Image.new("RGB", (int(rgb.width) + map_width + panel_width, canvas_height), color=(248, 248, 248))
    canvas.paste(rgb, (0, 0))
    if isinstance(top_down_map_panel, Image.Image):
        canvas.paste(top_down_map_panel, (int(rgb.width), 0))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    panel_x0 = int(rgb.width) + map_width
    draw.rectangle((panel_x0, 0, canvas.width, canvas.height), fill=(250, 250, 250))
    draw.line((panel_x0, 0, panel_x0, canvas.height), fill=(180, 180, 180), width=1)

    y = 12
    x = panel_x0 + 12
    line_height = 14

    def _section(title: str) -> None:
        nonlocal y
        draw.text((x, y), title, font=font, fill=(20, 20, 20))
        y += line_height + 2

    def _field(label: str, value: Any, max_chars: int = 48) -> None:
        nonlocal y
        y = _draw_wrapped_text(
            draw,
            f"{label}: {value}",
            (x, y),
            font=font,
            fill=(40, 40, 40),
            max_chars=max_chars,
            line_height=line_height,
        )
        y += 2

    for idx, (title, fields) in enumerate(sections):
        if idx > 0:
            y += 6
        _section(title)
        for label, value in fields:
            _field(label, value, max_chars=44)
    return canvas


def _pad_debug_video_frame(frame: Image.Image, width: int, height: int) -> Image.Image:
    if frame.size == (int(width), int(height)):
        return frame
    padded = Image.new("RGB", (int(width), int(height)), color=(248, 248, 248))
    padded.paste(frame, (0, 0))
    return padded


def _draw_debug_map_path(
    draw: ImageDraw.ImageDraw,
    coords: Sequence[Sequence[int]],
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    points = [
        (int(coord[1]), int(coord[0]))
        for coord in coords
        if isinstance(coord, (list, tuple)) and len(coord) >= 2
    ]
    if len(points) >= 2:
        draw.line(points, fill=color, width=int(width))
    for x, y in points:
        radius = max(2, int(width))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _render_debug_top_down_map_panel(
    episode_result: Dict[str, Any],
    step_index: int,
    output_height: int,
) -> Optional[Image.Image]:
    top_down_map = episode_result.get("debug_top_down_map")
    if not isinstance(top_down_map, dict):
        return None
    raw_map = top_down_map.get("map")
    if not isinstance(raw_map, np.ndarray):
        return None
    from habitat.utils.visualizations import maps as habitat_maps

    fog_mask = top_down_map.get("fog_of_war_mask")
    map_rgb = habitat_maps.colorize_topdown_map(
        raw_map,
        fog_mask if isinstance(fog_mask, np.ndarray) else None,
    )
    map_image = Image.fromarray(np.asarray(map_rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(map_image)
    _draw_debug_map_path(
        draw,
        episode_result.get("debug_reference_path_map_coords", []),
        color=(72, 176, 88),
        width=3,
    )
    agent_coords = [
        step.get("map_agent_coord", [])
        for step in episode_result.get("trace", {}).get("steps", [])
        if int(step.get("step_index", -1)) <= int(step_index)
    ]
    _draw_debug_map_path(draw, agent_coords, color=(220, 76, 70), width=4)

    if map_image.height > map_image.width:
        map_image = map_image.transpose(Image.Transpose.ROTATE_90)
    target_height = max(1, int(output_height))
    target_width = max(1, int(round(map_image.width * (target_height / max(map_image.height, 1)))))
    return map_image.resize((target_width, target_height), Image.Resampling.BICUBIC)


def _write_debug_video(path: Path, episode_result: Dict[str, Any], fps: int = 2) -> None:
    import cv2

    trace = episode_result.get("trace", {})
    watcher_event_lookup = _event_by_last_step(trace)
    steps = [
        step
        for step in trace.get("steps", [])
        if step.get("env_step_index") is not None and isinstance(step.get("image"), Image.Image)
    ]
    if not steps:
        return

    rendered_frames = [
        _render_debug_video_frame(
            episode_key=str(episode_result.get("episode_key", "")),
            step=step,
            watcher_event=watcher_event_lookup.get(int(step.get("step_index", -1))),
            top_down_map_panel=_render_debug_top_down_map_panel(
                episode_result,
                step_index=int(step.get("step_index", -1)),
                output_height=int(step.get("image").height) if isinstance(step.get("image"), Image.Image) else 360,
            ),
        )
        for step in steps
    ]
    width = max(int(frame.width) for frame in rendered_frames)
    height = max(int(frame.height) for frame in rendered_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, int(fps))),
        (int(width), int(height)),
    )
    is_opened = getattr(writer, "isOpened", None)
    if callable(is_opened) and not is_opened():
        raise RuntimeError(f"Failed to open debug video writer: {path}")
    try:
        for frame in rendered_frames:
            padded = _pad_debug_video_frame(frame, width=width, height=height)
            frame_bgr = cv2.cvtColor(np.asarray(padded, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def evaluate(config: Dict[str, Any]) -> Dict[str, Any]:
    from thinkvln.eval.close_eval_dist import all_reduce_scalar_dict, init_dist_mode
    from thinkvln.eval.close_eval_models import build_nav_model
    from thinkvln.eval.close_eval_runner import VLNEvaluator
    import torch

    runtime_cfg = config["runtime"]
    env_cfg = config["env"]
    output_cfg = config["output"]

    rank, world_size, gpu = init_dist_mode(timeout_minutes=runtime_cfg["dist_timeout_minutes"])
    device = f"cuda:{gpu}" if world_size > 1 else str(config["actor"]["device"])
    actor_args = _build_actor_args(config, device=device)
    nav_model = build_nav_model(actor_args, device, rank, world_size)
    watcher_backend = _build_watcher_backend(config, device=device)
    ensure_loaded = getattr(watcher_backend, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()
    summary_full = load_summary_full(env_cfg["summary_full_path"])

    output_dir = Path(output_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator_args = SimpleNamespace(
        device=device,
        sample_rate=float(env_cfg["sample_rate"]),
        target_episode_key=str(env_cfg["target_episode_key"] or ""),
    )
    evaluator = VLNEvaluator(
        config_path=env_cfg["habitat_config_path"],
        split=env_cfg["eval_split"],
        env_num=world_size,
        output_path=str(output_dir),
        nav_model=nav_model,
        epoch=0,
        args=evaluator_args,
    )
    env = evaluator.config_env()
    runner = TwoSystemEpisodeRunner(
        nav_model=nav_model,
        watcher_backend=watcher_backend,
        max_steps_per_wakeup=config["rollout"]["max_steps_per_wakeup"],
        progress_threshold=config["rollout"]["progress_threshold"],
        episode_step_cap=config["rollout"]["episode_step_cap"],
        max_watcher_wakeups=config["rollout"]["max_watcher_wakeups"],
        forbidden_actions=config["rollout"]["forbidden_actions"],
    )

    local_stats = {
        "episodes_total": 0.0,
        "episodes_evaluated": 0.0,
        "episodes_missing_meta": 0.0,
        "watcher_complete": 0.0,
        "steps_total": 0.0,
        "watcher_wakeups": 0.0,
        "success_sum": 0.0,
    }

    trace_handle = None
    if output_cfg["save_trace_jsonl"]:
        trace_path = output_dir / f"rank_{rank:02d}_episodes.jsonl"
        trace_handle = open(trace_path, "a", encoding="utf-8")

    try:
        for scene_id, episode in evaluator._iter_assigned_episodes(env, rank):
            local_stats["episodes_total"] += 1.0
            episode_key = f"{scene_id}_{episode.episode_id}"
            record = summary_full.get(episode_key)
            if not record:
                local_stats["episodes_missing_meta"] += 1.0
                continue

            local_stats["episodes_evaluated"] += 1.0
            env.current_episode = episode
            instruction = evaluator._episode_instruction(env_cfg["habitat_config_path"], episode)
            plan_steps = parse_plan_steps(record.get("plan") or record.get("subtasks") or record.get("subtask_text"))
            episode_result = runner.run_episode(
                env=env,
                episode_key=episode_key,
                instruction=instruction,
                plan_steps=plan_steps,
            )
            local_stats["watcher_complete"] += float(episode_result["watcher_complete"])
            local_stats["steps_total"] += float(episode_result["steps_total"])
            local_stats["watcher_wakeups"] += float(episode_result["watcher_wakeups"])
            local_stats["success_sum"] += float(episode_result["metrics"].get("success", 0.0))

            if trace_handle is not None:
                payload = _json_safe({
                    "episode_key": episode_result["episode_key"],
                    "nav_success": _episode_nav_success(episode_result),
                    "watcher_complete": episode_result["watcher_complete"],
                    "steps_total": episode_result["steps_total"],
                    "watcher_wakeups": episode_result["watcher_wakeups"],
                    "done_steps": episode_result["done_steps"],
                    "metrics": _sanitize_episode_metrics(episode_result["metrics"]),
                    "trace": _trace_without_images(episode_result["trace"]),
                })
                write_jsonl_record(trace_handle, payload, sync_to_disk=False)

            if output_cfg["save_step_debug_html"]:
                _write_debug_html(output_dir / "debug" / f"{episode_key}.html", episode_result)
            if output_cfg["save_debug_video"]:
                _write_debug_video(
                    output_dir / "debug_video" / f"{episode_key}.mp4",
                    episode_result,
                    fps=int(output_cfg["debug_video_fps"]),
                )
    finally:
        if trace_handle is not None:
            trace_handle.close()

    reduced = all_reduce_scalar_dict(local_stats, torch.device(device))
    episodes_eval = max(1.0, float(reduced.get("episodes_evaluated", 0.0)))
    summary = {
        "episodes_total": int(reduced.get("episodes_total", 0.0)),
        "episodes_evaluated": int(reduced.get("episodes_evaluated", 0.0)),
        "episodes_missing_meta": int(reduced.get("episodes_missing_meta", 0.0)),
        "nav_success_rate": float(reduced.get("success_sum", 0.0)) / episodes_eval,
        "watcher_complete_rate": float(reduced.get("watcher_complete", 0.0)) / episodes_eval,
        "avg_steps_total": float(reduced.get("steps_total", 0.0)) / episodes_eval,
        "avg_watcher_wakeups": float(reduced.get("watcher_wakeups", 0.0)) / episodes_eval,
        "avg_success": float(reduced.get("success_sum", 0.0)) / episodes_eval,
    }

    if output_cfg["save_summary_json"] and rank == 0:
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_debug(config: Dict[str, Any]) -> Dict[str, Any]:
    from thinkvln.eval.close_eval_dist import init_dist_mode
    from thinkvln.eval.close_eval_models import build_nav_model
    from thinkvln.eval.close_eval_runner import VLNEvaluator

    if not bool(config["debug"]["enabled"]):
        raise ValueError("evaluate_debug requires debug.enabled=true")

    runtime_cfg = config["runtime"]
    env_cfg = config["env"]
    debug_cfg = config["debug"]
    episode_key = _debug_episode_key(config)
    sample_limit = _debug_sample_limit(config)
    single_episode = bool(episode_key)

    rank, world_size, gpu = init_dist_mode(timeout_minutes=runtime_cfg["dist_timeout_minutes"])
    if int(world_size) != 1:
        raise ValueError("debug mode requires world_size=1")

    device = f"cuda:{gpu}" if world_size > 1 else str(config["actor"]["device"])
    actor_args = _build_actor_args(config, device=device)
    nav_model = build_nav_model(actor_args, device, rank, world_size)
    watcher_backend = _build_watcher_backend(config, device=device)
    ensure_loaded = getattr(watcher_backend, "_ensure_loaded", None)
    if callable(ensure_loaded):
        ensure_loaded()

    summary_full = load_summary_full(env_cfg["summary_full_path"])
    output_dir = _debug_output_dir(config, episode_key=episode_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator_args = SimpleNamespace(
        device=device,
        sample_rate=float(env_cfg["sample_rate"]),
        target_episode_key=episode_key if single_episode else "",
    )
    evaluator = VLNEvaluator(
        config_path=env_cfg["habitat_config_path"],
        split=env_cfg["eval_split"],
        env_num=world_size,
        output_path=str(output_dir),
        nav_model=nav_model,
        epoch=0,
        args=evaluator_args,
    )
    env = evaluator.config_env()
    runner = TwoSystemEpisodeRunner(
        nav_model=nav_model,
        watcher_backend=watcher_backend,
        max_steps_per_wakeup=config["rollout"]["max_steps_per_wakeup"],
        progress_threshold=config["rollout"]["progress_threshold"],
        episode_step_cap=config["rollout"]["episode_step_cap"],
        max_watcher_wakeups=config["rollout"]["max_watcher_wakeups"],
        forbidden_actions=config["rollout"]["forbidden_actions"],
    )

    if single_episode:
        episode_result = None
        instruction = ""
        plan_steps: List[str] = []
        for scene_id, episode in evaluator._iter_assigned_episodes(env, rank):
            current_episode_key = f"{scene_id}_{episode.episode_id}"
            if current_episode_key != episode_key:
                continue
            record = summary_full.get(current_episode_key)
            if not record:
                raise ValueError(f"Episode metadata not found for {current_episode_key}")
            instruction = evaluator._episode_instruction(env_cfg["habitat_config_path"], episode)
            plan_steps = parse_plan_steps(record.get("plan") or record.get("subtasks") or record.get("subtask_text"))
            env.current_episode = episode
            episode_result = runner.run_episode(
                env=env,
                episode_key=current_episode_key,
                instruction=instruction,
                plan_steps=plan_steps,
            )
            break

        if episode_result is None:
            raise ValueError(f"Target episode not found: {episode_key}")

        episode_payload = _json_safe(
            _debug_episode_payload(
                config=config,
                episode_result=episode_result,
                instruction=instruction,
                plan_steps=plan_steps,
                output_dir=output_dir,
            )
        )
        summary = _json_safe(_debug_summary(episode_result))
        _write_debug_episode_artifacts(
            output_dir=output_dir,
            episode_result=episode_result,
            episode_payload=episode_payload,
            summary_payload=summary,
            fps=int(debug_cfg["video_fps"]),
            single_episode=True,
        )
        return summary

    episode_summaries: List[Dict[str, Any]] = []
    candidates: List[tuple[str, Any, Dict[str, Any]]] = []
    for scene_id, episode in evaluator._iter_assigned_episodes(env, rank):
        current_episode_key = f"{scene_id}_{episode.episode_id}"
        record = summary_full.get(current_episode_key)
        if not record:
            continue
        candidates.append((current_episode_key, episode, record))

    for current_episode_key, episode, record in _sample_debug_candidates(
        candidates,
        sample_limit=sample_limit,
        sample_seed=_debug_sample_seed(config),
    ):
        instruction = evaluator._episode_instruction(env_cfg["habitat_config_path"], episode)
        plan_steps = parse_plan_steps(record.get("plan") or record.get("subtasks") or record.get("subtask_text"))
        env.current_episode = episode
        episode_result = runner.run_episode(
            env=env,
            episode_key=current_episode_key,
            instruction=instruction,
            plan_steps=plan_steps,
        )
        episode_payload = _json_safe(
            _debug_episode_payload(
                config=config,
                episode_result=episode_result,
                instruction=instruction,
                plan_steps=plan_steps,
                output_dir=output_dir,
            )
        )
        summary_payload = _json_safe(_debug_summary(episode_result))
        _write_debug_episode_artifacts(
            output_dir=output_dir,
            episode_result=episode_result,
            episode_payload=episode_payload,
            summary_payload=summary_payload,
            fps=int(debug_cfg["video_fps"]),
            single_episode=False,
        )
        episode_summaries.append(
            _json_safe(
                {
                    "episode_key": episode_result["episode_key"],
                    "summary": summary_payload,
                }
            )
        )

    if not episode_summaries:
        raise ValueError("No debug episodes were evaluated.")

    episodes_manifest = output_dir / "episodes.jsonl"
    with open(episodes_manifest, "w", encoding="utf-8") as handle:
        for payload in episode_summaries:
            write_jsonl_record(handle, payload, sync_to_disk=False)

    summary = _json_safe(_debug_batch_summary([item["summary"] for item in episode_summaries]))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    _setup_logging(config["runtime"]["log_level"])
    logger.info("Loaded config: %s", args.config)
    if bool(config["debug"]["enabled"]):
        return evaluate_debug(config)
    return evaluate(config)


if __name__ == "__main__":
    main()
