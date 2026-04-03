from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from thinkvln.datagen.generation.watcher_utils import load_jsonl, load_jsonl_by_key, resolve_bundle_image_path
from thinkvln.tools.dataset_utils import load_image


ROLLOUT_SYSTEM_PROMPT = """You are the watcher updating navigation memory and deciding if the current subtask is complete.

You MUST return JSON only, and strictly in this EXACT order:
{"memory_end":"...","done":true/false,"next_subtask":"..."}

# Thinking Guidelines (For your internal reasoning before generating JSON)
1. Identify the robot's current 'Active' step from the plan.
2. Look at the FINAL rollout frame: Has the specific visual or physical goal of this active step been achieved? (e.g., if the step is "enter kitchen", is it clearly inside the kitchen?)
3. Is the robot fully aligned and ready to start the "Pending" step, or is it still adjusting?

# Output Rules

Step 1: memory_end
- Must be exactly three short semicolon-separated fragments: [traj summary]; [current physical state]; [neutral status].
- Update the memory_start with the new progress.
- Keep it concise and state exactly where the robot is in the final frame.

Step 2: done
- Set to true ONLY IF your internal reasoning confirms the active step's goal is fully reached AND the robot is in a stable position to begin the next step.
- Set to false if the robot is still moving toward the goal, still turning, halfway through a door, or recovering from a mistake.
- NEVER set to true just because the robot is "close" to the goal.

Step 3: next_subtask
- If done=false: Write an imperative command to continue or finish the current active step (e.g., "finish turning left").
- If done=true: Write the imperative command for the NEW active step (promoted from pending), or "stop" if no steps remain.

Examples:
{"memory_end":"Reached the hallway entrance; mid-turn facing the wall; step ongoing","done":false,"next_subtask":"finish turning right to face down the hallway"}

{"memory_end":"Cleared the dining area and entered bathroom; standing inside facing the sink; ready for next step","done":true,"next_subtask":"approach the sink"}
"""


@dataclass
class WatcherRolloutPrompt:
    system_prompt: str
    user_text: str
    images: List[Image.Image]


def normalize_plan_steps(plan: Any) -> List[str]:
    if isinstance(plan, list):
        return [str(step).strip() for step in plan if str(step).strip()]
    if isinstance(plan, str):
        return [line.strip() for line in plan.splitlines() if line.strip()]
    return []


def build_episode_key(scene_id: str, episode_id: Any) -> str:
    return f"{scene_id}_{episode_id}"


def extract_scene_id(scene_id_or_path: str) -> str:
    parts = str(scene_id_or_path or "").split("/")
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1] if parts else ""


def load_summary_full(path: str | Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(Path(path)):
        episode_key = str(row.get("episode_key", "")).strip()
        if episode_key:
            records[episode_key] = row
        scene_id = row.get("scene_id")
        episode_id = row.get("episode_id", row.get("id"))
        if scene_id is not None and episode_id is not None:
            records[build_episode_key(extract_scene_id(str(scene_id)), episode_id)] = row
    return records


def split_plan_state(subtask_id: Any, plan_steps: Sequence[str]) -> tuple[List[str], str, List[str]]:
    if not plan_steps:
        return [], "", []
    index = min(max(int(subtask_id or 1) - 1, 0), len(plan_steps) - 1)
    return list(plan_steps[:index]), str(plan_steps[index]), list(plan_steps[index + 1 :])


def format_plan_section(steps: Sequence[str]) -> str:
    if not steps:
        return "- None."
    return "\n".join(f"- {step}" for step in steps)


def select_rollout_paths(paths: Sequence[str], stride: int) -> List[str]:
    items = [str(path) for path in paths]
    if not items:
        return []
    step = max(1, int(stride))
    selected = items[::step]
    if selected[-1] != items[-1]:
        selected.append(items[-1])
    return selected


def format_watcher_update_json(memory_end: str, done: bool, next_subtask: str) -> str:
    payload = {
        "memory_end": str(memory_end),
        "done": bool(done),
        "next_subtask": str(next_subtask),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_rollout_prompt(
    instruction: str,
    plan_steps: Sequence[str],
    done_steps: Sequence[str],
    active_step: str,
    pending_steps: Sequence[str],
    memory_start: str,
    rollout_actions: Sequence[str],
    rollout_images: Sequence[Image.Image],
) -> WatcherRolloutPrompt:
    del plan_steps
    user_text = (
        "Plan state\n"
        "Done\n"
        f"{format_plan_section(done_steps)}\n"
        "Active\n"
        f"{format_plan_section([active_step] if active_step else [])}\n"
        "Pending\n"
        f"{format_plan_section(pending_steps)}\n"
        "State\n"
        f"- Memory start: {memory_start}\n"
        f"- Rollout actions: {list(rollout_actions)}\n"
        "Decision\n"
        "- Set done=true only if the active step reaches a natural handoff by rollout end.\n"
        "- The rollout end must also be a good starting point for the next subtask.\n"
        "- Do not hand off early just because the active step looks mostly complete.\n"
        "- If the next pending step is a turn, judge whether the robot has actually reached the turning point.\n"
        "- If the active step is a turn, judge it together with the next pending step and only hand off once the robot is aligned for that next movement.\n"
        "- If the active step is still the right step, set done=false.\n"
        "- With done=false, next_subtask should continue the active step or give a short recovery step.\n"
        "- With done=true, next_subtask should describe the promoted pending step, or stop if pending is empty.\n"
        "Memory\n"
        "- Rewrite memory_start into a new cumulative watcher memory.\n"
        "- Keep only the still-relevant part of memory_start.\n"
        "- Write memory_end as a direct update of memory_start.\n"
        "- Prefer extending memory_start forward with the new observation and then compressing if needed.\n"
        "- Do not reduce memory_end to only the final frame.\n"
        "- Write memory_end as exactly three short semicolon-separated fragments.\n"
        "- Use this exact order: traj summary; current state; task status.\n"
        "- The first fragment must summarize the path already traveled before the final state.\n"
        "- The third fragment must say whether the step is ongoing, ready for next step, or task complete.\n"
        "- Sentence fragments are allowed."
    )
    return WatcherRolloutPrompt(
        system_prompt=ROLLOUT_SYSTEM_PROMPT,
        user_text=user_text,
        images=list(rollout_images),
    )


class WatcherSFTDataset(Dataset):
    def __init__(
        self,
        manifest_file: str,
        annotation_file: str,
        bundle_root: str,
        summary_full_path: str,
        image_stride: int = 1,
        sample_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.manifest_file = Path(manifest_file)
        self.annotation_file = Path(annotation_file)
        self.bundle_root = Path(bundle_root)
        self.summary_full_path = Path(summary_full_path)
        self.image_stride = max(1, int(image_stride))
        self.sample_ratio = float(sample_ratio)
        self.seed = int(seed)
        self.samples = self._load_samples()
        if not self.samples:
            raise ValueError("WatcherSFTDataset loaded no valid samples")

    def _load_samples(self) -> List[Dict[str, Any]]:
        manifest_rows = load_jsonl(self.manifest_file)
        annotations = load_jsonl_by_key(self.annotation_file, "sample_id")
        summary_lookup = load_summary_full(self.summary_full_path)
        samples: List[Dict[str, Any]] = []

        for row in manifest_rows:
            sample_id = str(row.get("sample_id", "")).strip()
            annotation = annotations.get(sample_id)
            if not sample_id or annotation is None:
                continue
            summary_row = summary_lookup.get(str(row.get("episode_key", "")).strip(), {})
            plan_steps = normalize_plan_steps(row.get("plan") or summary_row.get("plan"))
            if not plan_steps:
                plan_steps = normalize_plan_steps([row.get("subtask_text", "")])
            instruction = str(row.get("instruction") or summary_row.get("instruction") or "").strip()
            done_steps, active_step, pending_steps = split_plan_state(row.get("subtask_id"), plan_steps)
            image_paths = [
                str(resolve_bundle_image_path(self.bundle_root, str(row.get("base_image_path", "")), relpath))
                for relpath in select_rollout_paths(row.get("rollout_image_relpaths", []), self.image_stride)
            ]
            samples.append(
                {
                    "sample_id": sample_id,
                    "episode_key": str(row.get("episode_key", "")).strip(),
                    "instruction": instruction,
                    "plan_steps": plan_steps,
                    "done_steps": done_steps,
                    "active_step": active_step,
                    "pending_steps": pending_steps,
                    "memory_start": str(annotation.get("memory_start", "")).strip(),
                    "rollout_actions": list(row.get("actions", [])),
                    "memory_end": str(annotation.get("memory_end", "")).strip(),
                    "done": bool(annotation.get("done", False)),
                    "next_subtask": str(annotation.get("next_subtask", "")).strip(),
                    "rollout_image_paths": image_paths,
                }
            )

        if self.sample_ratio < 1.0 and samples:
            count = max(1, int(len(samples) * self.sample_ratio))
            rng = random.Random(self.seed)
            samples = rng.sample(samples, count)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class WatcherSFTCollator:
    def __init__(self, processor):
        self.processor = processor

    def _load_rollout_images(self, sample: Dict[str, Any]) -> List[Image.Image]:
        if sample.get("rollout_images"):
            return list(sample["rollout_images"])
        return [load_image(path) for path in sample.get("rollout_image_paths", [])]

    def _build_messages(self, prompt: WatcherRolloutPrompt, assistant_text: str | None, add_generation_prompt: bool):
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt.user_text}]
        user_content.extend({"type": "image", "image": image} for image in prompt.images)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": user_content},
        ]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return messages, text

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples: List[Dict[str, Any]] = []
        for sample in batch:
            if sample is None:
                continue
            rollout_images = self._load_rollout_images(sample)
            prompt = build_rollout_prompt(
                instruction=str(sample.get("instruction", "")),
                plan_steps=sample.get("plan_steps", []),
                done_steps=sample.get("done_steps", []),
                active_step=str(sample.get("active_step", "")),
                pending_steps=sample.get("pending_steps", []),
                memory_start=str(sample.get("memory_start", "")),
                rollout_actions=sample.get("rollout_actions", []),
                rollout_images=rollout_images,
            )
            target_text = format_watcher_update_json(
                memory_end=str(sample.get("memory_end", "")),
                done=bool(sample.get("done", False)),
                next_subtask=str(sample.get("next_subtask", "")),
            )

            _, prompt_text = self._build_messages(prompt, assistant_text=None, add_generation_prompt=True)
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[rollout_images],
                return_tensors="pt",
                padding=False,
            )
            prompt_len = int(prompt_inputs["attention_mask"][0].sum().item())

            _, full_text = self._build_messages(prompt, assistant_text=target_text, add_generation_prompt=False)
            full_inputs = self.processor(
                text=[full_text],
                images=[rollout_images],
                return_tensors="pt",
                padding=False,
            )

            input_ids = full_inputs["input_ids"][0]
            attention_mask = full_inputs["attention_mask"][0]
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "pixel_values": full_inputs.get("pixel_values"),
                    "image_grid_thw": full_inputs.get("image_grid_thw"),
                }
            )

        if not samples:
            raise ValueError("Empty batch passed to WatcherSFTCollator")

        batch_size = len(samples)
        max_len = max(sample["input_ids"].shape[0] for sample in samples)
        tokenizer = getattr(self.processor, "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        pixel_values_list: List[torch.Tensor] = []
        image_grid_list: List[torch.Tensor] = []

        for idx, sample in enumerate(samples):
            seq_len = sample["input_ids"].shape[0]
            input_ids[idx, :seq_len] = sample["input_ids"]
            attention_mask[idx, :seq_len] = sample["attention_mask"]
            labels[idx, :seq_len] = sample["labels"]
            if sample["pixel_values"] is not None:
                pixel_values_list.append(sample["pixel_values"])
            if sample["image_grid_thw"] is not None:
                image_grid_list.append(sample["image_grid_thw"])

        batch_pixel_values = None
        batch_image_grid_thw = None
        if pixel_values_list:
            batch_pixel_values = torch.cat(
                [values.view(-1, values.shape[-1]) for values in pixel_values_list],
                dim=0,
            )
        if image_grid_list:
            batch_image_grid_thw = torch.cat(image_grid_list, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": batch_pixel_values,
            "image_grid_thw": batch_image_grid_thw,
        }


__all__ = [
    "ROLLOUT_SYSTEM_PROMPT",
    "WatcherRolloutPrompt",
    "WatcherSFTCollator",
    "WatcherSFTDataset",
    "build_rollout_prompt",
    "format_watcher_update_json",
    "select_rollout_paths",
]
