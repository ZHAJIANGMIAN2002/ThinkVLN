import copy
import json
import os
import random
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from streamvln.utils.utils import DEFAULT_MEMORY_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX
from thinkvln.datagen.generation.watcher_actor_memory_dataset import parse_sample_id_to_episode_frame
from thinkvln.dataset.dataset import _episode_key_to_dir_key
from thinkvln.tools.dataset_utils import (
    compute_current_step_progress,
    compute_done_label,
    compute_previous_step_progress,
    extract_action_chunk,
    select_memory_frame_indices,
)


INT_ACTION_TO_SYMBOL = {
    0: "STOP",
    1: "↑",
    2: "←",
    3: "→",
}


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_action_id(action) -> int:
    if isinstance(action, str):
        mapping = {
            "stop": 0,
            "forward": 1,
            "turn_left": 2,
            "turn_right": 3,
            "STOP": 0,
            "↑": 1,
            "←": 2,
            "→": 3,
        }
        return int(mapping.get(action, 0))
    try:
        value = int(action)
    except (TypeError, ValueError):
        return 0
    if value < 0:
        return 0
    return value if value in INT_ACTION_TO_SYMBOL else 0


def _actions_to_text(actions: List[int]) -> str:
    return " ".join(INT_ACTION_TO_SYMBOL.get(int(action), "STOP") for action in actions)


def _materialized_actor_row(row: Dict) -> bool:
    return (
        isinstance(row, dict)
        and "frame_idx" in row
        and "subtask" in row
        and "action_labels" in row
        and ("image_path" in row or "episode_key" in row)
    )


def build_streamvln_actor_prompt(
    instruction: str,
    subtask: str,
    watcher_hint: Optional[str] = None,
    include_visual_memory: bool = False,
    previous_progress: Optional[float] = None,
) -> str:
    lines = [
        "<image>",
        f"Instruction: {str(instruction or '').strip()}",
        f"Current subtask: {str(subtask or '').strip()}",
    ]
    if previous_progress is not None:
        lines.append(f"Previous progress: {float(previous_progress):.4f}")
    hint_text = str(watcher_hint or "").strip()
    if hint_text:
        lines.append(f"Watcher hint: {hint_text}")
    if include_visual_memory:
        lines.append(f"Historical observations: {DEFAULT_MEMORY_TOKEN}")
    lines.append("Predict the next 4 actions using STOP, ↑, ←, →.")
    return "\n".join(lines)


def _load_watcher_hints(path: Optional[str], watcher_memory_ratio: float, seed: int) -> Dict[tuple, str]:
    if not path or not os.path.exists(path):
        return {}
    ratio = max(0.0, min(1.0, float(watcher_memory_ratio)))
    rng = random.Random(int(seed))
    hints: Dict[tuple, str] = {}
    for row in _load_jsonl(path):
        hint = str(row.get("watcher_hint") or row.get("memory_start") or "").strip()
        if not hint:
            continue
        if ratio < 1.0 and rng.random() > ratio:
            continue
        episode_key = str(row.get("episode_key") or "").strip()
        frame_idx = row.get("frame_idx", row.get("pivot_frame"))
        if (not episode_key or frame_idx is None) and row.get("sample_id"):
            try:
                episode_key, frame_idx = parse_sample_id_to_episode_frame(str(row.get("sample_id")))
            except ValueError:
                continue
        if not episode_key or frame_idx is None:
            continue
        episode_key = str(episode_key).strip()
        if frame_idx is None:
            continue
        hints[(episode_key, int(frame_idx))] = hint
    return hints


def _resolve_sparse_history_indices(
    frame_idx: int,
    subtask_sequence: List[int],
    memory_num_history_images: int,
) -> List[int]:
    return select_memory_frame_indices(
        frame_idx=frame_idx,
        subtask_sequence=subtask_sequence,
        memory_num_history_images=memory_num_history_images,
    )


def _pad_history_indices(
    history_indices: List[int],
    frame_idx: int,
    memory_num_history_images: int,
) -> List[int]:
    target = max(0, int(memory_num_history_images))
    if target == 0:
        return []
    history = [int(idx) for idx in history_indices[:target]]
    pad_value = history[0] if history else max(0, int(frame_idx))
    if len(history) < target:
        history = [pad_value] * (target - len(history)) + history
    return history[:target]


def _pad_history_paths(
    history_paths: List[str],
    current_image_path: Optional[str],
    memory_num_history_images: int,
) -> List[str]:
    target = max(0, int(memory_num_history_images))
    if target == 0:
        return []
    paths = [str(path) for path in history_paths[:target] if str(path or "").strip()]
    pad_value = paths[0] if paths else str(current_image_path or "").strip()
    if len(paths) < target and pad_value:
        paths = [pad_value] * (target - len(paths)) + paths
    return paths[:target]


def _frame_to_image_path(image_root: Optional[str], episode_key: str, frame_idx: int) -> Optional[str]:
    if not image_root:
        return None
    dir_key = _episode_key_to_dir_key(episode_key)
    return os.path.join(str(image_root), dir_key, f"{int(frame_idx):06d}_rgb.jpg")


def _resolve_existing_image_path(image_path: str) -> str:
    path = str(image_path)
    if os.path.exists(path):
        return path
    r2r_marker = f"{os.sep}R2R_back{os.sep}images{os.sep}"
    if r2r_marker in path:
        fallback = path.replace(r2r_marker, f"{os.sep}R2R_back{os.sep}r2r{os.sep}")
        if os.path.exists(fallback):
            return fallback
    return path


def _build_actor_sample(
    traj: Dict,
    frame_idx: int,
    done_threshold: float,
    watcher_hint: Optional[str] = None,
    dataset_name: Optional[str] = None,
    image_root: Optional[str] = None,
    memory_num_history_images: int = 0,
) -> Dict:
    episode_key = str(traj["episode_key"])
    instruction = str(traj.get("instruction", ""))
    plan = [str(step) for step in traj.get("plan", [])]
    actions = list(traj.get("actions", []))
    subtask_sequence = list(traj.get("subtask_sequence", []))
    subtask_idx = max(1, int(subtask_sequence[frame_idx]))
    plan_idx = min(subtask_idx - 1, len(plan) - 1)
    subtask = plan[plan_idx]
    action_labels, _ = extract_action_chunk(
        frame_idx=frame_idx,
        actions=[_normalize_action_id(action) for action in actions],
        subtask_sequence=subtask_sequence,
        num_steps=4,
    )
    progress_label = compute_current_step_progress(frame_idx, subtask_sequence)
    previous_progress = compute_previous_step_progress(frame_idx, subtask_sequence)
    done_label = compute_done_label(progress_label, threshold=done_threshold)
    history_frame_indices = _resolve_sparse_history_indices(
        frame_idx=frame_idx,
        subtask_sequence=subtask_sequence,
        memory_num_history_images=memory_num_history_images,
    )
    sample = {
        "episode_key": episode_key,
        "frame_idx": int(frame_idx),
        "instruction": instruction,
        "subtask": subtask,
        "watcher_hint": str(watcher_hint).strip() if str(watcher_hint or "").strip() else None,
        "action_labels": list(action_labels),
        "progress_label": float(progress_label),
        "previous_progress": float(previous_progress),
        "done_label": float(done_label),
        "history_frame_indices": list(history_frame_indices),
    }
    if dataset_name is not None:
        sample["dataset_name"] = str(dataset_name)
    if image_root:
        sample["image_path"] = _frame_to_image_path(image_root, episode_key, int(frame_idx))
        sample["history_image_paths"] = [
            _frame_to_image_path(image_root, episode_key, idx) for idx in history_frame_indices
        ]
    else:
        sample["history_image_paths"] = []
    sample["input_mode"] = "instruction_subtask_hint" if sample["watcher_hint"] else "instruction_subtask"
    return sample


def load_streamvln_actor_samples(
    summary_path: str,
    watcher_memory_path: Optional[str] = None,
    watcher_memory_ratio: float = 1.0,
    done_threshold: float = 0.85,
    memory_num_history_images: int = 0,
    seed: int = 42,
) -> List[Dict]:
    rows = _load_jsonl(summary_path)
    if rows and _materialized_actor_row(rows[0]):
        return rows

    watcher_hints = _load_watcher_hints(watcher_memory_path, watcher_memory_ratio, seed)
    samples: List[Dict] = []

    for traj in rows:
        episode_key = str(traj["episode_key"])
        subtask_sequence = list(traj.get("subtask_sequence", []))
        num_frames = int(traj.get("num_frames", len(subtask_sequence)))

        if not subtask_sequence or num_frames <= 0:
            continue

        for frame_idx in range(min(num_frames, len(subtask_sequence))):
            watcher_hint = watcher_hints.get((episode_key, frame_idx))
            samples.append(
                _build_actor_sample(
                    traj=traj,
                    frame_idx=frame_idx,
                    done_threshold=done_threshold,
                    watcher_hint=watcher_hint,
                    memory_num_history_images=memory_num_history_images,
                )
            )

    return samples


def build_materialized_streamvln_actor_records(
    summary_specs: List[Dict],
    watcher_memory_path: Optional[str] = None,
    watcher_memory_ratio: float = 1.0,
    done_threshold: float = 0.85,
    memory_num_history_images: int = 0,
    seed: int = 42,
) -> List[Dict]:
    watcher_hints = _load_watcher_hints(watcher_memory_path, watcher_memory_ratio, seed)
    records: List[Dict] = []
    for spec in summary_specs:
        dataset_name = str(spec.get("dataset_name") or "")
        summary_path = str(spec["summary_path"])
        image_root = str(spec.get("image_root") or "")
        for traj in _load_jsonl(summary_path):
            subtask_sequence = list(traj.get("subtask_sequence", []))
            num_frames = int(traj.get("num_frames", len(subtask_sequence)))
            if not subtask_sequence or num_frames <= 0:
                continue
            episode_key = str(traj["episode_key"])
            for frame_idx in range(min(num_frames, len(subtask_sequence))):
                records.append(
                    _build_actor_sample(
                        traj=traj,
                        frame_idx=frame_idx,
                        done_threshold=done_threshold,
                        watcher_hint=watcher_hints.get((episode_key, frame_idx)),
                        dataset_name=dataset_name or None,
                        image_root=image_root or None,
                        memory_num_history_images=memory_num_history_images,
                    )
                )
    return records


def _replace_special_token_ids(tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    input_ids = input_ids.clone()
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    if image_token_id is not None:
        input_ids[input_ids == int(image_token_id)] = IMAGE_TOKEN_INDEX
    memory_token_id = tokenizer.convert_tokens_to_ids("<memory>")
    if memory_token_id is not None:
        input_ids[input_ids == int(memory_token_id)] = MEMORY_TOKEN_INDEX
    return input_ids


def _tokenize_actor_sample(tokenizer, prompt_text: str, target_text: str):
    tokenizer = copy.deepcopy(tokenizer)
    if hasattr(tokenizer, "add_tokens"):
        tokenizer.add_tokens(["<image>", "<memory>"], special_tokens=True)

    prompt_messages = [{"role": "user", "content": prompt_text}]
    full_messages = prompt_messages + [{"role": "assistant", "content": target_text}]

    prompt_rendered = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_rendered = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_rendered, return_tensors="pt").input_ids[0]
    input_ids = tokenizer(full_rendered, return_tensors="pt").input_ids[0]
    input_ids = _replace_special_token_ids(tokenizer, input_ids)

    labels = input_ids.clone()
    labels[: prompt_ids.shape[0]] = IGNORE_INDEX
    return input_ids, labels


class StreamVLNActorDataset(Dataset):
    def __init__(self, tokenizer, data_args, task_id: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.task_id = int(task_id)
        self.image_root = str(getattr(data_args, "image_root", "") or "")
        if not self.image_root:
            self.image_root = str(getattr(data_args, "image_folder", "") or "")
        self.memory_num_history_images = max(
            0,
            int(
                getattr(
                    data_args,
                    "memory_num_history_images",
                    getattr(data_args, "num_history", 0) or 0,
                )
            ),
        )
        summary_path = str(getattr(data_args, "summary_data_path", "") or "")
        if not summary_path:
            summary_path = str(getattr(data_args, "data_path", "") or "")
        from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor

        self.image_processor = SigLipImageProcessor()
        self.samples = load_streamvln_actor_samples(
            summary_path=summary_path,
            watcher_memory_path=getattr(data_args, "watcher_memory_path", None),
            watcher_memory_ratio=float(getattr(data_args, "watcher_memory_ratio", 1.0)),
            done_threshold=float(getattr(data_args, "done_threshold", 0.85)),
            memory_num_history_images=self.memory_num_history_images,
            seed=int(getattr(data_args, "watcher_memory_seed", 42)),
        )

    def __len__(self):
        return len(self.samples)

    def _load_current_image(
        self,
        episode_key: str,
        frame_idx: int,
        image_path: Optional[str] = None,
    ) -> torch.Tensor:
        if not image_path:
            dir_key = _episode_key_to_dir_key(episode_key)
            image_path = os.path.join(self.image_root, dir_key, f"{int(frame_idx):06d}_rgb.jpg")
        image_path = _resolve_existing_image_path(str(image_path))
        image = Image.open(image_path).convert("RGB")
        return self.image_processor.preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def _load_sample_images(self, sample: Dict) -> torch.Tensor:
        padded_history_paths = _pad_history_paths(
            history_paths=list(sample.get("history_image_paths", [])),
            current_image_path=sample.get("image_path"),
            memory_num_history_images=self.memory_num_history_images,
        )
        if padded_history_paths:
            tensors = [
                self._load_current_image(sample["episode_key"], idx, image_path=path)
                for idx, path in enumerate(padded_history_paths)
            ]
        else:
            padded_history = _pad_history_indices(
                history_indices=list(sample.get("history_frame_indices", [])),
                frame_idx=int(sample["frame_idx"]),
                memory_num_history_images=self.memory_num_history_images,
            )
            tensors = [
                self._load_current_image(sample["episode_key"], idx)
                for idx in padded_history
            ]
        tensors.append(
            self._load_current_image(
                sample["episode_key"],
                sample["frame_idx"],
                image_path=sample.get("image_path"),
            )
        )
        return torch.stack(tensors, dim=0)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        include_visual_memory = self.memory_num_history_images > 0
        prev_prog = float(sample.get("previous_progress", 0.0))
        prompt = build_streamvln_actor_prompt(
            instruction=sample["instruction"],
            subtask=sample["subtask"],
            watcher_hint=sample.get("watcher_hint"),
            include_visual_memory=include_visual_memory,
            previous_progress=prev_prog,
        )
        target_text = _actions_to_text(sample["action_labels"])
        input_ids, labels = _tokenize_actor_sample(self.tokenizer, prompt, target_text)
        image_tensor = self._load_sample_images(sample)
        time_anchor = int(sample["frame_idx"])
        if include_visual_memory:
            time_anchor = max(1, time_anchor)
        time_ids = torch.tensor([time_anchor], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": image_tensor,
            "time_ids": time_ids,
            "task_type": self.task_id,
            "progress_labels": torch.tensor(sample["progress_label"], dtype=torch.float32),
            "done_labels": torch.tensor(sample["done_label"], dtype=torch.float32),
        }


def streamvln_actor_collate_fn(batch, tokenizer):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids_batch = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels_batch = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )
    input_ids_batch = input_ids_batch[:, : tokenizer.model_max_length]
    labels_batch = labels_batch[:, : tokenizer.model_max_length]
    images = torch.stack([item["images"] for item in batch], dim=0)
    time_ids = pad_sequence(
        [item["time_ids"] for item in batch],
        batch_first=True,
        padding_value=-1,
    )
    attention_mask = input_ids_batch.ne(pad_token_id)
    return {
        "images": images,
        "time_ids": time_ids,
        "attention_mask": attention_mask,
        "input_ids": input_ids_batch,
        "labels": labels_batch,
        "task_type": [item["task_type"] for item in batch],
        "progress_labels": torch.stack([item["progress_labels"] for item in batch], dim=0),
        "done_labels": torch.stack([item["done_labels"] for item in batch], dim=0),
    }
