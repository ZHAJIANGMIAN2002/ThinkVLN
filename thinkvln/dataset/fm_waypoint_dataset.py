"""Dataset and collator for flow-matching waypoint training."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .dataset import _episode_key_to_dir_key
from ..tools.dataset_utils import compute_previous_step_progress, load_image, select_memory_frame_indices


def extract_delta_waypoints(
    frame_idx: int,
    positions: List[List[float]],
    horizon: int = 5,
    action_dim: int = 2,
) -> torch.Tensor:
    """Build future delta waypoints relative to current frame position."""
    if not positions:
        return torch.zeros((horizon, action_dim), dtype=torch.float32)

    cur_idx = max(0, min(frame_idx, len(positions) - 1))
    curr = positions[cur_idx]
    out = []
    for step in range(1, horizon + 1):
        next_idx = min(cur_idx + step, len(positions) - 1)
        nxt = positions[next_idx]
        dx = float(nxt[0]) - float(curr[0])
        dz = float(nxt[1]) - float(curr[1])
        item = [dx, dz]
        if action_dim > 2:
            item += [0.0] * (action_dim - 2)
        out.append(item[:action_dim])
    return torch.tensor(out, dtype=torch.float32)


class ThinkVLNFMWaypointDataset(Dataset):
    """Frame-level dataset containing waypoint labels."""

    def __init__(
        self,
        waypoint_data_path: str,
        image_root: str,
        action_horizon: int = 5,
        action_dim: int = 2,
        skip_missing_images: bool = True,
    ):
        self.waypoint_data_path = waypoint_data_path
        self.image_root = image_root
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.skip_missing_images = bool(skip_missing_images)
        self.samples: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.waypoint_data_path):
            raise FileNotFoundError(f'waypoint data not found: {self.waypoint_data_path}')

        with open(self.waypoint_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)

                episode_key = rec['episode_key']
                positions = rec.get('positions') or []
                actions = rec.get('actions') or []
                num_frames = rec.get('num_frames', len(actions))
                if num_frames <= 0:
                    continue

                dir_key = _episode_key_to_dir_key(episode_key)
                subtask_sequence = rec.get('subtask_sequence', [1] * num_frames)
                plan = rec.get('plan', [''])

                for frame_idx in range(num_frames):
                    image_path = os.path.join(self.image_root, dir_key, f'{frame_idx:06d}_rgb.jpg')
                    if self.skip_missing_images and not os.path.exists(image_path):
                        continue
                    subtask_idx = int(subtask_sequence[min(frame_idx, len(subtask_sequence) - 1)])
                    plan_step = plan[max(0, min(subtask_idx - 1, len(plan) - 1))] if plan else ''

                    self.samples.append({
                        'data_type': 'fm_waypoint',
                        'episode_key': episode_key,
                        'frame_idx': frame_idx,
                        'instruction': rec.get('instruction', ''),
                        'current_plan_step': plan_step,
                        'subtask_sequence': subtask_sequence,
                        'positions': positions,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class ThinkVLNFMDataCollator:
    """Collator that batches FM waypoint samples."""

    def __init__(
        self,
        processor,
        image_root: str,
        action_horizon: int = 5,
        action_dim: int = 2,
        num_query_tokens: int = 4,
        action_query_token_id: int = 151700,
        progress_query_token_id: int = 151701,
        memory_num_history_images: int = 8,
    ):
        self.processor = processor
        self.image_root = image_root
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.action_query_token_id = int(action_query_token_id)
        self.progress_query_token_id = int(progress_query_token_id)
        self.memory_num_history_images = int(memory_num_history_images)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        proc = [self._process_one(sample) for sample in batch]
        max_len = max(x['input_ids'].shape[0] for x in proc)
        pad_id = self.processor.tokenizer.pad_token_id

        input_ids = torch.full((len(proc), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(proc), max_len), dtype=torch.long)
        waypoint_labels = torch.zeros((len(proc), self.action_horizon, self.action_dim), dtype=torch.float32)

        pixel_values_list = []
        image_grid_list = []

        for i, item in enumerate(proc):
            n = item['input_ids'].shape[0]
            input_ids[i, :n] = item['input_ids']
            attention_mask[i, :n] = item['attention_mask']
            waypoint_labels[i] = item['waypoint_labels']
            if item['pixel_values'] is not None:
                pixel_values_list.append(item['pixel_values'])
            if item['image_grid_thw'] is not None:
                image_grid_list.append(item['image_grid_thw'])

        pixel_values = None
        image_grid_thw = None
        if pixel_values_list:
            pixel_values = torch.cat([pv.view(-1, pv.shape[-1]) for pv in pixel_values_list], dim=0)
        if image_grid_list:
            image_grid_thw = torch.cat(image_grid_list, dim=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'waypoint_labels': waypoint_labels,
        }

    def _process_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        dir_episode_key = _episode_key_to_dir_key(sample['episode_key'])
        frame_idx = int(sample['frame_idx'])
        subtask_sequence = sample.get('subtask_sequence', [])

        history_indices = select_memory_frame_indices(
            frame_idx=frame_idx,
            subtask_sequence=subtask_sequence,
            memory_num_history_images=self.memory_num_history_images,
        )
        selected_indices = history_indices + [frame_idx]

        images = []
        for idx in selected_indices:
            path = os.path.join(self.image_root, dir_episode_key, f'{idx:06d}_rgb.jpg')
            images.append(load_image(path))

        prev_progress = compute_previous_step_progress(frame_idx, subtask_sequence)
        prompt = (
            f"Instruction: {sample['instruction']}\n"
            f"Current subgoal: {sample['current_plan_step']}\n"
            f"Previous progress: {prev_progress:.3f}\n"
            "Predict waypoint deltas for the next horizon."
        )

        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors='pt', padding=False)

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        query_tokens = []
        for _ in range(self.num_query_tokens):
            query_tokens.append(self.action_query_token_id)
            query_tokens.append(self.progress_query_token_id)
        query_tokens = torch.tensor(query_tokens, dtype=torch.long)

        input_ids = torch.cat([input_ids, query_tokens], dim=0)
        attention_mask = torch.cat([attention_mask, torch.ones_like(query_tokens)], dim=0)

        waypoint_labels = extract_delta_waypoints(
            frame_idx=frame_idx,
            positions=sample.get('positions', []),
            horizon=self.action_horizon,
            action_dim=self.action_dim,
        )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': inputs['pixel_values'] if 'pixel_values' in inputs else None,
            'image_grid_thw': inputs['image_grid_thw'] if 'image_grid_thw' in inputs else None,
            'waypoint_labels': waypoint_labels,
        }
