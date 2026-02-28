"""Auto-regressive VLN dataset for future action + progress prediction.

This dataset reuses the existing ThinkVLN action JSONL format, but instead of
producing separate action/progress heads, it serializes the next N steps as
plain language tokens:

    ↑ <p_0> → <p_25> ↑ <p_50> STOP <p_100>

All supervision is standard causal LM loss on these tokens.
"""

import os
import json
from typing import List, Dict, Any, Optional

import torch
from torch.utils.data import Dataset

from ..tools.dataset_utils import load_image, extract_action_chunk


# Follow StreamVLN arrow-style actions
INT_ACTION_TO_SYMBOL = {
    0: "STOP",
    1: "↑",
    2: "←",
    3: "→",
}


class ARVLNDataset(Dataset):
    """Frame-level AR dataset built from action trajectories."""

    def __init__(
        self,
        action_data_path: str,
        image_root: str,
        num_future_steps: int = 4,
    ):
        self.action_data_path = action_data_path
        self.image_root = image_root
        self.num_future_steps = num_future_steps

        self.samples: List[Dict[str, Any]] = []
        self._load_action_data()

        print(
            f"ARVLNDataset: {len(self.samples)} frame-level action samples "
            f"(future_steps={self.num_future_steps})"
        )

    def _load_action_data(self) -> None:
        """Load action trajectories and create frame-level samples."""
        if not self.action_data_path or not os.path.exists(self.action_data_path):
            raise FileNotFoundError(f"action_data_path not found: {self.action_data_path}")

        with open(self.action_data_path, "r") as f:
            for line in f:
                traj = json.loads(line.strip())
                num_frames = traj["num_frames"]
                for frame_idx in range(num_frames):
                    subtask_idx = traj["subtask_sequence"][frame_idx]
                    plan_step = (
                        traj["plan"][subtask_idx - 1]
                        if subtask_idx > 0
                        else traj["plan"][0]
                    )

                    self.samples.append(
                        {
                            "episode_key": traj["episode_key"],
                            "frame_idx": frame_idx,
                            "instruction": traj["instruction"],
                            "current_plan_step": plan_step,
                            "actions": traj["actions"],
                            "subtask_sequence": traj["subtask_sequence"],
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class ARVLNDataCollator:
    """Collator that builds AR LM inputs from raw ARVLNDataset samples.

    It constructs:
      - User: image + textual prompt.
      - Assistant: a compact sequence of 4 (action, progress-bin) pairs.

    Labels are standard causal LM labels with the user part masked to -100.
    """

    def __init__(
        self,
        processor,
        image_root: str,
        num_future_steps: int = 4,
        progress_bin_step: int = 5,
    ):
        self.processor = processor
        self.image_root = image_root
        self.num_future_steps = num_future_steps
        self.progress_bin_step = progress_bin_step

        # Define discrete progress tokens, e.g. <p_0>, <p_5>, ..., <p_100>
        self.progress_tokens = [
            f"<p_{i * self.progress_bin_step}>" for i in range(0, 101 // self.progress_bin_step + 1)
        ]

        # Verify progress tokens are properly registered as single tokens
        self._verify_progress_tokens()

        self.prompt_template = (
            "Based on the current observation and subgoal '{subgoal}', "
            "predict the next 4 actions and their within-subtask progress. "
            "Use actions STOP, ↑, ←, → and progress bins <p_0> ... <p_100> "
            "where each bin represents a 5% interval from 0% to 100%."
        )
    
    def _verify_progress_tokens(self) -> None:
        """Verify that progress tokens are registered as single tokens in tokenizer."""
        sample_tokens = [self.progress_tokens[0], self.progress_tokens[-1]]
        for tok in sample_tokens:
            encoded = self.processor.tokenizer.encode(tok, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    f"Progress token '{tok}' is split into {len(encoded)} tokens: {encoded}. "
                    f"Make sure to call add_progress_tokens() with special_tokens=True "
                    f"before creating the collator."
                )
        print(f"✓ Progress tokens verified: {sample_tokens[0]} and {sample_tokens[-1]} "
              f"are single tokens in tokenizer")

    def _episode_to_dir_key(self, episode_key: str) -> str:
        """Convert dataset episode_key to directory key."""
        parts = episode_key.split("_")
        if len(parts) == 2:
            scene_id = parts[0]
            episode_id = parts[1]
            return f"{scene_id}_r2r_{int(episode_id):06d}"
        return episode_key

    def _progress_to_bin_token(self, progress: float) -> str:
        """Map continuous progress [0,1] to nearest 5% bin token."""
        percent = max(0.0, min(1.0, float(progress))) * 100.0
        bin_idx = int(round(percent / self.progress_bin_step))
        max_idx = len(self.progress_tokens) - 1
        bin_idx = max(0, min(max_idx, bin_idx))
        return self.progress_tokens[bin_idx]

    def _build_target_text(
        self,
        actions: List[int],
        progresses: List[float],
    ) -> str:
        """Serialize 4 (action, progress) pairs into a short text sequence."""
        tokens: List[str] = []
        for a, p in zip(actions, progresses):
            action_sym = INT_ACTION_TO_SYMBOL.get(int(a), "STOP")
            prog_tok = self._progress_to_bin_token(p)
            tokens.append(action_sym)
            tokens.append(prog_tok)
        return " ".join(tokens)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples = []
        for sample in batch:
            if sample is None:
                continue

            episode_key = sample["episode_key"]
            frame_idx = sample["frame_idx"]
            dir_episode_key = self._episode_to_dir_key(episode_key)

            image_path = os.path.join(
                self.image_root,
                dir_episode_key,
                f"{frame_idx:06d}_rgb.jpg",
            )
            image = load_image(image_path)

            # Extract future actions/progress (within-subtask, 4 steps)
            actions, progresses = extract_action_chunk(
                frame_idx,
                sample["actions"],
                sample["subtask_sequence"],
                num_steps=self.num_future_steps,
            )
            target_text = self._build_target_text(actions, progresses)

            prompt = self.prompt_template.format(subgoal=sample["current_plan_step"])

            # Prompt-only to find boundary
            messages_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                messages_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs_prompt = self.processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt",
                padding=False,
            )
            prompt_len = inputs_prompt["input_ids"].shape[1]

            # Full (user + assistant) sequence
            messages_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target_text},
                    ],
                },
            ]
            text_full = self.processor.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = self.processor(
                text=[text_full],
                images=[image],
                return_tensors="pt",
                padding=False,
            )

            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]

            labels = input_ids.clone()
            labels[:prompt_len] = -100

            samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "pixel_values": inputs.get("pixel_values"),
                    "image_grid_thw": inputs.get("image_grid_thw"),
                }
            )

        if not samples:
            raise ValueError("Empty batch passed to ARVLNDataCollator")

        batch_size = len(samples)
        max_len = max(s["input_ids"].shape[0] for s in samples)
        pad_id = self.processor.tokenizer.pad_token_id

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        pixel_values_list: List[torch.Tensor] = []
        image_grid_list: List[torch.Tensor] = []

        for i, s in enumerate(samples):
            seq_len = s["input_ids"].shape[0]
            input_ids[i, :seq_len] = s["input_ids"]
            attention_mask[i, :seq_len] = s["attention_mask"]
            labels[i, :seq_len] = s["labels"]

            if s["pixel_values"] is not None:
                pixel_values_list.append(s["pixel_values"])
            if s["image_grid_thw"] is not None:
                image_grid_list.append(s["image_grid_thw"])

        batch_pixel_values: Optional[torch.Tensor] = None
        batch_image_grid_thw: Optional[torch.Tensor] = None
        if pixel_values_list:
            batch_pixel_values = torch.cat(
                [pv.view(-1, pv.shape[-1]) for pv in pixel_values_list],
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

