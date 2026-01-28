"""
ThinkVLN Dataset API Contract

This module defines the API contract for ThinkVLN datasets and data collators.
The actual implementation will be completed in a separate task.

Expected Data Format:
    - Action Data: JSONL with trajectory, actions, subtask_sequence
    - CoT Data: JSONL with frame_key, instruction, plan, answer (reasoning)
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
import os


# Action encoding mapping
ACTION_MAPPING = {
    0: "stop",         # Stop navigation
    1: "forward",      # Move forward
    2: "turn_left",    # Turn left 15 degrees
    3: "turn_right"    # Turn right 15 degrees
}


class ThinkVLNDataset(Dataset):
    """
    Abstract dataset class for ThinkVLN training.
    
    This dataset loads and manages both action and CoT data from JSONL files.
    It returns raw samples that will be processed by the collator.
    
    Expected raw data formats:
    
    CoT Sample (from cot_dataset_*.jsonl):
        {
            "frame_key": "17DRP5sb8fy_10154_000035",
            "episode_key": "17DRP5sb8fy_10154",
            "instruction": "Walk forward in the direction...",
            "plan": "1. Walk forward toward...\n2. Veer right...",
            "step_id": 35,
            "action": 1,
            "answer": "[localization]\n...\n[reason]\n...\n[action]\nforward"
        }
    
    Action Sample (from summary_full.jsonl):
        {
            "video": "images/17DRP5sb8fy_r2r_010154",
            "actions": [-1, 2, 2, 2, 2, 1, 1, ...],
            "episode_key": "17DRP5sb8fy_10154",
            "instruction": "Walk forward in the direction...",
            "plan": ["Walk forward toward...", "Veer right.", ...],
            "num_frames": 40,
            "subtask_sequence": [1, 1, 1, 1, 1, ...],
            "keyframes": [0, 4, 5, 10, 15, 20, ...]
        }
    """
    
    def __init__(
        self,
        action_data_path: Optional[str] = None,
        cot_data_path: Optional[str] = None,
        data_root: str = "data",
        action_cot_ratio: float = 0.5,
    ):
        """
        Initialize dataset.
        
        Args:
            action_data_path: Path to action JSONL file (summary_full.jsonl)
            cot_data_path: Path to CoT JSONL file (cot_dataset_*.jsonl)
            data_root: Root directory for data files
            action_cot_ratio: Ratio of action samples (0.5 = 50% action, 50% CoT)
        """
        self.action_data_path = action_data_path
        self.cot_data_path = cot_data_path
        self.data_root = data_root
        self.action_cot_ratio = action_cot_ratio
        
        # Placeholder - actual implementation will load data
        self.samples = []
        
        # TODO: Implement data loading logic
        # - Load action data from JSONL
        # - Load CoT data from JSONL
        # - Mix samples according to ratio
        # - Create index mapping
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dict with keys:
                - data_type: "action" or "cot"
                - For action: episode_key, instruction, plan, frame_idx, actions, subtask_sequence
                - For cot: frame_key, episode_key, instruction, plan, reasoning
        """
        # TODO: Implement sample retrieval
        # This is a placeholder that returns empty dict
        return self.samples[idx] if self.samples else {}


class ThinkVLNDataCollator:
    """
    Data collator for ThinkVLN training.
    
    Processes raw samples from the dataset and converts them into model inputs.
    Handles both action and CoT samples, mixing them in the same batch.
    
    Expected Output Format:
    
    For Action Samples:
        {
            "input_ids": torch.LongTensor,           # [batch_size, seq_len] with query tokens
            "attention_mask": torch.LongTensor,      # [batch_size, seq_len]
            "pixel_values": torch.FloatTensor,       # Processed images
            "image_grid_thw": torch.LongTensor,      # Image grid info
            "action_labels": torch.LongTensor,       # [batch_size, 4] values 0-3
            "progress_labels": torch.FloatTensor,    # [batch_size, 4] values 0.0-1.0
            "labels": None,
        }
    
    For CoT Samples:
        {
            "input_ids": torch.LongTensor,           # [batch_size, seq_len] NO query tokens
            "attention_mask": torch.LongTensor,      # [batch_size, seq_len]
            "pixel_values": torch.FloatTensor,       # Processed images
            "image_grid_thw": torch.LongTensor,      # Image grid info
            "labels": torch.LongTensor,              # [batch_size, seq_len] for LM loss
            "action_labels": None,
            "progress_labels": None,
        }
    """
    
    def __init__(
        self,
        processor,
        num_query_tokens: int = 4,
        query_token_id: int = 151700,
        data_root: str = "data",
    ):
        """
        Initialize collator.
        
        Args:
            processor: Qwen3VL processor for image and text processing
            num_query_tokens: Number of query tokens for action prediction (default: 4)
            query_token_id: Token ID for query tokens (default: 151700)
            data_root: Root directory for loading images
        """
        self.processor = processor
        self.num_query_tokens = num_query_tokens
        self.query_token_id = query_token_id
        self.data_root = data_root
        
        # Action prompts
        self.action_prompt_template = (
            "Based on the current observation and subgoal '{subgoal}', "
            "predict the next 4 actions."
        )
        
        # CoT prompts
        self.cot_prompt_template = (
            "Based on the current observation and subgoal '{subgoal}', "
            "think step by step to determine the action."
        )
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of samples.
        
        Args:
            batch: List of raw samples from dataset
        
        Returns:
            Dictionary of tensors ready for model input
        """
        # TODO: Implement collation logic
        # 1. Separate action and CoT samples
        # 2. Load images from disk
        # 3. Process action samples:
        #    - Apply action prompt
        #    - Load image(s)
        #    - Process with Qwen3VL processor
        #    - Append query token IDs
        #    - Extract action labels (next 4 actions)
        #    - Compute progress labels from subtask_sequence
        # 4. Process CoT samples:
        #    - Apply CoT prompt with reasoning text
        #    - Load image
        #    - Process with Qwen3VL processor
        #    - Create labels for LM loss
        # 5. Pad and combine into single batch
        
        # Placeholder return - will be replaced with actual implementation
        return {
            "input_ids": torch.zeros(len(batch), 128, dtype=torch.long),
            "attention_mask": torch.ones(len(batch), 128, dtype=torch.long),
            "pixel_values": None,
            "image_grid_thw": None,
            "action_labels": None,
            "progress_labels": None,
            "labels": None,
        }


def load_image(image_path: str) -> Image.Image:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
    
    Returns:
        PIL Image in RGB format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert('RGB')

