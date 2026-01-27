import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Any, Optional
import os

def load_image(image_path: str) -> Image.Image:
    """Load image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert('RGB')

class ThinkVLNDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = sample.get("messages", [])
        images_paths = sample.get("images", [])
        
        # action_label
        action_label = sample.get("action", sample.get("action_label", None))
        if action_label is not None:
            action_label = int(action_label)
            assert 0 <= action_label <= 3, f"Action label must be 0-3, got {action_label}"

        user_message = ""
        assistant_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
            elif msg.get("role") == "assistant":
                assistant_message = msg.get("content", "")

        image = None
        if images_paths:
            try:
                # Assuming images_paths is a list of paths
                image = load_image(images_paths[0])
            except Exception as e:
                # In a real scenario, we might want to handle this differently
                pass

        return {
            "idx": idx,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "image": image,
            "action_label": action_label,
            "sample": sample
        }

def collate_fn(batch):
    return batch

