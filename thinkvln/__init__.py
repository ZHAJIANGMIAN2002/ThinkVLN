"""
ThinkVLN: Vision-Language Navigation with Chain-of-Thought Reasoning

A modular framework for training and evaluating vision-language navigation models
with explicit reasoning capabilities.
"""

__version__ = "0.1.0"

from .dataset.dataset import ThinkVLNDataset, load_image, collate_fn

__all__ = [
    "ThinkVLNDataset",
    "load_image",
    "collate_fn",
]
