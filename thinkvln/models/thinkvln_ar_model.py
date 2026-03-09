"""Minimal auto-regressive VLN model helpers based on Qwen3VL.

This module intentionally keeps things simple:
- We directly use HuggingFace Qwen3VLForConditionalGeneration as the backbone.
- All supervision is standard causal LM loss on text tokens (actions + progress bins).
- Visual inputs are handled by the original Qwen3VL multi-modal stack via AutoProcessor.

The actual AR behavior (how actions/progress are serialized into tokens) is
defined in the AR dataset/collator and the trainer, not in this file.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel


def load_ar_model(
    model_name_or_path: str,
    use_flash_attention_2: bool = False,
    bf16: bool = True,
    device_map: Optional[str] = "auto",
) -> PreTrainedModel:
    """Load a minimal auto-regressive VLN model.

    This is just a thin wrapper around AutoModelForCausalLM using Qwen3VL.
    It relies entirely on HuggingFace's implementation without extra heads.
    """
    dtype = torch.bfloat16 if bf16 else torch.float32

    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    return model
