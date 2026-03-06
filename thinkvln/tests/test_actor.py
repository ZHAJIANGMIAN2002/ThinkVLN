#!/usr/bin/env python3
"""Unit tests for ThinkVLNActor heads and forward routing."""

import os
import sys
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, os.path.abspath("."))

from thinkvln.models.actor_config import ThinkVLNActorConfig
from thinkvln.models.thinkvln_actor import (
    ActionClassificationHead,
    ProgressRegressionHead,
    SharedProjector,
    ThinkVLNActor,
)


class _DummyOutputs:
    def __init__(self, hidden_states: torch.Tensor):
        self._hidden_states = hidden_states
        self.past_key_values = None
        self.hidden_states = None
        self.attentions = None

    def __getitem__(self, idx):
        if idx == 0:
            return self._hidden_states
        raise IndexError(idx)


def _create_mock_config():
    from thinkvln.models.thinkvln_config import ThinkVLNConfig

    return ThinkVLNConfig(
        use_return_dict=True,
        num_query_tokens=4,
        text_config={
            "hidden_size": 512,
            "vocab_size": 32000,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
        },
        vision_config={
            "hidden_size": 512,
            "image_size": 224,
        },
    )


def test_shared_projector_forward_shape():
    projector = SharedProjector(input_size=768, hidden_size=384, dropout=0.1)
    x = torch.randn(2, 5, 768)
    out = projector(x)
    assert out.shape == (2, 5, 384)


def test_basic_heads_forward_shape():
    action_head = ActionClassificationHead(hidden_size=384, num_classes=4, dropout=0.1)
    progress_head = ProgressRegressionHead(hidden_size=384, dropout=0.1)
    x = torch.randn(3, 4, 384)
    action_logits = action_head(x)
    progress = progress_head(x)
    assert action_logits.shape == (3, 4, 4)
    assert progress.shape == (3, 4)
    assert torch.all(progress >= 0.0) and torch.all(progress <= 1.0)


@patch("thinkvln.models.thinkvln_actor.ThinkVLNForConditionalGeneration.__init__")
def test_actor_forward_action_mode_shapes_and_done_loss(mock_parent_init):
    mock_parent_init.return_value = None
    config = _create_mock_config()
    actor_config = ThinkVLNActorConfig(num_query_tokens=4)
    actor = ThinkVLNActor(config, actor_config)

    # super().__init__ is patched out in this unit test path.
    actor.config = config
    hidden_states = torch.randn(2, 32, 512)
    actor.model = MagicMock(return_value=_DummyOutputs(hidden_states))

    outputs = actor.forward(
        input_ids=torch.randint(0, 32000, (2, 24)),
        attention_mask=torch.ones(2, 24, dtype=torch.long),
        action_labels=torch.randint(0, 4, (2, 4)),
        progress_labels=torch.tensor([0.2, 0.8], dtype=torch.float32),
        done_labels=torch.tensor([0.0, 1.0], dtype=torch.float32),
        return_dict=True,
    )

    assert outputs["action_logits"].shape == (2, 4, 4)
    assert outputs["progress_preds"].shape == (2,)
    assert outputs["done_logits"].shape == (2,)
    assert outputs["done_preds"].shape == (2,)
    assert outputs["done_loss"] is not None
    assert outputs["loss"] is not None


@patch("thinkvln.models.thinkvln_actor.ThinkVLNForConditionalGeneration.__init__")
def test_actor_forward_cot_mode_keeps_action_heads_off(mock_parent_init):
    mock_parent_init.return_value = None
    config = _create_mock_config()
    actor = ThinkVLNActor(config, ThinkVLNActorConfig(num_query_tokens=4))

    actor.config = config
    hidden_states = torch.randn(2, 20, 512)
    actor.model = MagicMock(return_value=_DummyOutputs(hidden_states))
    actor.lm_head = MagicMock(return_value=torch.randn(2, 20, 32000))
    actor.loss_function = MagicMock(return_value=torch.tensor(1.5))

    outputs = actor.forward(
        input_ids=torch.randint(0, 32000, (2, 20)),
        attention_mask=torch.ones(2, 20, dtype=torch.long),
        labels=torch.randint(0, 32000, (2, 20)),
        return_dict=True,
    )

    assert outputs["logits"].shape == (2, 20, 32000)
    assert outputs["action_logits"] is None
    assert outputs["progress_preds"] is None
    assert outputs["done_logits"] is None
    assert outputs["done_preds"] is None
