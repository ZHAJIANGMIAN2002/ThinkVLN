#!/usr/bin/env python3

import torch

from thinkvln.models.fm_actor_config import FlowMatchingActorConfig
from thinkvln.models.fm_components import ActionDecoder, ActionEncoder
from thinkvln.models.thinkvln_fm_actor import FlowMatchingActionHead


def test_action_encoder_decoder_shapes():
    enc = ActionEncoder(action_dim=2, hidden_size=64)
    dec = ActionDecoder(input_dim=64, hidden_dim=64, output_dim=2)
    x = torch.randn(2, 5, 2)
    t = torch.randint(0, 100, (2,))
    h = enc(x, t)
    assert h.shape == (2, 5, 64)
    y = dec(h)
    assert y.shape == (2, 5, 2)


def test_flow_head_train_and_infer_shapes():
    cfg = FlowMatchingActorConfig(
        action_dim=2,
        action_horizon=5,
        input_embedding_dim=64,
        decoder_hidden_size=32,
        num_inference_timesteps=3,
    )
    head = FlowMatchingActionHead(cfg)

    cond = torch.randn(3, 64)
    target = torch.randn(3, 5, 2)
    loss = head(cond, target)
    assert loss.ndim == 0

    with torch.no_grad():
        pred = head.get_action(cond)
    assert pred.shape == (3, 5, 2)
