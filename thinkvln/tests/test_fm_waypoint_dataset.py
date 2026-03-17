#!/usr/bin/env python3

import torch

from thinkvln.dataset.fm_waypoint_dataset import extract_delta_waypoints


def test_extract_delta_waypoints_basic():
    positions = [
        [0.0, 0.0],
        [1.0, 0.5],
        [2.5, 1.0],
        [3.0, 1.5],
    ]
    out = extract_delta_waypoints(frame_idx=1, positions=positions, horizon=3, action_dim=2)
    assert out.shape == (3, 2)
    assert torch.allclose(out[0], torch.tensor([1.5, 0.5]))
    assert torch.allclose(out[2], torch.tensor([2.0, 1.0]))


def test_extract_delta_waypoints_pad_tail():
    positions = [[0.0, 0.0], [0.2, 0.1]]
    out = extract_delta_waypoints(frame_idx=1, positions=positions, horizon=4, action_dim=2)
    assert out.shape == (4, 2)
    assert torch.allclose(out, torch.zeros_like(out))
