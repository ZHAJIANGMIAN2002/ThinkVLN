"""Flow matching components adapted for ThinkVLN."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Produces sinusoidal embeddings for discretized timesteps."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        bsz, seq_len = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            math.log(10000.0) / max(half_dim, 1)
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)
        if enc.shape[-1] < self.embedding_dim:
            pad = torch.zeros(bsz, seq_len, self.embedding_dim - enc.shape[-1], device=device, dtype=enc.dtype)
            enc = torch.cat([enc, pad], dim=-1)
        return enc


class ActionEncoder(nn.Module):
    """Encode noised waypoint trajectories with timestep conditioning."""

    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.w1 = nn.Linear(action_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size * 2, hidden_size)
        self.w3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == bsz:
            timesteps = timesteps.unsqueeze(1).expand(-1, seq_len)
        else:
            raise ValueError('timesteps must have shape (B,)')

        a_emb = self.w1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.w2(x))
        return self.w3(x)


class ActionDecoder(nn.Module):
    """Decode transformer features into trajectory velocity."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(torch.relu(self.layer1(x)))


class ConditionalDiT(nn.Module):
    """Minimal conditional transformer for flow-matching."""

    def __init__(self, hidden_size: int, output_dim: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.timestep_embed = nn.Embedding(1000, hidden_size)
        self.cond_proj = nn.Linear(hidden_size, hidden_size)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            activation='gelu',
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states.dim() == 2:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

        t = timestep.clamp(min=0, max=999)
        t_emb = self.timestep_embed(t).unsqueeze(1)
        cond = self.cond_proj(encoder_hidden_states)
        tgt = hidden_states + t_emb

        memory_key_padding_mask = None
        if encoder_attention_mask is not None and encoder_attention_mask.dim() == 2:
            memory_key_padding_mask = ~encoder_attention_mask.bool()

        out = self.decoder(
            tgt=tgt,
            memory=cond,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out(out)
