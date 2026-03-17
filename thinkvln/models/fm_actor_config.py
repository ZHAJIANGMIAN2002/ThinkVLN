"""Configuration for flow-matching waypoint actor."""

from dataclasses import dataclass


@dataclass
class FlowMatchingActorConfig:
    """Hyperparameters for flow matching waypoint prediction."""

    action_dim: int = 2
    action_horizon: int = 5
    input_embedding_dim: int = 1536
    decoder_hidden_size: int = 1024
    num_timestep_buckets: int = 100
    num_inference_timesteps: int = 10
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    add_pos_embed: bool = True
    max_seq_len: int = 1024
    flow_loss_weight: float = 1.0
