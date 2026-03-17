"""ThinkVLN actor with flow-matching waypoint head."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Beta
from transformers import AutoConfig

from .fm_actor_config import FlowMatchingActorConfig
from .fm_components import ActionDecoder, ActionEncoder, ConditionalDiT
from .thinkvln_actor import SharedProjector
from .thinkvln_config import ThinkVLNConfig
from .thinkvln_model import ThinkVLNForConditionalGeneration


class FlowMatchingActionHead(nn.Module):
    """Conditional flow-matching head for waypoint trajectories."""

    def __init__(self, config: FlowMatchingActorConfig):
        super().__init__()
        self.config = config
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.num_timestep_buckets = int(config.num_timestep_buckets)
        self.num_inference_timesteps = int(config.num_inference_timesteps)

        self.action_encoder = ActionEncoder(
            action_dim=self.action_dim,
            hidden_size=config.input_embedding_dim,
        )
        self.model = ConditionalDiT(
            hidden_size=config.input_embedding_dim,
            output_dim=config.decoder_hidden_size,
            num_layers=4,
            num_heads=8,
        )
        self.action_decoder = ActionDecoder(
            input_dim=config.decoder_hidden_size,
            hidden_dim=config.decoder_hidden_size,
            output_dim=self.action_dim,
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.position_embedding = None

        alpha = torch.tensor(config.noise_beta_alpha, dtype=torch.float32)
        beta = torch.tensor(config.noise_beta_beta, dtype=torch.float32)
        self.beta_dist = Beta(alpha, beta)

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    @torch.no_grad()
    def get_action(self, conditioning: torch.Tensor) -> torch.Tensor:
        bsz = conditioning.shape[0]
        device = conditioning.device
        actions = torch.randn(
            (bsz, self.action_horizon, self.action_dim),
            dtype=conditioning.dtype,
            device=device,
        )

        dt = 1.0 / float(max(self.num_inference_timesteps, 1))
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(max(self.num_inference_timesteps, 1))
            t_disc = int(t_cont * self.num_timestep_buckets)
            timestep = torch.full((bsz,), t_disc, dtype=torch.long, device=device)

            action_features = self.action_encoder(actions, timestep)
            if self.position_embedding is not None:
                pos = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos).unsqueeze(0)

            model_output = self.model(
                hidden_states=action_features,
                encoder_hidden_states=conditioning,
                timestep=timestep,
            )
            pred_velocity = self.action_decoder(model_output)
            actions = actions + dt * pred_velocity
        return actions

    def forward(self, conditioning: torch.Tensor, target_waypoints: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(target_waypoints)
        t = self.sample_time(target_waypoints.shape[0], target_waypoints.device, target_waypoints.dtype)
        t_view = t[:, None, None]

        noisy = (1.0 - t_view) * noise + t_view * target_waypoints
        velocity = target_waypoints - noise

        t_discretized = (t * self.num_timestep_buckets).long().clamp(min=0, max=self.num_timestep_buckets - 1)
        action_features = self.action_encoder(noisy, t_discretized)
        if self.position_embedding is not None:
            pos = torch.arange(action_features.shape[1], dtype=torch.long, device=target_waypoints.device)
            action_features = action_features + self.position_embedding(pos).unsqueeze(0)

        model_output = self.model(
            hidden_states=action_features,
            encoder_hidden_states=conditioning,
            timestep=t_discretized,
        )
        pred = self.action_decoder(model_output)
        return nn.functional.mse_loss(pred, velocity)


class ThinkVLNFMActor(ThinkVLNForConditionalGeneration):
    """Parallel actor variant that predicts waypoints with flow matching."""

    def __init__(self, config: ThinkVLNConfig, fm_config: Optional[FlowMatchingActorConfig] = None):
        super().__init__(config)
        self.fm_config = fm_config or FlowMatchingActorConfig(
            input_embedding_dim=config.text_config.hidden_size
        )
        hidden_size = config.text_config.hidden_size
        proj_size = self.fm_config.input_embedding_dim

        self.shared_projector = SharedProjector(hidden_size, proj_size, 0.1)
        self.flow_head = FlowMatchingActionHead(self.fm_config)

    @staticmethod
    def _module_param_dtype(module: nn.Module, default: torch.dtype = torch.float32) -> torch.dtype:
        """Best-effort dtype lookup for a module's floating-point parameters."""
        for p in module.parameters():
            if p.is_floating_point():
                return p.dtype
        return default

    def _extract_action_condition(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_query_tokens = int(getattr(self.config, 'num_query_tokens', 4))
        total_query_tokens = 2 * num_query_tokens
        query_hidden = hidden_states[:, -total_query_tokens:, :]
        action_hidden = query_hidden[:, 0::2, :]
        # Always align projector input dtype with projector weights.
        projector_dtype = self._module_param_dtype(self.shared_projector, default=torch.float32)
        action_hidden = action_hidden.to(projector_dtype)
        projected = self.shared_projector(action_hidden)
        return projected.mean(dim=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        waypoint_labels=None,
        return_dict=True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        hidden_states = outputs[0]
        conditioning = self._extract_action_condition(hidden_states)
        flow_dtype = self._module_param_dtype(self.flow_head, default=conditioning.dtype)
        conditioning = conditioning.to(flow_dtype)

        flow_loss = None
        waypoint_preds = None
        if waypoint_labels is not None:
            target = waypoint_labels.to(flow_dtype)
            flow_loss = self.flow_head(conditioning, target)
            total_loss = self.fm_config.flow_loss_weight * flow_loss
        else:
            waypoint_preds = self.flow_head.get_action(conditioning)
            total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        if not return_dict:
            return (total_loss, waypoint_preds)

        return {
            'loss': total_loss,
            'flow_loss': flow_loss,
            'waypoint_preds': waypoint_preds,
            'logits': None,
            'past_key_values': outputs.past_key_values,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, fm_config=None, *args, **kwargs):
        explicit_config = kwargs.pop('config', None)
        if explicit_config is not None:
            base_config = explicit_config
        else:
            try:
                base_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
            except ValueError as exc:
                if 'model type `thinkvln`' not in str(exc):
                    raise
                base_config = ThinkVLNConfig.from_pretrained(pretrained_model_name_or_path)

        if getattr(base_config, 'model_type', None) == ThinkVLNConfig.model_type:
            thinkvln_config = base_config if isinstance(base_config, ThinkVLNConfig) else ThinkVLNConfig(**base_config.to_dict())
        else:
            thinkvln_config = ThinkVLNConfig(**base_config.to_dict())

        model = cls(thinkvln_config, fm_config=fm_config)

        from transformers import Qwen3VLForConditionalGeneration

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model.model.visual.load_state_dict(base_model.model.visual.state_dict())
        model.model.language_model.load_state_dict(base_model.model.language_model.state_dict())
        model.lm_head.load_state_dict(base_model.lm_head.state_dict())
        return model
