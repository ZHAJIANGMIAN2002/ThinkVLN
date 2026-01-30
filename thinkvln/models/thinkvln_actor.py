"""ThinkVLNActor model implementation with action classification and progress regression heads"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union
from transformers import AutoConfig

from .actor_config import ThinkVLNActorConfig
from .thinkvln_model import ThinkVLNForConditionalGeneration
from .thinkvln_config import ThinkVLNConfig


class SharedProjector(nn.Module):
    """Shared projection layer with LayerNorm for converting semantic features to control signals"""
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), 
            nn.GELU(), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)


class ActionClassificationHead(nn.Module):
    """MLP head for classifying discrete actions"""
    def __init__(self, hidden_size: int, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
    def forward(self, x): return self.net(x)


class ProgressRegressionHead(nn.Module):
    """MLP head for regressing continuous progress values"""
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)


class ThinkVLNActor(ThinkVLNForConditionalGeneration):
    """ThinkVLN Actor extending ThinkVLN with action/progress prediction heads"""
    
    def __init__(self, config: ThinkVLNConfig, actor_config: Optional[ThinkVLNActorConfig] = None):
        super().__init__(config)
        self.actor_config = actor_config or ThinkVLNActorConfig()
        
        # Get hidden size from config
        hidden_size = config.text_config.hidden_size
        
        # Setup projection layers for action/progress prediction
        proj_size = self.actor_config.projector_hidden_size or hidden_size // 2
        self.shared_projector = SharedProjector(hidden_size, proj_size, 
                                              getattr(self.actor_config, 'projector_dropout', 0.1))
        self.action_head = ActionClassificationHead(proj_size, self.actor_config.num_action_classes, 
                                                  self.actor_config.action_head_dropout)
        self.progress_head = ProgressRegressionHead(proj_size, self.actor_config.progress_head_dropout)
    
    
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, pixel_values=None, pixel_values_videos=None,
                image_grid_thw=None, video_grid_thw=None, cache_position=None, logits_to_keep=0,
                action_labels=None, progress_labels=None, return_dict=None, **kwargs):
        """
        Forward pass with Aux-Think hybrid training support.
        
        Two training modes:
        - Action Task: If action_labels is provided, compute action/progress loss
          (query tokens should already be in input_ids from collator)
        - CoT Task: If action_labels is None, compute only LM loss for chain-of-thought
        
        Note: Query token IDs are added by the data collator, not here.
        The model only handles query token embedding injection (in thinkvln_model.py).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is None:
            raise ValueError("input_ids is required for ThinkVLNActor")
        
        # Store sequence info
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        num_query_tokens = self.config.num_query_tokens
        
        # Determine training mode based on action_labels presence
        use_action_mode = action_labels is not None
        
        # Ensure attention mask is present
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Forward through model
        # Query token embeddings are automatically injected by ThinkVLNModel if query tokens are present in input_ids
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values, 
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw, 
            video_grid_thw=video_grid_thw, 
            position_ids=position_ids,
            attention_mask=attention_mask, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds,
            cache_position=cache_position, 
            **kwargs)
        
        hidden_states = outputs[0]  # [batch, seq_len, hidden_size]
        
        if use_action_mode:
            # ACTION MODE: Only compute action and progress losses
            # No LM loss in action mode
            lm_logits = None
            lm_loss = None
            
            # Action/progress prediction on the last num_query_tokens positions
            # These correspond to the query tokens that were appended by the collator
            query_hidden = hidden_states[:, -num_query_tokens:, :]
            projected = self.shared_projector(query_hidden)
            action_logits = self.action_head(projected)
            progress_values = self.progress_head(projected)
            
            # Compute action and progress losses
            action_loss = nn.CrossEntropyLoss()(
                action_logits.view(-1, action_logits.shape[-1]), 
                action_labels.view(-1)
            ) if action_labels is not None else None
            
            # Compute progress loss with masking for -100 (CoT sample padding)
            if progress_labels is not None:
                valid_mask = progress_labels != -100.0
                if valid_mask.any():
                    progress_loss = nn.MSELoss()(
                        progress_values[valid_mask], 
                        progress_labels[valid_mask]
                    )
                else:
                    progress_loss = None
            else:
                progress_loss = None
            
            # Combine action and progress losses only
            total_loss = torch.tensor(0.0, device=hidden_states.device)
            if action_loss is not None:
                total_loss += self.actor_config.action_loss_weight * action_loss
            if progress_loss is not None:
                total_loss += self.actor_config.progress_loss_weight * progress_loss
        else:
            # COT MODE: Only compute LM loss for chain-of-thought generation
            slice_idx = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            lm_logits = self.lm_head(hidden_states[:, slice_idx, :])
            
            # Compute LM loss
            lm_loss = None
            if labels is not None:
                lm_loss = self.loss_function(
                    logits=lm_logits, 
                    labels=labels, 
                    vocab_size=self.config.text_config.vocab_size
                )
            
            # No action/progress prediction in CoT mode
            action_logits = None
            progress_values = None
            action_loss = None
            progress_loss = None
            
            # Total loss is just LM loss
            total_loss = lm_loss if lm_loss is not None else torch.tensor(0.0, device=hidden_states.device)
        
        if not return_dict:
            return (total_loss, lm_logits, action_logits, progress_values, 
                   outputs.past_key_values, outputs.hidden_states, outputs.attentions)
        
        result = {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "action_loss": action_loss,
            "progress_loss": progress_loss,
            "logits": lm_logits,
            "action_logits": action_logits,
            "progress_values": progress_values,
            "past_key_values": outputs.past_key_values,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions
        }
        if hasattr(outputs, 'rope_deltas'):
            result["rope_deltas"] = outputs.rope_deltas
        return result
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, actor_config=None, *args, **kwargs):
        """Load pretrained Qwen3VL and add actor components"""
        from transformers import Qwen3VLForConditionalGeneration
        
        # Load base Qwen3VL config and convert to ThinkVLNConfig
        base_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        actor_config = actor_config or ThinkVLNActorConfig()
        
        # Create ThinkVLN config from Qwen3VL config
        thinkvln_config = ThinkVLNConfig(
            **base_config.to_dict(),
            num_query_tokens=actor_config.num_query_tokens
        )
        
        # Create ThinkVLN model with new config
        model = cls(thinkvln_config, actor_config)
        
        # Load pretrained Qwen3VL weights
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        
        # Copy weights from base model (skip query_embeddings as they're new)
        model.model.visual.load_state_dict(base_model.model.visual.state_dict())
        model.model.language_model.load_state_dict(base_model.model.language_model.state_dict())
        model.lm_head.load_state_dict(base_model.lm_head.state_dict())
        
        return model
