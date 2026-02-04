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


# OLD: Simple single-layer heads (commented out, kept for reference)
# class ActionClassificationHead(nn.Module):
#     """MLP head for classifying discrete actions"""
#     def __init__(self, hidden_size: int, num_classes: int = 4, dropout: float = 0.1):
#         super().__init__()
#         self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
#     def forward(self, x): return self.net(x)
# 
# 
# class ProgressRegressionHead(nn.Module):
#     """MLP head for regressing continuous progress values"""
#     def __init__(self, hidden_size: int, dropout: float = 0.1):
#         super().__init__()
#         self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1), nn.Sigmoid())
#     def forward(self, x): return self.net(x).squeeze(-1)


# NEW: OpenVLA-OFT style two-layer MLP heads with residual connections
class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection (from OpenVLA-OFT)"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # Pre-Layer Normalization with residual connection
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks (from OpenVLA-OFT)"""
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class ActionClassificationHeadV2(nn.Module):
    """Two-layer MLP head for classifying discrete actions (OpenVLA-OFT style)"""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 4, num_blocks: int = 2):
        super().__init__()
        self.mlp = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes
        )
    
    def forward(self, x):
        # x: (batch_size, num_query_tokens, input_dim)
        batch_size, num_tokens, _ = x.shape
        # Flatten batch and token dimensions
        x_flat = x.reshape(batch_size * num_tokens, -1)
        # Apply MLP
        logits = self.mlp(x_flat)
        # Reshape back
        logits = logits.reshape(batch_size, num_tokens, -1)
        return logits


class ProgressRegressionHeadV2(nn.Module):
    """Two-layer MLP head for regressing progress (OpenVLA-OFT style, no sigmoid)"""
    def __init__(self, input_dim: int, hidden_dim: int, num_blocks: int = 2):
        super().__init__()
        self.mlp = MLPResNet(
            num_blocks=num_blocks,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1  # Single output for progress regression
        )
    
    def forward(self, x):
        # x: (batch_size, num_query_tokens, input_dim)
        batch_size, num_tokens, _ = x.shape
        # Flatten batch and token dimensions
        x_flat = x.reshape(batch_size * num_tokens, -1)
        # Apply MLP (no sigmoid, direct linear regression)
        progress = self.mlp(x_flat)
        # Reshape and squeeze
        progress = progress.reshape(batch_size, num_tokens)
        return progress


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
        
        # OLD: Simple single-layer heads (commented out)
        # self.action_head = ActionClassificationHead(proj_size, self.actor_config.num_action_classes, 
        #                                           self.actor_config.action_head_dropout)
        # self.progress_head = ProgressRegressionHead(proj_size, self.actor_config.progress_head_dropout)
        
        # NEW: Two-layer MLP heads (OpenVLA-OFT style)
        self.action_head = ActionClassificationHeadV2(
            input_dim=proj_size,
            hidden_dim=proj_size,
            num_classes=self.actor_config.num_action_classes,
            num_blocks=2
        )
        self.progress_head = ProgressRegressionHeadV2(
            input_dim=proj_size,
            hidden_dim=proj_size,
            num_blocks=2
        )
    
    
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
            
            # OLD: Single query type, all positions for both heads
            # query_hidden = hidden_states[:, -num_query_tokens:, :]
            # head_dtype = next(self.shared_projector.parameters()).dtype
            # query_hidden = query_hidden.to(head_dtype)
            # projected = self.shared_projector(query_hidden)
            # action_logits = self.action_head(projected)
            # progress_values = self.progress_head(projected)
            
            # NEW: Two query types (action and progress), split by position
            # Total query length is 2 * num_query_tokens (interleaved: action_0, progress_0, action_1, progress_1, ...)
            total_query_tokens = 2 * num_query_tokens
            query_hidden = hidden_states[:, -total_query_tokens:, :]  # [batch, 2*K, hidden_size]
            
            # Split by position: even indices (0,2,4,...) are action queries, odd indices (1,3,5,...) are progress queries
            action_hidden = query_hidden[:, 0::2, :]  # [batch, K, hidden_size]
            progress_hidden = query_hidden[:, 1::2, :]  # [batch, K, hidden_size]
            
            # Project each through the shared projector
            head_dtype = next(self.shared_projector.parameters()).dtype
            action_hidden = action_hidden.to(head_dtype)
            progress_hidden = progress_hidden.to(head_dtype)
            
            action_projected = self.shared_projector(action_hidden)  # [batch, K, proj_size]
            progress_projected = self.shared_projector(progress_hidden)  # [batch, K, proj_size]
            
            # Apply respective heads
            action_logits = self.action_head(action_projected)  # [batch, K, num_classes]
            progress_values = self.progress_head(progress_projected)  # [batch, K]
            
            # Compute action and progress losses
            action_loss = nn.CrossEntropyLoss()(
                action_logits.view(-1, action_logits.shape[-1]), 
                action_labels.view(-1)
            ) if action_labels is not None else None
            
            # Compute progress loss with masking for -100 (CoT sample padding)
            if progress_labels is not None:
                valid_mask = progress_labels != -100.0
                if valid_mask.any():
                    if getattr(self.actor_config, 'use_huber_loss_for_progress', False):
                        progress_loss = nn.SmoothL1Loss()(
                            progress_values[valid_mask], progress_labels[valid_mask]
                        )
                    else:
                        progress_loss = nn.MSELoss()(
                            progress_values[valid_mask], progress_labels[valid_mask]
                        )
                else:
                    progress_loss = None
            else:
                progress_loss = None

            # Combine action and progress losses (dtype for mixed precision)
            total_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
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
            
            # Total loss is just LM loss (dtype for mixed precision)
            total_loss = lm_loss if lm_loss is not None else torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        
        if not return_dict:
            return (total_loss, lm_logits, action_logits, progress_values,  # Note: tuple uses progress_values
                   outputs.past_key_values, outputs.hidden_states, outputs.attentions)
        
        result = {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "action_loss": action_loss,
            "progress_loss": progress_loss,
            "logits": lm_logits,
            "action_logits": action_logits,
            "progress_preds": progress_values,  # Use 'progress_preds' for consistency with evaluation code
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
        
        # OLD: Initialize query_embeddings with "Action" token embedding (commented out for zero initialization)
        # with torch.no_grad():
        #     try:
        #         from transformers import AutoTokenizer
        #         tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        #         action_token_id = tokenizer.encode("Action", add_special_tokens=False)[0]
        #     except:
        #         action_token_id = 4227  # Fallback default for Qwen3VL
        #     
        #     embedding_layer = model.model.get_input_embeddings()
        #     action_embedding = embedding_layer.weight[action_token_id]
        #     model.query_embeddings.data.copy_(action_embedding.unsqueeze(0).repeat(actor_config.num_query_tokens, 1))
        
        # NEW: Query embeddings are zero-initialized buffers, no need to initialize from token embeddings
        
        return model
