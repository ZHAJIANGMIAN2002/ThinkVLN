"""ThinkVLN model classes extending Qwen3VL with action query token support"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModelOutputWithPast,
    Qwen3VLCausalLMOutputWithPast,
    auto_docstring,
    check_model_inputs,
)
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache
from transformers.utils import Unpack
from transformers.utils.generic import TransformersKwargs

from .thinkvln_config import ThinkVLNConfig


class ThinkVLNModel(Qwen3VLModel):
    """ThinkVLN model extending Qwen3VL with action query token injection support"""
    
    config_class = ThinkVLNConfig
    
    def __init__(self, config: ThinkVLNConfig):
        super().__init__(config)
        self.config = config
        
        # Placeholder for query embeddings (will be set by parent class)
        self.query_embeddings = None
    
    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        """
        Forward pass with action query token injection.
        
        Action query tokens are injected similar to how vision tokens are processed:
        1. Get input embeddings from input_ids
        2. Process vision tokens (images/videos) using masked_scatter
        3. Process action query tokens using masked_scatter
        4. Forward through language model
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Step 1: Get input embeddings (like Qwen3VL line 1132)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        # Step 2: Process vision tokens (like Qwen3VL lines 1137-1151)
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Step 3: Process action query tokens (NEW - similar to vision processing)
        if self.query_embeddings is not None and input_ids is not None:
            action_query_mask = input_ids == self.config.action_query_token_id
            if action_query_mask.any():
                batch_size = input_ids.shape[0]
                # Expand query embeddings to batch size
                query_embeds = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                # Flatten to match masked_scatter requirements
                query_embeds_flat = query_embeds.reshape(-1, query_embeds.shape[-1])
                # Create expanded mask for embeddings
                action_query_mask_expanded = action_query_mask.unsqueeze(-1).expand_as(inputs_embeds)
                # Inject query embeddings using masked_scatter (like vision tokens)
                inputs_embeds = inputs_embeds.masked_scatter(action_query_mask_expanded, query_embeds_flat)

        # Aggregate visual masks (like Qwen3VL lines 1154-1175)
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # Step 4: Position IDs handling (like Qwen3VL lines 1177-1221)
        # Qwen3VL's get_rope_index handles variable sequence lengths automatically
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Import required function for compilation check
            from transformers.utils import is_torchdynamo_compiling
            
            # Calculate RoPE index in pre-fill stage
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                # Use cached rope_deltas for subsequent forward passes
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Step 5: Forward through language model (like Qwen3VL lines 1223-1233)
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class ThinkVLNForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """ThinkVLN conditional generation model with action query support"""
    
    config_class = ThinkVLNConfig
    
    def __init__(self, config: ThinkVLNConfig):
        # Call grandparent __init__ to avoid Qwen3VL's model initialization
        # We'll create our custom model instead
        super(Qwen3VLForConditionalGeneration, self).__init__(config)
        
        # Create custom ThinkVLN model instead of Qwen3VL model
        self.model = ThinkVLNModel(config)
        
        # LM head (same as Qwen3VL)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        
        # Add learnable query embeddings parameter
        hidden_size = config.text_config.hidden_size
        num_query_tokens = config.num_query_tokens
        self.query_embeddings = nn.Parameter(torch.randn(num_query_tokens, hidden_size) * 0.02)
        
        # Share query embeddings with the model
        self.model.query_embeddings = self.query_embeddings
        
        # Initialize weights
        self.post_init()
    
    # Inherit all other methods from Qwen3VLForConditionalGeneration:
    # - get_input_embeddings, set_input_embeddings
    # - set_decoder, get_decoder
    # - get_video_features, get_image_features
    # - forward (with loss computation)
    # - prepare_inputs_for_generation
    # - _get_image_nums_and_video_nums
    # - _expand_inputs_for_generation
    
    # Make sure to expose language_model and visual properties
    @property
    def language_model(self):
        return self.model.language_model
    
    @property
    def visual(self):
        return self.model.visual
