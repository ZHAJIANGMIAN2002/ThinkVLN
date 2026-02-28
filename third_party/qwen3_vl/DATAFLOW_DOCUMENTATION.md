# Qwen3VL Data Flow Documentation

## Table of Contents
1. [Overview](#overview)
2. [Input Processing](#input-processing)
3. [Vision Encoding Pipeline](#vision-encoding-pipeline)
4. [Multimodal Embedding Fusion](#multimodal-embedding-fusion)
5. [Position Encoding with MRoPE](#position-encoding-with-mrope)
6. [Text Decoding with DeepStack](#text-decoding-with-deepstack)
7. [Language Modeling Head](#language-modeling-head)
8. [Generation Pipeline](#generation-pipeline)
9. [Key Data Structures](#key-data-structures)

---

## Overview

`Qwen3VLForConditionalGeneration` is the main entry point for the Qwen3-VL multimodal model. It implements a vision-language architecture that processes images/videos alongside text using a novel **DeepStack** mechanism for deep vision-language integration.

**Architecture Components:**
- **Vision Encoder** (`Qwen3VLVisionModel`): Processes images/videos into visual embeddings
- **Language Model** (`Qwen3VLTextModel`): Transformer decoder with DeepStack integration
- **Language Modeling Head** (`lm_head`): Projects hidden states to vocabulary logits

**Main Entry Point:** `modeling_qwen3_vl.py:1535`

---

## Input Processing

### Input Format

The model accepts the following inputs:

```python
forward(
    input_ids: torch.LongTensor,              # (batch_size, seq_len)
    pixel_values: torch.Tensor,               # (total_patches, channels, patch_h, patch_w)
    pixel_values_videos: torch.FloatTensor,   # (total_video_patches, channels, patch_h, patch_w)
    image_grid_thw: torch.LongTensor,         # (num_images, 3) - [temporal, height, width]
    video_grid_thw: torch.LongTensor,         # (num_videos, 3) - [temporal, height, width]
    attention_mask: torch.Tensor,             # (batch_size, seq_len)
    position_ids: torch.LongTensor,           # (3 or 4, batch_size, seq_len)
    labels: torch.LongTensor,                 # (batch_size, seq_len) - for training
    ...
)
```

### Special Tokens

The input text contains special placeholder tokens:
- `<|vision_start|>` (token_id: 151652) - Marks beginning of visual content
- `<|vision_end|>` (token_id: 151653) - Marks end of visual content
- `<|image_pad|>` (token_id: 151655) - Image placeholder tokens
- `<|video_pad|>` (token_id: 151656) - Video placeholder tokens

**Example sequence:**
```
Text <|vision_start|> <|image_pad|> <|image_pad|> ... <|vision_end|> More text
```

### Grid Shape Parameters

`image_grid_thw` and `video_grid_thw` encode the 3D grid structure:
- **T (Temporal)**: Number of frames (1 for images, >1 for videos)
- **H (Height)**: Height in patches after Conv3d
- **W (Width)**: Width in patches after Conv3d

The actual number of tokens per image/video in the LLM sequence:
```
num_tokens = T * (H // spatial_merge_size) * (W // spatial_merge_size)
```
where `spatial_merge_size = 2` by default.

---

## Vision Encoding Pipeline

**Location:** `modeling_qwen3_vl.py:1375-1417` (`Qwen3VLModel.get_image_features`)

### Step 1: Forward Through Vision Encoder

```python
vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
```

**Vision Model Architecture** (`modeling_qwen3_vl.py:811-1006`):

#### 1.1 Patch Embedding
**Location:** `Qwen3VLVisionPatchEmbed` (modeling_qwen3_vl.py:157)

```python
# Input: (num_patches, channels, temporal_patch_size, patch_size, patch_size)
# Conv3d kernel: [temporal_patch_size, patch_size, patch_size]
# Output: (seq_len, hidden_size)
hidden_states = patch_embed(pixel_values)
```

- Uses 3D convolution to extract patches from images/videos
- Default: `patch_size=16`, `temporal_patch_size=2`
- Projects to `hidden_size=1152`

#### 1.2 Position Embeddings
**Location:** `modeling_qwen3_vl.py:920-950`

```python
# Learned 2D position embeddings
pos_embeds = fast_pos_embed_interpolate(grid_thw)  # Bilinear interpolation
hidden_states = hidden_states + pos_embeds

# Rotary position embeddings
rotary_pos_emb = rot_pos_emb(grid_thw)
```

Two types of position embeddings:
1. **Learned 2D positional embeddings**: Added to patch embeddings
2. **Rotary position embeddings (RoPE)**: Used in attention layers

#### 1.3 Vision Transformer Blocks
**Location:** `Qwen3VLVisionBlock` (modeling_qwen3_vl.py:467)

```python
for layer_idx, block in enumerate(blocks):
    hidden_states = block(
        hidden_states,
        cu_seqlens=cu_seqlens,  # Cumulative sequence lengths for variable-length attention
        position_embeddings=(cos, sin)
    )

    # Extract DeepStack features at specific layers
    if layer_idx in deepstack_visual_indexes:  # [8, 16, 24]
        deepstack_features.append(deepstack_merger(hidden_states))
```

Each block contains:
- **Attention** (`Qwen3VLVisionAttention`): Multi-head attention with RoPE
- **MLP** (`Qwen3VLVisionMLP`): Feed-forward network with GELU activation
- **Layer Normalization**: Pre-norm architecture

**Key Feature: Variable-Length Attention**
- Uses `cu_seqlens` (cumulative sequence lengths) to handle multiple images/videos efficiently
- Flash Attention support for memory-efficient computation

#### 1.4 DeepStack Feature Extraction
**Location:** `modeling_qwen3_vl.py:990-997`

```python
deepstack_visual_indexes = [8, 16, 24]  # Extract at layers 8, 16, 24
deepstack_feature_lists = []

for layer_num, block in enumerate(blocks):
    if layer_num in deepstack_visual_indexes:
        # Apply separate merger for each DeepStack layer
        deepstack_feature = deepstack_merger_list[idx](hidden_states)
        deepstack_feature_lists.append(deepstack_feature)
```

- Extracts intermediate features from layers 8, 16, and 24 (out of 27 total)
- Each layer has its own `Qwen3VLVisionPatchMerger`
- These features will be injected into early text decoder layers

#### 1.5 Final Patch Merging
**Location:** `Qwen3VLVisionPatchMerger` (modeling_qwen3_vl.py:201)

```python
# Spatial merging: merge 2x2 patches
merged_hidden_states = merger(hidden_states)

# Process:
# 1. Reshape to (seq_len, hidden_size * spatial_merge_size^2)
# 2. LayerNorm
# 3. Linear projection + GELU
# 4. Linear projection to text hidden size (e.g., 3584)
```

### Step 2: Split Visual Embeddings

**Location:** `modeling_qwen3_vl.py:1412-1416`

```python
# Split merged embeddings back into per-image/video chunks
split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size**2).tolist()
image_embeds = torch.split(merged_embeds, split_sizes)

# Result: List of tensors, one per image/video
# Each tensor shape: (num_tokens, text_hidden_size)
```

### Output Structure

**Returns:** `BaseModelOutputWithDeepstackFeatures`
- `pooler_output`: List of merged visual embeddings (one per image/video)
- `deepstack_features`: List of 3 tensors from intermediate layers
- `last_hidden_state`: Final vision encoder hidden states

---

## Multimodal Embedding Fusion

**Location:** `modeling_qwen3_vl.py:1473-1521` (`Qwen3VLModel.forward`)

### Step 1: Text Embedding Initialization

```python
if inputs_embeds is None:
    inputs_embeds = self.get_input_embeddings()(input_ids)
# Shape: (batch_size, seq_len, hidden_size)
```

### Step 2: Process Images (if present)

```python
if pixel_values is not None:
    # Get visual features
    image_outputs = self.get_image_features(pixel_values, image_grid_thw)
    image_embeds = torch.cat(image_outputs.pooler_output, dim=0)
    deepstack_image_embeds = image_outputs.deepstack_features

    # Create placeholder mask
    image_mask, _ = self.get_placeholder_mask(
        input_ids, inputs_embeds, image_features=image_embeds
    )

    # Replace placeholder tokens with visual embeddings
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

**Mask Creation** (`modeling_qwen3_vl.py:1419-1469`):
```python
# Find positions where input_ids == image_token_id
special_image_mask = (input_ids == self.config.image_token_id)
special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)

# Verify token count matches feature count
assert inputs_embeds[special_image_mask].numel() == image_embeds.numel()
```

### Step 3: Process Videos (if present)

```python
if pixel_values_videos is not None:
    # Same process as images
    video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)
    video_embeds = torch.cat(video_outputs.pooler_output, dim=0)
    deepstack_video_embeds = video_outputs.deepstack_features

    _, video_mask = self.get_placeholder_mask(
        input_ids, inputs_embeds, video_features=video_embeds
    )

    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
```

### Step 4: Aggregate DeepStack Features

**Location:** `modeling_qwen3_vl.py:1494-1521`

```python
# Combine image and video DeepStack features
if image_mask is not None and video_mask is not None:
    visual_pos_masks = image_mask | video_mask  # Union of both masks

    # Merge DeepStack features at each layer
    deepstack_visual_embeds = []
    for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
        embed_joint = torch.zeros_like(...)
        embed_joint[image_mask_joint] = img_embed
        embed_joint[video_mask_joint] = vid_embed
        deepstack_visual_embeds.append(embed_joint)

elif image_mask is not None:
    visual_pos_masks = image_mask
    deepstack_visual_embeds = deepstack_image_embeds

elif video_mask is not None:
    visual_pos_masks = video_mask
    deepstack_visual_embeds = deepstack_video_embeds
```

**Result:**
- `inputs_embeds`: Text embeddings with visual embeddings fused at placeholder positions
- `visual_pos_masks`: Boolean mask indicating which positions contain visual features
- `deepstack_visual_embeds`: List of 3 tensors for injection into decoder layers

---

## Position Encoding with MRoPE

**Location:** `modeling_qwen3_vl.py:972-1087` (`Qwen3VLModel.get_rope_index`)

Qwen3-VL uses **Multi-dimensional Rotary Position Embedding (MRoPE)** with 3 dimensions: temporal (T), height (H), and width (W).

### Position ID Structure

```python
position_ids.shape = (3, batch_size, seq_len)
# Dimension 0: Temporal positions
# Dimension 1: Height positions
# Dimension 2: Width positions
```

### Computation Process

#### Step 1: Handle Video Timestamps
```python
# Videos use timestamps rather than absolute frame positions
if video_grid_thw is not None:
    # Split each video into per-frame entries
    video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
    video_grid_thw[:, 0] = 1  # Each frame has temporal=1
```

#### Step 2: Build Position IDs Per Sample
```python
for i, input_ids in enumerate(batch):
    # Find all vision tokens
    vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)

    llm_pos_ids_list = []
    st = 0  # Start index

    for each vision segment:
        # Text before vision: simple sequential positions
        text_len = ed - st
        st_idx = previous_max + 1
        llm_pos_ids_list.append(
            torch.arange(text_len).expand(3, -1) + st_idx
        )

        # Vision tokens: 3D grid positions
        llm_grid_t = t.item()
        llm_grid_h = h.item() // spatial_merge_size
        llm_grid_w = w.item() // spatial_merge_size

        # Create 3D position indices
        t_index = torch.arange(llm_grid_t).expand(-1, h*w).flatten()
        h_index = torch.arange(llm_grid_h).expand(t, -1, w).flatten()
        w_index = torch.arange(llm_grid_w).expand(t, h, -1).flatten()

        llm_pos_ids_list.append(
            torch.stack([t_index, h_index, w_index]) + text_len + st_idx
        )

        st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    # Remaining text after last vision segment
    if st < len(input_tokens):
        text_len = len(input_tokens) - st
        st_idx = previous_max + 1
        llm_pos_ids_list.append(
            torch.arange(text_len).expand(3, -1) + st_idx
        )

    position_ids[..., i, :] = torch.cat(llm_pos_ids_list, dim=1)
```

#### Step 3: Compute RoPE Deltas
```python
# Delta tracks the offset between actual sequence length and multimodal position
mrope_position_deltas = []
for each sample:
    delta = position_ids.max() + 1 - seq_len
    mrope_position_deltas.append(delta)

mrope_position_deltas = torch.tensor(mrope_position_deltas)
```

### RoPE Application in Text Model

**Location:** `Qwen3VLTextRotaryEmbedding` (modeling_qwen3_vl.py:496)

```python
# position_ids: (3, batch_size, seq_len)
inv_freq_expanded = inv_freq[None, None, :, None].expand(3, batch_size, -1, 1)
position_ids_expanded = position_ids[:, :, None, :].float()

freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)

# Apply interleaved MRoPE
freqs = apply_interleaved_mrope(freqs, mrope_section=[24, 20, 20])

# mrope_section: How to interleave the 3 dimensions
# [24, 20, 20] means:
#   - First 24 dimensions: temporal
#   - Next 20 dimensions: height (interleaved)
#   - Next 20 dimensions: width (interleaved)

emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos()
sin = emb.sin()
```

**Interleaved MRoPE Pattern:**
```
Original: [T, T, T, ..., H, H, H, ..., W, W, W, ...]
After:    [T, H, W, T, H, W, T, H, W, ..., T, T, ...]
```

This interleaving ensures spatial and temporal information are mixed at fine-grained level.

---

## Text Decoding with DeepStack

**Location:** `modeling_qwen3_vl.py:1090-1186` (`Qwen3VLTextModel.forward`)

### Initialization

```python
# Initialize cache for generation
if use_cache and past_key_values is None:
    past_key_values = DynamicCache(config=self.config)

# Get initial embeddings (already fused with visual features)
if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)

# Create causal attention mask
attention_mask = create_causal_mask(
    config=self.config,
    input_embeds=inputs_embeds,
    attention_mask=attention_mask,
    cache_position=cache_position,
    past_key_values=past_key_values,
    position_ids=text_position_ids
)

# Create position embeddings (cos, sin for RoPE)
position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

### Layer-by-Layer Processing

```python
hidden_states = inputs_embeds

for layer_idx, decoder_layer in enumerate(self.layers):
    # Standard transformer decoder layer
    layer_outputs = decoder_layer(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=text_position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        position_embeddings=position_embeddings
    )
    hidden_states = layer_outputs

    # DeepStack: Inject visual features into early layers
    if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
        hidden_states = _deepstack_process(
            hidden_states,
            visual_pos_masks,
            deepstack_visual_embeds[layer_idx]
        )
```

### DeepStack Processing

**Location:** `modeling_qwen3_vl.py:1178-1186`

```python
def _deepstack_process(hidden_states, visual_pos_masks, visual_embeds):
    """
    Add visual features to text hidden states at visual token positions.

    Args:
        hidden_states: (batch_size, seq_len, hidden_size)
        visual_pos_masks: (batch_size, seq_len) - boolean mask
        visual_embeds: (num_visual_tokens, hidden_size)
    """
    hidden_states = hidden_states.clone()

    # Add visual features to positions marked by mask
    hidden_states[visual_pos_masks, :] += visual_embeds

    return hidden_states
```

**Key Points:**
- DeepStack features are **added** to (not replacing) hidden states
- Only applied at visual token positions
- Happens at layers 0, 1, 2 (corresponding to vision layers 8, 16, 24)
- Enables deep multimodal interaction throughout the network

### Decoder Layer Structure

**Location:** `Qwen3VLTextDecoderLayer` (modeling_qwen3_vl.py:700)

```python
# Pre-norm architecture
residual = hidden_states
hidden_states = input_layernorm(hidden_states)

# Self-attention with MRoPE
hidden_states, _ = self_attn(
    hidden_states,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values,
    position_embeddings=position_embeddings  # (cos, sin)
)
hidden_states = residual + hidden_states

# Feed-forward network
residual = hidden_states
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = mlp(hidden_states)  # SwiGLU activation
hidden_states = residual + hidden_states
```

**Attention with QK-Norm:**
```python
# Query and key normalization for stability
query_states = q_norm(q_proj(hidden_states))
key_states = k_norm(k_proj(hidden_states))
value_states = v_proj(hidden_states)

# Apply RoPE
query_states, key_states = apply_rotary_pos_emb(
    query_states, key_states, cos, sin
)
```

### Final Normalization

```python
hidden_states = self.norm(hidden_states)  # RMSNorm

return BaseModelOutputWithPast(
    last_hidden_state=hidden_states,
    past_key_values=past_key_values
)
```

---

## Language Modeling Head

**Location:** `modeling_qwen3_vl.py:1375-1466` (`Qwen3VLForConditionalGeneration.forward`)

### Forward Pass

```python
# Get hidden states from text model (with visual features integrated)
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
    cache_position=cache_position
)

hidden_states = outputs[0]  # (batch_size, seq_len, hidden_size)
```

### Compute Logits

```python
# Only compute logits for necessary positions (efficiency during generation)
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
# Shape: (batch_size, kept_seq_len, vocab_size)
```

**Note on `logits_to_keep`:**
- During generation: Only compute logits for the last token (`logits_to_keep=1`)
- During training: Compute all logits (`logits_to_keep=0` means all)
- Saves computation and memory

### Compute Loss (Training)

```python
if labels is not None:
    loss = self.loss_function(
        logits=logits,
        labels=labels,
        vocab_size=self.config.text_config.vocab_size
    )
```

Uses cross-entropy loss with label smoothing (if configured).

### Return Output

```python
return Qwen3VLCausalLMOutputWithPast(
    loss=loss,
    logits=logits,
    past_key_values=outputs.past_key_values,
    hidden_states=outputs.hidden_states,
    attentions=outputs.attentions,
    rope_deltas=outputs.rope_deltas
)
```

---

## Generation Pipeline

**Location:** `modeling_qwen3_vl.py:1468-1531` (`prepare_inputs_for_generation`)

### Prefill Stage (First Forward Pass)

```python
if cache_position[0] == 0 or self.model.rope_deltas is None:
    # Calculate position IDs with vision-aware MRoPE
    position_ids, rope_deltas = self.model.get_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask
    )
    self.model.rope_deltas = rope_deltas  # Cache for subsequent tokens

    # Concatenate text + vision position IDs
    text_positions = model_inputs["position_ids"][None, ...]
    vision_positions = position_ids  # (3, batch_size, seq_len)
    model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)
    # Final shape: (4, batch_size, seq_len)
```

### Decode Stage (Subsequent Tokens)

```python
else:
    # Use cached rope_deltas for incremental position
    batch_size, seq_length = model_inputs["position_ids"].shape
    position_ids = torch.arange(seq_length, device=device)
    position_ids = position_ids.view(1, -1).expand(batch_size, -1)

    # Apply cached delta
    delta = cache_position[0] + self.model.rope_deltas
    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
    position_ids = position_ids.add(delta)

    # Expand to 3D for MRoPE
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # Remove visual inputs (already processed in prefill)
    model_inputs["pixel_values"] = None
    model_inputs["pixel_values_videos"] = None
```

### Beam Search / Sampling Expansion

**Location:** `modeling_qwen3_vl.py:1574-1684` (`_expand_inputs_for_generation`)

When using beam search or multiple samples, all inputs must be expanded:

```python
def _expand_inputs_for_generation(self, expand_size, input_ids, **model_kwargs):
    if expand_size == 1:
        return input_ids, model_kwargs

    # Visual inputs need special handling
    for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
        if key == "pixel_values":
            # Split by sample, then repeat each sample
            samples = torch.split(image_grid_thw, list(image_nums))
            lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
            dict_to_expand[key] = _repeat_interleave_samples(
                dict_to_expand[key], lengths=lengths, repeat_times=expand_size
            )
        # Similar for other visual keys...

    # Regular text inputs: simple repeat_interleave
    input_ids = input_ids.repeat_interleave(expand_size, dim=0)
    attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)

    return input_ids, model_kwargs
```

### Generation Flow Summary

```
Initial Input:
  ├─ Text tokens + visual placeholders
  ├─ pixel_values / pixel_values_videos
  └─ grid_thw metadata

Prefill Stage:
  ├─ Process vision (once)
  ├─ Fuse embeddings
  ├─ Calculate all position_ids
  ├─ Forward through decoder
  └─ Cache KV and rope_deltas

Decode Loop (for each new token):
  ├─ Use previous KV cache
  ├─ Compute position from rope_deltas
  ├─ Forward only last token
  ├─ Sample next token
  └─ Update KV cache

Output:
  └─ Generated token IDs
```

---

## Key Data Structures

### Tensor Shapes Throughout Pipeline

```
Input Stage:
  input_ids:              (batch_size, seq_len)
  pixel_values:           (total_patches, C, T*P, P, P)  # Flattened across all images
  image_grid_thw:         (num_images, 3)
  attention_mask:         (batch_size, seq_len)

Vision Encoding:
  patch_embeddings:       (total_patches, hidden_size)
  vision_hidden_states:   (total_patches, hidden_size)
  pooler_output:          List[(image_tokens, text_hidden_size)]
  deepstack_features:     List[3][(visual_tokens, text_hidden_size)]

Embedding Fusion:
  inputs_embeds:          (batch_size, seq_len, text_hidden_size)
  visual_pos_masks:       (batch_size, seq_len)  # Boolean mask

Position Encoding:
  position_ids:           (3 or 4, batch_size, seq_len)
                          # [T, H, W] or [text, T, H, W]
  rope_deltas:            (batch_size, 1)

Text Decoding:
  hidden_states:          (batch_size, seq_len, text_hidden_size)
  attention_mask:         (batch_size, 1, seq_len, past_seq_len + seq_len)
  position_embeddings:    (cos: (bs, seq_len, head_dim), sin: (bs, seq_len, head_dim))
  past_key_values:        Cache object with KV for each layer

Output:
  logits:                 (batch_size, seq_len, vocab_size)
  loss:                   (1,)  # If labels provided
```

### Important Configuration Parameters

```python
# Vision Config
vision_config:
  depth: 27                           # Number of vision transformer layers
  hidden_size: 1152                   # Vision hidden dimension
  num_heads: 16                       # Vision attention heads
  patch_size: 16                      # Spatial patch size
  temporal_patch_size: 2              # Temporal patch size for videos
  spatial_merge_size: 2               # Merge 2x2 patches before LLM
  out_hidden_size: 3584               # Output dimension (matches text)
  deepstack_visual_indexes: [8, 16, 24]  # DeepStack feature layers

# Text Config
text_config:
  vocab_size: 151936                  # Vocabulary size
  hidden_size: 3584                   # Text hidden dimension
  intermediate_size: 18944            # MLP intermediate dimension
  num_hidden_layers: 28               # Number of decoder layers
  num_attention_heads: 28             # Attention heads
  num_key_value_heads: 4              # KV heads (GQA)
  head_dim: 128                       # Dimension per head
  max_position_embeddings: 128000     # Max sequence length
  rope_parameters:
    rope_type: "default"
    rope_theta: 1000000.0             # RoPE frequency base
    mrope_section: [24, 20, 20]       # MRoPE dimension allocation

# Special Tokens
image_token_id: 151655                # <|image_pad|>
video_token_id: 151656                # <|video_pad|>
vision_start_token_id: 151652         # <|vision_start|>
vision_end_token_id: 151653           # <|vision_end|>
```

### Cache Structure

```python
class DynamicCache:
    """
    Stores past key-value pairs for efficient generation.
    """
    key_cache: List[torch.Tensor]     # [(batch_size, num_kv_heads, past_len, head_dim)]
    value_cache: List[torch.Tensor]   # [(batch_size, num_kv_heads, past_len, head_dim)]

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        """Append new KV to cache and return concatenated result."""
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
```

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT PROCESSING                           │
├─────────────────────────────────────────────────────────────────────┤
│  Text: "Describe <|vision_start|><|image_pad|>...<|vision_end|>"   │
│  pixel_values: (patches, 3, 32, 16, 16)                            │
│  image_grid_thw: [(1, 256, 256)]                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       VISION ENCODING                                │
├─────────────────────────────────────────────────────────────────────┤
│  PatchEmbed (Conv3d) → (seq_len, 1152)                             │
│       ↓                                                              │
│  + Position Embeddings (Learned + RoPE)                             │
│       ↓                                                              │
│  Vision Transformer (27 layers)                                     │
│       ├─ Layer 8  ──→ DeepStack Feature 1                          │
│       ├─ Layer 16 ──→ DeepStack Feature 2                          │
│       ├─ Layer 24 ──→ DeepStack Feature 3                          │
│       ↓                                                              │
│  PatchMerger → (visual_tokens, 3584)                                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING FUSION                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Text Embeddings: (batch, seq_len, 3584)                           │
│       ↓                                                              │
│  Find <|image_pad|> positions                                       │
│       ↓                                                              │
│  Replace with Visual Embeddings                                     │
│       ↓                                                              │
│  Fused Embeddings: (batch, seq_len, 3584)                          │
│  Visual Masks: (batch, seq_len) [Boolean]                          │
│  DeepStack Features: List[3] × (visual_tokens, 3584)               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   POSITION ENCODING (MRoPE)                          │
├─────────────────────────────────────────────────────────────────────┤
│  Build 3D Position IDs:                                             │
│    - Text regions: [pos, pos, pos] (same for T,H,W)                │
│    - Visual regions: [t, h, w] (grid indices)                       │
│       ↓                                                              │
│  position_ids: (3, batch, seq_len)                                  │
│       ↓                                                              │
│  Compute RoPE: cos, sin embeddings                                  │
│       ↓                                                              │
│  Apply Interleaved MRoPE: [T,H,W,T,H,W,...]                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   TEXT DECODING (28 Layers)                          │
├─────────────────────────────────────────────────────────────────────┤
│  For layer_idx in [0..27]:                                          │
│    ┌────────────────────────────────────────┐                      │
│    │  LayerNorm                              │                      │
│    │    ↓                                    │                      │
│    │  Q,K,V Projections + QK-Norm          │                      │
│    │    ↓                                    │                      │
│    │  Apply MRoPE (cos, sin)                │                      │
│    │    ↓                                    │                      │
│    │  Self-Attention (with KV cache)        │                      │
│    │    ↓                                    │                      │
│    │  + Residual                             │                      │
│    └────────────────────────────────────────┘                      │
│              ↓                                                       │
│    ┌────────────────────────────────────────┐                      │
│    │  LayerNorm                              │                      │
│    │    ↓                                    │                      │
│    │  SwiGLU MLP                             │                      │
│    │    ↓                                    │                      │
│    │  + Residual                             │                      │
│    └────────────────────────────────────────┘                      │
│              ↓                                                       │
│    ┌────────────────────────────────────────┐                      │
│    │  If layer_idx in [0,1,2]:              │                      │
│    │    Add DeepStack Visual Features       │                      │
│    │    at visual token positions           │                      │
│    └────────────────────────────────────────┘                      │
│                                                                      │
│  Final LayerNorm                                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LANGUAGE MODELING HEAD                             │
├─────────────────────────────────────────────────────────────────────┤
│  hidden_states: (batch, seq_len, 3584)                             │
│       ↓                                                              │
│  Linear Projection                                                   │
│       ↓                                                              │
│  logits: (batch, seq_len, 151936)                                  │
│       ↓                                                              │
│  Cross-Entropy Loss (if training)                                   │
│       ↓                                                              │
│  Sampling / Beam Search (if generating)                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         OUTPUT TOKENS
```

---

## Summary

The Qwen3VL data flow implements a sophisticated multimodal architecture with these key innovations:

1. **Deep Vision Processing**: 27-layer vision transformer with multi-scale feature extraction
2. **DeepStack Integration**: Visual features from layers 8, 16, 24 are injected into early decoder layers for deep multimodal fusion
3. **Multi-dimensional RoPE (MRoPE)**: 3D position encoding (temporal, height, width) with interleaved pattern for spatial-temporal awareness
4. **Efficient Token Fusion**: Direct replacement of placeholder tokens with visual embeddings
5. **Unified Generation**: Single autoregressive decoder handles both text and multimodal understanding

The model processes images/videos through a dedicated vision encoder, fuses visual embeddings into the text sequence, and uses a transformer decoder with DeepStack to generate text outputs with deep vision-language understanding.
