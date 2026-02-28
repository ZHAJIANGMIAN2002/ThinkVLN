# Qwen3-VL 主要类结构及关系说明

## 目录
1. [架构概览](#架构概览)
2. [配置类](#配置类)
3. [视觉编码器类](#视觉编码器类)
4. [文本解码器类](#文本解码器类)
5. [统一模型类](#统一模型类)
6. [输出数据类](#输出数据类)
7. [类继承关系图](#类继承关系图)
8. [数据流关系图](#数据流关系图)

---

## 架构概览

Qwen3-VL 采用**视觉-语言融合架构**，主要由三大部分组成：

```
Qwen3VLForConditionalGeneration (主入口)
    │
    ├─── Qwen3VLModel (核心模型)
    │       ├─── Qwen3VLVisionModel (视觉编码器)
    │       └─── Qwen3VLTextModel (文本解码器)
    │
    └─── lm_head (语言建模头)
```

---

## 配置类

这些类定义在 `configuration_qwen3_vl.py` 中。

### 1. Qwen3VLVisionConfig
**位置:** `configuration_qwen3_vl.py:37`

**作用:** 配置视觉编码器的参数

**关键参数:**
```python
class Qwen3VLVisionConfig(PreTrainedConfig):
    depth: int = 27                    # 视觉 Transformer 层数
    hidden_size: int = 1152            # 视觉隐藏层维度
    num_heads: int = 16                # 注意力头数
    patch_size: int = 16               # 空间 patch 大小
    temporal_patch_size: int = 2       # 时间 patch 大小（视频）
    spatial_merge_size: int = 2        # 空间合并大小
    out_hidden_size: int = 3584        # 输出维度（对齐文本维度）
    deepstack_visual_indexes: list = [8, 16, 24]  # DeepStack 特征提取层
```

### 2. Qwen3VLTextConfig
**位置:** `configuration_qwen3_vl.py:76`

**作用:** 配置文本解码器的参数

**关键参数:**
```python
class Qwen3VLTextConfig(PreTrainedConfig):
    vocab_size: int = 151936           # 词汇表大小
    hidden_size: int = 4096            # 文本隐藏层维度
    num_hidden_layers: int = 32        # 解码器层数
    num_attention_heads: int = 32      # 注意力头数
    num_key_value_heads: int = 32      # KV 头数（GQA）
    max_position_embeddings: int = 128000  # 最大序列长度
    rope_parameters: dict              # RoPE 配置
```

### 3. Qwen3VLConfig
**位置:** `configuration_qwen3_vl.py:206`

**作用:** 整体模型配置，组合视觉和文本配置

**组成:**
```python
class Qwen3VLConfig(PreTrainedConfig):
    vision_config: Qwen3VLVisionConfig    # 视觉编码器配置
    text_config: Qwen3VLTextConfig        # 文本解码器配置

    # 特殊 token ID
    image_token_id: int = 151655          # <|image_pad|>
    video_token_id: int = 151656          # <|video_pad|>
    vision_start_token_id: int = 151652   # <|vision_start|>
    vision_end_token_id: int = 151653     # <|vision_end|>
```

---

## 视觉编码器类

这些类组成视觉编码器，负责处理图像和视频输入。

### 1. Qwen3VLVisionModel
**位置:** `modeling_qwen3_vl.py:613`
**继承:** `Qwen3VLPreTrainedModel`

**作用:** 视觉编码器的主类，协调所有视觉处理组件

**主要组件:**
```python
class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    def __init__(self, config):
        # 图像分块嵌入
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)

        # 学习的位置嵌入
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)

        # 旋转位置嵌入
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        # Vision Transformer 层
        self.blocks = nn.ModuleList([
            Qwen3VLVisionBlock(config) for _ in range(config.depth)
        ])

        # 最终的 patch 合并器
        self.merger = Qwen3VLVisionPatchMerger(config)

        # DeepStack 特征合并器列表（3个）
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLVisionPatchMerger(config)
            for _ in range(len(config.deepstack_visual_indexes))
        ])
```

**关键方法:**
- `forward()`: 主前向传播，输出视觉特征
- `rot_pos_emb()`: 计算旋转位置嵌入
- `fast_pos_embed_interpolate()`: 位置嵌入插值

**输出:**
```python
BaseModelOutputWithDeepstackFeatures(
    last_hidden_state: 最终隐藏状态,
    pooler_output: 合并后的视觉嵌入（列表），
    deepstack_features: DeepStack 特征（3层）
)
```

---

### 2. Qwen3VLVisionPatchEmbed
**位置:** `modeling_qwen3_vl.py:70`
**继承:** `nn.Module`

**作用:** 将图像/视频转换为 patch 嵌入

**实现:**
```python
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        # 3D 卷积：[temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels=config.in_channels,    # 3 (RGB)
            out_channels=config.hidden_size,   # 1152
            kernel_size=[config.temporal_patch_size, config.patch_size, config.patch_size],
            stride=[config.temporal_patch_size, config.patch_size, config.patch_size],
            bias=True
        )

    def forward(self, hidden_states):
        # 输入: (num_patches, 3, temporal_patch_size, patch_size, patch_size)
        # 输出: (seq_len, hidden_size)
        return self.proj(hidden_states).view(-1, self.embed_dim)
```

**用途:** 提取图像的空间特征和视频的时空特征

---

### 3. Qwen3VLVisionBlock
**位置:** `modeling_qwen3_vl.py:264`
**继承:** `GradientCheckpointingLayer`

**作用:** Vision Transformer 的单个块

**组成:**
```python
class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config):
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config)  # 多头注意力
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = Qwen3VLVisionMLP(config)         # 前馈网络

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        # Pre-norm 架构
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), ...)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
```

**特点:** 使用 pre-norm 架构，支持可变长度注意力（cu_seqlens）

---

### 4. Qwen3VLVisionAttention
**位置:** `modeling_qwen3_vl.py:181`
**继承:** `nn.Module`

**作用:** 视觉编码器的多头注意力

**实现:**
```python
class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config):
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        # 1. QKV 投影
        q, k, v = self.qkv(hidden_states).split(...)

        # 2. 应用旋转位置嵌入
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # 3. 注意力计算（支持 Flash Attention）
        attn_output = attention_interface(q, k, v, ...)

        # 4. 输出投影
        return self.proj(attn_output)
```

**支持:** Flash Attention, SDPA (Scaled Dot-Product Attention)

---

### 5. Qwen3VLVisionMLP
**位置:** `modeling_qwen3_vl.py:57`
**继承:** `nn.Module`

**作用:** 视觉编码器的前馈网络

**实现:**
```python
class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]  # GELU

    def forward(self, hidden_state):
        # hidden_size (1152) → intermediate_size (4304) → hidden_size (1152)
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
```

---

### 6. Qwen3VLVisionPatchMerger
**位置:** `modeling_qwen3_vl.py:106`
**继承:** `nn.Module`

**作用:** 合并视觉 patches 并投影到文本维度

**实现:**
```python
class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm=False):
        # spatial_merge_size = 2, 合并 2x2 patches
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(self.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x):
        # 输入: (seq_len, vision_hidden_size)
        # 合并 2x2: (seq_len/4, vision_hidden_size * 4)
        # 输出: (seq_len/4, text_hidden_size)
        x = self.norm(x.view(-1, self.hidden_size))
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x
```

**用途:**
- 主 merger: 生成最终的视觉嵌入
- DeepStack mergers: 为 DeepStack 生成中间层特征

---

### 7. Qwen3VLVisionRotaryEmbedding
**位置:** `modeling_qwen3_vl.py:90`
**继承:** `nn.Module`

**作用:** 为视觉编码器生成旋转位置嵌入（RoPE）

**实现:**
```python
class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen):
        # 生成 cos 和 sin 嵌入
        seq = torch.arange(seqlen, device=self.inv_freq.device)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs  # 用于后续计算 cos/sin
```

---

## 文本解码器类

这些类组成文本解码器，负责处理文本和融合后的多模态序列。

### 1. Qwen3VLTextModel
**位置:** `modeling_qwen3_vl.py:823`
**继承:** `Qwen3VLPreTrainedModel`

**作用:** 文本解码器的主类，支持 DeepStack 视觉特征注入

**主要组件:**
```python
class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    def __init__(self, config):
        # Token 嵌入
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx
        )

        # 解码器层
        self.layers = nn.ModuleList([
            Qwen3VLTextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # 最终归一化
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MRoPE（多维旋转位置嵌入）
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
```

**关键方法:**
```python
def forward(
    self,
    input_ids=None,
    inputs_embeds=None,              # 已融合视觉特征的嵌入
    visual_pos_masks=None,           # 视觉 token 位置掩码
    deepstack_visual_embeds=None,    # DeepStack 特征列表
    ...
):
    # 1. 获取初始嵌入
    hidden_states = inputs_embeds

    # 2. 逐层处理
    for layer_idx, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(hidden_states, ...)

        # 3. DeepStack: 在前几层注入视觉特征
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx]
            )

    # 4. 最终归一化
    hidden_states = self.norm(hidden_states)
    return hidden_states
```

**DeepStack 处理:**
```python
def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
    """在视觉 token 位置添加 DeepStack 特征"""
    hidden_states = hidden_states.clone()
    hidden_states[visual_pos_masks, :] += visual_embeds
    return hidden_states
```

---

### 2. Qwen3VLTextDecoderLayer
**位置:** `modeling_qwen3_vl.py:519`
**继承:** `GradientCheckpointingLayer`

**作用:** 文本解码器的单个层

**组成:**
```python
class Qwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        self.hidden_size = config.hidden_size

        # 自注意力
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)

        # 前馈网络
        self.mlp = Qwen3VLTextMLP(config)

        # 归一化层
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_embeddings, ...):
        # Pre-norm 架构

        # 1. 自注意力
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            ...
        )
        hidden_states = residual + hidden_states

        # 2. 前馈网络
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

---

### 3. Qwen3VLTextAttention
**位置:** `modeling_qwen3_vl.py:428`
**继承:** `nn.Module`

**作用:** 文本解码器的多头注意力（支持 GQA）

**实现:**
```python
class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        # QKV 投影
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

        # QK 归一化（提高训练稳定性）
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, past_key_values, ...):
        # 1. QKV 投影 + 归一化
        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        # 2. 应用 MRoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 3. 更新 KV 缓存
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, ...
            )

        # 4. 注意力计算
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, ...
        )

        # 5. 输出投影
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

**特点:**
- **Grouped Query Attention (GQA)**: 减少 KV 缓存大小
- **QK Normalization**: 提高训练稳定性
- **KV Cache**: 支持高效生成

---

### 4. Qwen3VLTextMLP
**位置:** `modeling_qwen3_vl.py:503`
**继承:** `nn.Module`

**作用:** 文本解码器的前馈网络（SwiGLU）

**实现:**
```python
class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # SwiGLU 需要两个门控投影
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]  # SiLU

    def forward(self, x):
        # SwiGLU: (W_gate(x) ⊙ σ(x)) * W_up(x)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**说明:** SwiGLU 是一种门控前馈网络，性能优于标准 FFN

---

### 5. Qwen3VLTextRotaryEmbedding
**位置:** `modeling_qwen3_vl.py:291`
**继承:** `nn.Module`

**作用:** 为文本解码器生成多维旋转位置嵌入（MRoPE）

**实现:**
```python
class Qwen3VLTextRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        # 基础 RoPE 参数
        base = config.rope_parameters["rope_theta"]  # 1000000.0
        dim = config.head_dim

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # MRoPE 配置：[temporal, height, width]
        self.mrope_section = config.rope_parameters.get("mrope_section", [24, 20, 20])

    def forward(self, x, position_ids):
        # position_ids: (3, batch_size, seq_len) - [T, H, W]

        # 1. 计算每个维度的频率
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, bs, -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)

        # 2. 应用交错 MRoPE
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        # 3. 生成 cos 和 sin
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos, sin

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """
        将 [TTT...HHH...WWW] 交错为 [THWTHWTHW...]
        mrope_section = [24, 20, 20] 表示:
          - 前 24 维: temporal
          - 接下来 20 维: height (每隔 3 个位置)
          - 接下来 20 维: width (每隔 3 个位置)
        """
        freqs_t = freqs[0]  # 时间维度
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t
```

**MRoPE 特点:**
- **3D 位置编码**: 同时编码时间、高度、宽度
- **交错模式**: 细粒度混合空间和时间信息
- **适应多模态**: 文本使用简单位置，视觉使用网格位置

---

### 6. Qwen3VLTextRMSNorm
**位置:** `modeling_qwen3_vl.py:381`
**继承:** `nn.Module`

**作用:** RMS 归一化层（相比 LayerNorm 更高效）

**实现:**
```python
class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # RMS 归一化: x / sqrt(mean(x^2) + eps)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)
```

**优势:** 相比 LayerNorm，不需要计算均值，计算更快

---

## 统一模型类

这些类将视觉和文本组件组合成完整的多模态模型。

### 1. Qwen3VLModel
**位置:** `modeling_qwen3_vl.py:949`
**继承:** `Qwen3VLPreTrainedModel`

**作用:** 核心多模态模型，融合视觉和文本

**主要组件:**
```python
class Qwen3VLModel(Qwen3VLPreTrainedModel):
    def __init__(self, config):
        # 视觉编码器
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)

        # 文本解码器
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)

        # 缓存 RoPE deltas
        self.rope_deltas = None
```

**核心方法:**

#### get_image_features() / get_video_features()
**位置:** `modeling_qwen3_vl.py:1088-1145`

```python
def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
    """
    提取图像特征

    输入:
        pixel_values: (total_patches, C, H, W) - 图像像素
        image_grid_thw: (num_images, 3) - [T, H, W] 网格信息

    输出:
        BaseModelOutputWithDeepstackFeatures:
            - pooler_output: 列表，每个图像的视觉嵌入
            - deepstack_features: 3 层 DeepStack 特征
    """
    # 1. 通过视觉编码器
    vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, ...)

    # 2. 按图像分割嵌入
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(vision_output.pooler_output, split_sizes)

    vision_output.pooler_output = image_embeds
    return vision_output
```

#### get_rope_index()
**位置:** `modeling_qwen3_vl.py:972-1087`

```python
def get_rope_index(self, input_ids, image_grid_thw, video_grid_thw, attention_mask):
    """
    计算多模态序列的 3D 位置 ID

    过程:
    1. 遍历输入序列，找到所有视觉 token
    2. 为文本区域分配顺序位置 [pos, pos, pos]
    3. 为视觉区域分配网格位置 [t, h, w]
    4. 计算 rope_deltas（位置偏移量）

    返回:
        position_ids: (3, batch_size, seq_len) - [T, H, W]
        rope_deltas: (batch_size, 1) - 位置增量
    """
    position_ids = torch.ones(3, batch_size, seq_len, ...)

    for i, input_ids in enumerate(batch):
        llm_pos_ids_list = []

        for each_segment:
            if is_text:
                # 文本: 简单递增位置
                llm_pos_ids_list.append(
                    torch.arange(text_len).expand(3, -1) + st_idx
                )
            else:
                # 视觉: 3D 网格位置
                t_index = torch.arange(llm_grid_t).expand(...).flatten()
                h_index = torch.arange(llm_grid_h).expand(...).flatten()
                w_index = torch.arange(llm_grid_w).expand(...).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + st_idx
                )

        position_ids[..., i, :] = torch.cat(llm_pos_ids_list, dim=1)

    rope_deltas = position_ids.max() + 1 - seq_len
    return position_ids, rope_deltas
```

#### forward()
**位置:** `modeling_qwen3_vl.py:1147-1296`

```python
def forward(
    self,
    input_ids,
    pixel_values,
    pixel_values_videos,
    image_grid_thw,
    video_grid_thw,
    ...
):
    """
    主前向传播

    流程:
    1. 获取文本嵌入
    2. 处理图像/视频，获取视觉嵌入和 DeepStack 特征
    3. 融合嵌入：用视觉嵌入替换占位符 token
    4. 计算位置 ID
    5. 通过文本解码器（注入 DeepStack 特征）
    """
    # 1. 初始化文本嵌入
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # 2. 处理图像
    if pixel_values is not None:
        image_outputs = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0)
        deepstack_image_embeds = image_outputs.deepstack_features

        # 找到 <|image_pad|> 位置
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_embeds)

        # 替换占位符
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # 3. 处理视频（类似图像）
    if pixel_values_videos is not None:
        # ... 类似处理 ...
        pass

    # 4. 聚合 DeepStack 特征
    if image_mask is not None and video_mask is not None:
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = [...]  # 合并图像和视频特征

    # 5. 计算位置 ID
    if position_ids is None:
        if self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        else:
            # 使用缓存的 rope_deltas
            position_ids = compute_from_cached_deltas(...)

    # 6. 通过文本解码器
    outputs = self.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        ...
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas
    )
```

---

### 2. Qwen3VLForConditionalGeneration
**位置:** `modeling_qwen3_vl.py:1322`
**继承:** `Qwen3VLPreTrainedModel`, `GenerationMixin`

**作用:** **主入口类**，添加语言建模头，支持训练和生成

**主要组件:**
```python
class Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)

        # 核心模型
        self.model = Qwen3VLModel(config)

        # 语言建模头
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False
        )

        # 权重绑定
        self._tied_weights_keys = {
            "lm_head.weight": "model.language_model.embed_tokens.weight"
        }
```

**核心方法:**

#### forward()
**位置:** `modeling_qwen3_vl.py:1375-1466`

```python
def forward(
    self,
    input_ids,
    pixel_values,
    labels=None,
    logits_to_keep=0,
    ...
):
    """
    前向传播，支持训练和推理

    输入:
        input_ids: token IDs
        pixel_values: 图像像素
        labels: 训练标签（可选）
        logits_to_keep: 保留多少个 logits（生成时只需最后一个）

    输出:
        Qwen3VLCausalLMOutputWithPast:
            - loss: 训练损失（如果提供 labels）
            - logits: 词汇表 logits
            - past_key_values: KV 缓存
    """
    # 1. 通过核心模型
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        ...
    )

    hidden_states = outputs[0]

    # 2. 计算 logits（仅保留需要的部分）
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    # 3. 计算损失（训练时）
    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, ...)

    return Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        rope_deltas=outputs.rope_deltas,
        ...
    )
```

#### prepare_inputs_for_generation()
**位置:** `modeling_qwen3_vl.py:1468-1531`

```python
def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    pixel_values=None,
    cache_position=None,
    is_first_iteration=False,
    ...
):
    """
    准备生成所需的输入

    Prefill 阶段（第一次）:
        - 处理所有视觉输入
        - 计算完整的 position_ids
        - 缓存 rope_deltas

    Decode 阶段（后续）:
        - 使用 KV 缓存
        - 从 rope_deltas 计算位置
        - 移除视觉输入（已处理）
    """
    model_inputs = super().prepare_inputs_for_generation(...)

    if position_ids is None:
        if cache_position[0] == 0 or self.model.rope_deltas is None:
            # Prefill: 计算完整位置
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.model.rope_deltas = rope_deltas

            # 拼接文本和视觉位置 ID
            text_positions = model_inputs["position_ids"][None, ...]
            vision_positions = position_ids
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)
        else:
            # Decode: 使用缓存的 delta
            position_ids = torch.arange(seq_length, ...)
            delta = cache_position[0] + self.model.rope_deltas
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # Decode 阶段移除视觉输入
    if not is_first_iteration and use_cache:
        model_inputs["pixel_values"] = None
        model_inputs["pixel_values_videos"] = None

    return model_inputs
```

#### _expand_inputs_for_generation()
**位置:** `modeling_qwen3_vl.py:1574-1684`

```python
def _expand_inputs_for_generation(self, expand_size, input_ids, **model_kwargs):
    """
    为 beam search 或多样本生成扩展输入

    特殊处理:
        - 视觉输入按样本分割后重复
        - 文本输入直接 repeat_interleave
    """
    if expand_size == 1:
        return input_ids, model_kwargs

    # 处理视觉输入
    for key in ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]:
        if key == "pixel_values":
            # 按图像分割，然后重复
            samples = torch.split(image_grid_thw, list(image_nums))
            lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
            dict_to_expand[key] = _repeat_interleave_samples(
                dict_to_expand[key], lengths, expand_size
            )
        # ... 其他键类似处理 ...

    # 处理文本输入
    input_ids = input_ids.repeat_interleave(expand_size, dim=0)
    attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)

    return input_ids, model_kwargs
```

---

### 3. Qwen3VLPreTrainedModel
**位置:** `modeling_qwen3_vl.py:589`
**继承:** `PreTrainedModel`

**作用:** 基类，提供权重初始化和通用功能

**实现:**
```python
class Qwen3VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen3VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        """初始化模型权重"""
        super()._init_weights(module)

        # 特殊处理视觉 RoPE
        if isinstance(module, Qwen3VLVisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2) / module.dim))
            module.inv_freq.copy_(inv_freq)
```

---

## 输出数据类

这些数据类封装模型的输出。

### 1. BaseModelOutputWithDeepstackFeatures
**位置:** `modeling_qwen3_vl.py:48`
**继承:** `BaseModelOutputWithPooling`

**作用:** 视觉编码器的输出，包含 DeepStack 特征

**字段:**
```python
@dataclass
class BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling):
    last_hidden_state: torch.FloatTensor          # 最终隐藏状态
    pooler_output: List[torch.FloatTensor]        # 合并后的视觉嵌入列表
    deepstack_features: List[torch.FloatTensor]   # 3 层 DeepStack 特征
```

---

### 2. Qwen3VLModelOutputWithPast
**位置:** `modeling_qwen3_vl.py:570`
**继承:** `ModelOutput`

**作用:** Qwen3VLModel 的输出

**字段:**
```python
@dataclass
class Qwen3VLModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor    # 最终隐藏状态
    past_key_values: Cache                  # KV 缓存
    hidden_states: Tuple[torch.FloatTensor] # 所有层隐藏状态（可选）
    attentions: Tuple[torch.FloatTensor]    # 注意力权重（可选）
    rope_deltas: torch.LongTensor           # RoPE 位置增量
```

---

### 3. Qwen3VLCausalLMOutputWithPast
**位置:** `modeling_qwen3_vl.py:1299`
**继承:** `ModelOutput`

**作用:** Qwen3VLForConditionalGeneration 的输出

**字段:**
```python
@dataclass
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor                 # 训练损失（可选）
    logits: torch.FloatTensor               # 词汇表 logits
    past_key_values: Cache                  # KV 缓存
    hidden_states: Tuple[torch.FloatTensor] # 所有层隐藏状态（可选）
    attentions: Tuple[torch.FloatTensor]    # 注意力权重（可选）
    rope_deltas: torch.LongTensor           # RoPE 位置增量
```

---

## 类继承关系图

```
PreTrainedModel (Hugging Face 基类)
    │
    └─── Qwen3VLPreTrainedModel (基类，权重初始化)
            │
            ├─── Qwen3VLVisionModel (视觉编码器)
            │
            ├─── Qwen3VLTextModel (文本解码器)
            │
            ├─── Qwen3VLModel (核心多模态模型)
            │
            └─── Qwen3VLForConditionalGeneration (主入口)
                     │
                     └─── GenerationMixin (生成功能)


nn.Module (PyTorch 基类)
    │
    ├─── GradientCheckpointingLayer
    │       │
    │       ├─── Qwen3VLVisionBlock
    │       └─── Qwen3VLTextDecoderLayer
    │
    ├─── Qwen3VLVisionPatchEmbed
    ├─── Qwen3VLVisionRotaryEmbedding
    ├─── Qwen3VLVisionPatchMerger
    ├─── Qwen3VLVisionAttention
    ├─── Qwen3VLVisionMLP
    │
    ├─── Qwen3VLTextRotaryEmbedding
    ├─── Qwen3VLTextRMSNorm
    ├─── Qwen3VLTextAttention
    └─── Qwen3VLTextMLP


PreTrainedConfig (Hugging Face 配置基类)
    │
    ├─── Qwen3VLVisionConfig
    ├─── Qwen3VLTextConfig
    └─── Qwen3VLConfig


ModelOutput (Hugging Face 输出基类)
    │
    ├─── BaseModelOutputWithPooling
    │       └─── BaseModelOutputWithDeepstackFeatures
    │
    ├─── Qwen3VLModelOutputWithPast
    └─── Qwen3VLCausalLMOutputWithPast
```

---

## 数据流关系图

```
                    用户输入
                       │
                       ├─ text: "描述图片 <|vision_start|><|image_pad|>...<|vision_end|>"
                       ├─ pixel_values: 图像像素
                       └─ image_grid_thw: [1, 256, 256]
                       │
                       ▼
    ┌──────────────────────────────────────────────────────┐
    │   Qwen3VLForConditionalGeneration (主入口)           │
    │   modeling_qwen3_vl.py:1322                          │
    └──────────────────────────────────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────────────────┐
    │   Qwen3VLModel (核心模型)                            │
    │   modeling_qwen3_vl.py:949                           │
    └──────────────────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐
│ Qwen3VLVisionModel  │    │ get_input_embeddings│
│ (视觉编码器)         │    │ (文本嵌入)           │
│ line:613            │    │                     │
└─────────────────────┘    └─────────────────────┘
         │                           │
         │ pixel_values              │ input_ids
         │ image_grid_thw            │
         ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐
│ Qwen3VLVisionPatch  │    │ embed_tokens        │
│ Embed (Conv3d)      │    │ (batch,seq,hidden)  │
│ line:70             │    └─────────────────────┘
└─────────────────────┘              │
         │                           │
         ▼                           │
┌─────────────────────┐              │
│ Position Embeddings │              │
│ (Learned + RoPE)    │              │
└─────────────────────┘              │
         │                           │
         ▼                           │
┌─────────────────────┐              │
│ 27 x Vision Blocks  │              │
│ ├─ Layer 8  ─┐      │              │
│ ├─ Layer 16 ─┼─ DeepStack Features │
│ └─ Layer 24 ─┘      │              │
└─────────────────────┘              │
         │                           │
         ▼                           │
┌─────────────────────┐              │
│ PatchMerger         │              │
│ (2x2 merge)         │              │
└─────────────────────┘              │
         │                           │
         │ visual_embeds             │
         │ deepstack_features        │
         └───────────┬───────────────┘
                     ▼
         ┌───────────────────────┐
         │ masked_scatter        │
         │ (融合视觉和文本嵌入)    │
         │ line:1212-1220        │
         └───────────────────────┘
                     │
                     │ fused_embeds
                     │ visual_pos_masks
                     │ deepstack_features
                     ▼
         ┌───────────────────────┐
         │ get_rope_index        │
         │ (计算 3D 位置 ID)      │
         │ line:972-1087         │
         └───────────────────────┘
                     │
                     │ position_ids (3, bs, seq)
                     ▼
    ┌────────────────────────────────────────┐
    │   Qwen3VLTextModel (文本解码器)         │
    │   modeling_qwen3_vl.py:823             │
    └────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ MRoPE Embedding       │
         │ (cos, sin)            │
         │ line:291-377          │
         └───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │   28 x Decoder Layers                  │
    │   ┌──────────────────────────────┐    │
    │   │ Qwen3VLTextDecoderLayer      │    │
    │   │ line:519                     │    │
    │   │  ├─ LayerNorm                │    │
    │   │  ├─ Self-Attention (MRoPE)   │    │
    │   │  ├─ Add Residual             │    │
    │   │  ├─ LayerNorm                │    │
    │   │  ├─ MLP (SwiGLU)             │    │
    │   │  └─ Add Residual             │    │
    │   └──────────────────────────────┘    │
    │              │                         │
    │              ▼                         │
    │   ┌──────────────────────────────┐    │
    │   │ DeepStack Process (前 3 层)  │    │
    │   │ hidden[visual_pos] += feat   │    │
    │   │ line:1178-1186               │    │
    │   └──────────────────────────────┘    │
    └────────────────────────────────────────┘
                     │
                     │ hidden_states
                     ▼
         ┌───────────────────────┐
         │ RMSNorm               │
         └───────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │   lm_head (语言建模头)                  │
    │   Linear(hidden_size, vocab_size)      │
    └────────────────────────────────────────┘
                     │
                     ▼
              ┌──────┴──────┐
              │             │
         训练时             生成时
              │             │
              ▼             ▼
    ┌────────────────┐ ┌────────────────┐
    │ Cross-Entropy  │ │ Sampling/Beam  │
    │ Loss           │ │ Search         │
    └────────────────┘ └────────────────┘
              │             │
              ▼             ▼
           loss          new_tokens
```

---

## 类之间的协作关系

### 1. 组合关系（Composition）

```
Qwen3VLForConditionalGeneration
    ├─ contains → Qwen3VLModel
    │               ├─ contains → Qwen3VLVisionModel
    │               │               ├─ contains → Qwen3VLVisionPatchEmbed
    │               │               ├─ contains → Qwen3VLVisionRotaryEmbedding
    │               │               ├─ contains → List[Qwen3VLVisionBlock]
    │               │               │               ├─ contains → Qwen3VLVisionAttention
    │               │               │               └─ contains → Qwen3VLVisionMLP
    │               │               ├─ contains → Qwen3VLVisionPatchMerger
    │               │               └─ contains → List[Qwen3VLVisionPatchMerger] (DeepStack)
    │               │
    │               └─ contains → Qwen3VLTextModel
    │                               ├─ contains → nn.Embedding (embed_tokens)
    │                               ├─ contains → Qwen3VLTextRotaryEmbedding
    │                               ├─ contains → List[Qwen3VLTextDecoderLayer]
    │                               │               ├─ contains → Qwen3VLTextAttention
    │                               │               │               ├─ contains → 2x Qwen3VLTextRMSNorm (QK-norm)
    │                               │               ├─ contains → Qwen3VLTextMLP
    │                               │               ├─ contains → Qwen3VLTextRMSNorm (input_layernorm)
    │                               │               └─ contains → Qwen3VLTextRMSNorm (post_attention_layernorm)
    │                               └─ contains → Qwen3VLTextRMSNorm (final norm)
    │
    └─ contains → nn.Linear (lm_head)
```

### 2. 调用关系（Call Flow）

```
训练/推理流程:

1. Qwen3VLForConditionalGeneration.forward()
      │
      ├─→ Qwen3VLModel.forward()
      │      │
      │      ├─→ Qwen3VLModel.get_image_features()
      │      │      └─→ Qwen3VLVisionModel.forward()
      │      │             ├─→ Qwen3VLVisionPatchEmbed.forward()
      │      │             ├─→ Qwen3VLVisionRotaryEmbedding.forward()
      │      │             ├─→ for each Qwen3VLVisionBlock:
      │      │             │      ├─→ Qwen3VLVisionAttention.forward()
      │      │             │      └─→ Qwen3VLVisionMLP.forward()
      │      │             ├─→ Qwen3VLVisionPatchMerger.forward() (main)
      │      │             └─→ for DeepStack layers:
      │      │                    └─→ Qwen3VLVisionPatchMerger.forward()
      │      │
      │      ├─→ Qwen3VLModel.get_placeholder_mask()
      │      ├─→ inputs_embeds.masked_scatter() (融合)
      │      ├─→ Qwen3VLModel.get_rope_index()
      │      │
      │      └─→ Qwen3VLTextModel.forward()
      │             ├─→ Qwen3VLTextRotaryEmbedding.forward()
      │             ├─→ for each Qwen3VLTextDecoderLayer:
      │             │      ├─→ Qwen3VLTextRMSNorm.forward() (input_layernorm)
      │             │      ├─→ Qwen3VLTextAttention.forward()
      │             │      │      ├─→ Qwen3VLTextRMSNorm.forward() (q_norm)
      │             │      │      ├─→ Qwen3VLTextRMSNorm.forward() (k_norm)
      │             │      │      └─→ apply_rotary_pos_emb()
      │             │      ├─→ Qwen3VLTextRMSNorm.forward() (post_attention_layernorm)
      │             │      ├─→ Qwen3VLTextMLP.forward()
      │             │      └─→ Qwen3VLTextModel._deepstack_process() (if early layer)
      │             └─→ Qwen3VLTextRMSNorm.forward() (final norm)
      │
      └─→ lm_head() (nn.Linear)


生成流程:

1. Qwen3VLForConditionalGeneration.generate() (from GenerationMixin)
      │
      ├─→ Qwen3VLForConditionalGeneration.prepare_inputs_for_generation()
      │      ├─→ Qwen3VLModel.get_rope_index() (prefill stage)
      │      └─→ use cached rope_deltas (decode stage)
      │
      ├─→ Qwen3VLForConditionalGeneration.forward()
      │      └─→ ... (same as training)
      │
      ├─→ Sampling / Beam Search
      │      └─→ Qwen3VLForConditionalGeneration._expand_inputs_for_generation()
      │
      └─→ loop until EOS or max_length
```

### 3. 数据依赖关系

```
配置依赖:
    Qwen3VLConfig
        ├─ used by → Qwen3VLModel
        │              ├─ vision_config used by → Qwen3VLVisionModel
        │              └─ text_config used by → Qwen3VLTextModel
        └─ used by → Qwen3VLForConditionalGeneration

数据流依赖:
    pixel_values (原始像素)
        └─→ Qwen3VLVisionModel
              └─→ visual_embeds + deepstack_features
                    └─→ Qwen3VLModel (融合到 inputs_embeds)
                          └─→ Qwen3VLTextModel (with DeepStack injection)
                                └─→ hidden_states
                                      └─→ lm_head
                                            └─→ logits

位置编码依赖:
    input_ids + image_grid_thw + video_grid_thw
        └─→ Qwen3VLModel.get_rope_index()
              └─→ position_ids (3D: T, H, W)
                    └─→ Qwen3VLTextRotaryEmbedding
                          └─→ (cos, sin) embeddings
                                └─→ Qwen3VLTextAttention (apply to Q, K)

缓存依赖:
    past_key_values (Cache object)
        └─→ Qwen3VLTextAttention
              ├─→ update cache with new K, V
              └─→ return concatenated K, V
                    └─→ used in attention computation
```

---

## 关键交互点

### 1. 视觉-文本融合点
**位置:** `Qwen3VLModel.forward()` (line:1212-1220)

```python
# 用视觉嵌入替换占位符 token
image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

### 2. DeepStack 注入点
**位置:** `Qwen3VLTextModel.forward()` (line:925-935)

```python
# 在前 3 层注入 DeepStack 特征
for layer_idx, decoder_layer in enumerate(self.layers):
    hidden_states = decoder_layer(hidden_states, ...)

    if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
        hidden_states[visual_pos_masks, :] += deepstack_visual_embeds[layer_idx]
```

### 3. 位置编码计算点
**位置:** `Qwen3VLModel.get_rope_index()` (line:972-1087)

```python
# 为文本和视觉区域分配不同的位置编码
# 文本: [pos, pos, pos] (3D 相同)
# 视觉: [t, h, w] (3D 网格)
position_ids = compute_multimodal_positions(input_ids, image_grid_thw, video_grid_thw)
```

### 4. 权重绑定点
**位置:** `Qwen3VLForConditionalGeneration` (line:1327-1329)

```python
# 输入嵌入和输出嵌入共享权重
_tied_weights_keys = {
    "lm_head.weight": "model.language_model.embed_tokens.weight"
}
```

---

## 总结

Qwen3-VL 的类结构设计体现了以下特点:

1. **模块化设计**: 视觉、文本、融合模块清晰分离
2. **组合优于继承**: 通过组合不同模块构建完整模型
3. **配置驱动**: 所有超参数通过配置类管理
4. **数据类输出**: 使用 dataclass 封装输出，类型安全
5. **DeepStack 创新**: 多层视觉特征注入，深度多模态融合
6. **MRoPE 位置编码**: 3D 位置编码适应多模态序列
7. **高效生成**: KV 缓存、rope_deltas 缓存优化推理

主要类之间形成清晰的调用链:
```
用户 → ForConditionalGeneration → Model → VisionModel + TextModel → 各种子模块
```

每个类职责明确，通过良好的接口协作完成多模态理解任务。
