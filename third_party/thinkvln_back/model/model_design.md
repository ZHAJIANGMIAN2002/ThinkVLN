这是一个关于基于 Qwen3-VL 构建 **Aux-Think + CycleVLA** 架构的工程设计文档。该文档可以直接用于指导开发。

---

# 工程设计文档：Qwen3-VLA (Aux-Think & CycleVLA)

**版本**: 1.0
**状态**: Draft
**基础模型**: Qwen3-VL (2B) / Qwen2.5-VL Compatible
**目标**: 构建一个高效的 Vision-Language-Action (VLA) 模型，具备训练时推理（Aux-Think）和推理时并行输出（CycleVLA）的能力。

---

## 1. 总体架构 (System Architecture)

本系统采用 **Dual-Head（双头）** 架构设计，核心思想是在共享的 Vision-LLM 主干网络之上，分离“推理（Reasoning）”与“执行（Action）”的职责。

* **Training (Aux-Think Mode)**: 这是一个多任务学习过程。模型同时学习生成 CoT（Chain-of-Thought）文本和预测动作。CoT Loss 作为辅助损失，强迫模型将推理逻辑内化到 Hidden States 中。
* **Inference (No-Think/CycleVLA Mode)**: 这是一个单次前向传播（One-pass Forward）过程。系统截断 CoT 生成路径，直接通过 Action Head 并行解码出  步动作序列和进度值。

### 1.1 模块依赖关系

```mermaid
graph TD
    Input[输入: Image + Text Prompt] --> Processor[Data Processor]
    Processor --> |Tensors| QwenBackbone[Qwen3-VL Backbone]
    
    QwenBackbone --> |Last Hidden State| FeatureRouter[Feature Router]
    
    subgraph "Heads"
        FeatureRouter --> |Action Token Index| ActionHead[Parallel Action Head (CycleVLA)]
        FeatureRouter --> |Full Sequence| LMHead[Causal LM Head (Aux-Think)]
    end
    
    subgraph "Outputs"
        ActionHead --> |Shape: [B, N, 4]| ActionLogits[Action Sequence]
        ActionHead --> |Shape: [B, N, 1]| ProgressVal[Progress Value]
        LMHead --> |Shape: [B, S, V]| TextLogits[CoT Text Generation]
    end

```

---

## 2. 模型规范 (Model Specification)

### 2.1 配置类: `Qwen3VLAConfig`

继承自原生配置，增加 VLA 任务特有的超参数。

| 参数名 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `prediction_horizon` | `int` | `8` | CycleVLA 中的 ，一次预测的动作步数 |
| `action_dim` | `int` | `4` | 离散动作空间大小 (Stop, Forward, Left, Right) |
| `progress_dim` | `int` | `1` | 进度值维度 (0.0 - 1.0) |
| `aux_loss_weight` | `float` | `0.1` | Aux-Think CoT 损失权重 |
| `action_loss_weight` | `float` | `1.0` | 动作分类损失权重 |
| `progress_loss_weight` | `float` | `0.5` | 进度回归损失权重 |

### 2.2 核心组件: `ParallelActionHead`

实现 CycleVLA 的并行解码逻辑。该组件是一个多层感知机（MLP），将单个 Token 的隐状态映射为未来的动作序列。

* **Input Shape**: `[Batch_Size, Hidden_Size]`
* **Layer Spec**:
1. `Linear(Hidden_Size, Hidden_Size)` + `GELU`
2. `Linear(Hidden_Size, N * (Action_Dim + Progress_Dim))`


* **Output Reshape**:
* `action_logits`: `[Batch_Size, N, Action_Dim]`
* `progress_values`: `[Batch_Size, N, 1]` (Apply Sigmoid)



### 2.3 主模型: `Qwen3VLAForAction`

对 HuggingFace `Qwen2_5_VLForConditionalGeneration` 的封装与修改。

#### 输入接口 (Forward Inputs)

| 参数名 | 形状 (Shape) | 必须 | 说明 |
| --- | --- | --- | --- |
| `input_ids` | `[B, Seq_Len]` | Yes | 包含 Prompt、Image Placeholders 和 (训练时) CoT 文本 |
| `pixel_values` | `[B, C, H, W]` | Yes | 图像张量 |
| `attention_mask` | `[B, Seq_Len]` | Yes | 掩码 |
| `action_token_indices` | `[B]` | Yes | 指示 `< |
| `labels` | `[B, Seq_Len]` | No | Aux-Think 的 CoT 文本标签 (用于 LM Loss) |
| `action_labels` | `[B, N]` | No | CycleVLA 的动作 Ground Truth (离散 ID) |
| `progress_labels` | `[B, N]` | No | CycleVLA 的进度 Ground Truth (浮点数) |

#### 输出接口 (Forward Outputs)

返回一个 `dataclass` 对象，包含：

* `loss`: `torch.FloatTensor` (加权总损失，仅训练时返回)
* `action_logits`: `torch.FloatTensor` `[B, N, 4]` (推理时主要输出)
* `progress_values`: `torch.FloatTensor` `[B, N, 1]` (推理时主要输出)

#### 逻辑流程规范

1. **Backbone Encoding**: 调用 Qwen3 Encoder 获取 `last_hidden_state`。
2. **Auxiliary Branch (Training Only)**:
* 如果提供了 `labels`，将 `last_hidden_state` 传入 `lm_head`。
* 计算 CrossEntropy Loss (`loss_cot`)。


3. **Action Branch (Always)**:
* 使用 `action_token_indices` 从 `last_hidden_state` 中 Gather 出对应位置的向量。
* 传入 `ParallelActionHead`。
* 如果提供了 `action_labels`，计算 CrossEntropy Loss (`loss_act`)。
* 如果提供了 `progress_labels`，计算 MSE Loss (`loss_prog`)。


4. **Loss Aggregation**: `Total Loss = w1 * loss_act + w2 * loss_prog + w3 * loss_cot`。

---

## 3. 数据规范 (Data Specification)

### 3.1 Prompt 模板设计

Aux-Think 的核心在于 Prompt 的设计，必须明确区分“思考区域”和“行动触发点”。

**Token 定义**:

* `<|vision_start|>`, `<|vision_end|>`: Qwen 原生图像标记。
* `<|action_start|>`: **关键自定义 Token**。Action Head 将挂载在此 Token 的输出 Embedding 上。

**模板结构**:

```text
<|im_start|>system
You are a spatial navigation agent.<|im_end|>
<|im_start|>user
<|vision_start|><image><|vision_end|>
Instruction: {instruction}
Current Subgoal: {subgoal}
<|im_end|>
<|im_start|>assistant
<|action_start|> {CoT_Text} <|im_end|>

```

### 3.2 标签构建规则 (Label Construction)

在 `DataCollator` 中处理：

1. **`input_ids`**: 包含完整的 System + User + Assistant (CoT)。
2. **`labels` (LM Loss)**:
* System / User 部分: Set to `-100` (Ignored).
* `<|action_start|>`: Set to `-100` (Action Head 负责此处，LM Head 忽略或学习预测 CoT 首字).
* `{CoT_Text}`: Set to `Token_IDs`.


3. **`action_token_indices`**: 找到 `<|action_start|>` 在 batch 中的 index。
4. **`action_labels`**: 从数据集中读取对应的  步动作 ID。
5. **`progress_labels`**: 从数据集中读取对应的  步进度值。

---

## 4. 训练与推理流程 (Pipeline Specification)

### 4.1 训练流程 (Training Loop)

采用 **Teacher Forcing** 模式训练 CoT，同时训练 Action Expert。

1. **Load Batch**: 获取图文对和 GT Actions。
2. **Forward Pass**: 开启所有 Head。
3. **Backward Pass**: 三部分 Loss 联合反向传播。
* *Rationale*: Aux-Think 论文证明，联合训练能让 Backbone 学习到 CoT 的推理模式，并将其压缩到 `<|action_start|>` 的 Embedding 中。



### 4.2 推理流程 (Inference - No Think)

**关键**: 推理时完全绕过 LLM 的解码循环（Generation Loop）。

**Input**: Image, Instruction.

**Process**:

1. **Construct Prompt**: 截断到 `<|action_start|>` 为止。
* `... <|im_start|>assistant\n<|action_start|>`


2. **Tokenize**: 转为 Tensor。
3. **Model Forward**:
* Backbone 计算特征。
* 定位到最后一个 Token (`<|action_start|>`)。
* 通过 `ParallelActionHead`。


4. **Decode**:
* `argmax(action_logits)` -> 得到长度为  的离散动作列表。
* `sigmoid(progress_values)` -> 得到长度为  的进度列表。


5. **Return**: List of Actions, List of Progress.

**Latency Analysis**:

* 相比传统 VLA (Auto-regressive generation of CoT + Actions): 速度提升约 **** 倍。
* 相比 Naive Parallel: 增加了推理能力的内化，准确率更高。

---

## 5. 接口定义 (Python Interface)

```python
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch

@dataclass
class VLAOutput:
    loss: Optional[torch.Tensor]
    actions: torch.Tensor          # [B, N]
    progress: torch.Tensor         # [B, N]

class Qwen3VLA(torch.nn.Module):
    def __init__(self, config: Qwen3VLAConfig):
        ...
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        action_token_indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        progress_labels: Optional[torch.Tensor] = None
    ) -> VLAOutput:
        ...

class VLAInference:
    def predict(self, image, instruction) -> Tuple[List[int], List[float]]:
        """
        End-to-end Inference Pipeline
        """
        ...

```