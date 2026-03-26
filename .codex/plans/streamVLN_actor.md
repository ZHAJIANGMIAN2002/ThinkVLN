## StreamVLN Actor 增强与评估集成计划（Progress/Done + Subtask/Hint）

### Summary
- 目标是新增一个独立的 `streamvln_actor` 路线，不改坏现有 `streamvln` 推理逻辑。
- 已锁定的关键决策：
1. watcher 标注来源：本地 Qwen3-VL-8B（主标注）。
2. 模型类型策略：新增 `streamvln_actor`（不覆盖旧类型）。
3. 训练数据配方：`GT + watcher 增强`（GT 主体监督 + watcher 样本补充 hint 语义）。
- 交付范围覆盖三块：数据采集、训练（含 LoRA）、two-system + subtask 两套评估接入。

### Key Changes
1. **数据与监督改造（训练输入输出统一）**
- 新增 Stream actor 训练样本构建路径：从 `summary_full.jsonl` 生成 GT 主数据（逐帧/分段样本），并混入 watcher 增强样本（来自 manifest+annotation merge）。
- 训练输入统一为三段：`instruction + current subtask + watcher hint`，并保留 `previous progress`（teacher forcing）字段，和 ThinkVLN actor 行为保持一致。
- 训练标签统一为 GT：
  - `action_labels`：GT 动作序列切片。
  - `progress_labels`：按当前 subtask 内归一化进度计算。
  - `done_labels`：`progress > done_threshold`（默认 0.85）。
- watcher 样本只提供语言增强（hint/subtask 语义），`progress/done` 仍用 GT 规则，不使用 watcher 的 `done` 作为监督标签。
- 样本混合默认：GT 主体 + watcher 过采样到约 20-30%（默认 25%）以确保 hint 学习有效但不过拟合。

2. **Stream 训练链路（含 LoRA）**
- 在现有 Stream 训练代码中接入新样本字段与 collate 输出：`progress_labels` / `done_labels`。
- 使用模型已有 aux head（`progress_head` / `done_head`）参与训练，不新增复杂结构。
- LoRA 训练增强：
  - 保留现有 LoRA 注入逻辑。
  - 显式保存/恢复与 `progress_head`、`done_head` 相关可训练模块（避免仅保存 LoRA 导致 aux 头丢失）。
- 新增一组明确可复现的训练参数入口（可通过新增配置或参数分组），包括：`done_threshold`、watcher 混合比例、aux loss 权重。

3. **导航模型与加载器集成**
- 新增 `model_type=streamvln_actor` 的加载分支：
  - 支持 full checkpoint。
  - 支持 LoRA adapter + `base_model_path` 自动解析加载。
- 完整实现 Stream 导航包装层缺失能力（当前调用链需要但未实现）：
  - episode 状态重置。
  - instruction/subtask/hint 拼接。
  - aux head 推断 `progress/done`（并提供 fallback）。
- 统一 `predict_action_with_progress_and_done` 行为，使其满足 two-system 与 close-eval 接口预期。

4. **评估接入（two-system + subtask）**
- two-system：
  - 复用现有 `subgoal/hint` 透传链路，接入 `streamvln_actor` 配置即可运行。
- subtask close-eval：
  - CLI 放开 `streamvln_actor` 选项。
  - Runner 从“仅 ThinkVLNActor 类”判断改为“能力接口判断”（支持任意实现 `predict_action_with_progress_and_done` 的导航模型）。
  - 对 memory-bank 相关调试字段做兼容处理，避免非 ThinkVLNActor 路径崩溃。
- 结果输出保持现有统计口径，便于和 ThinkVLN/旧 Stream 直接对比。

### Data Collection Runbook (本地 Qwen3-VL-8B)
1. 生成 watcher rollout bundle（train split）：
- 使用现有 rollout 生成脚本，输出 `manifest` + rollout 图像。

2. 本地 Qwen 标注 watcher：
- 使用 deploy 标注脚本，指定本地 OpenAI-compatible endpoint（例如 `http://localhost:11451/v1`）与本地模型名（Qwen3-VL-8B 服务名）。
- 开启 `--resume` 与并发 worker，支持断点续标。

3. 合并数据集：
- 将 manifest 与 annotation 合并为 watcher 训练增强集（保留 `memory_start/next_subtask/memory_end`）。
- 对 5-10% 样本做人工 spot-check（格式、done 逻辑、hint 可用性）。

4. 训练输入构建：
- GT 主数据来自 `summary_full`。
- watcher 增强数据来自 merge 结果。
- 最终生成统一 Stream actor SFT 样本清单供训练器读取。

### Test Plan
1. **单元测试**
- 数据集测试：验证 prompt 三段结构、`previous progress`、`progress/done` 标签规则与阈值行为。
- 导航包装测试：验证 `subgoal/hint` 注入、subtask 切换时 progress reset、aux head 推断与 fallback。
- 加载器测试：验证 `streamvln_actor` 在 full 与 LoRA 两种路径均可加载。

2. **评估链路测试**
- close-eval CLI/runner：新增 `streamvln_actor` 解析与 capability-check 覆盖。
- two-system：使用 fake watcher/backend 做 smoke test，确保 actor 能接收 `subgoal/hint` 并返回 `action/progress/done`。

3. **端到端验证**
- 小样本训练（smoke）确认 loss 曲线、checkpoint 可加载。
- 小样本两套评估跑通并产出 summary 文件。
- 全量评估对比三组结果：旧 `streamvln`、新 `streamvln_actor` full、新 `streamvln_actor` LoRA。

### Assumptions & Defaults
- `done_threshold` 默认 0.85（与现有 ThinkVLN 训练口径一致）。
- watcher 本地服务为 OpenAI-compatible 接口，模型名由本地服务注册名提供。
- 不新增文档文件；仅改代码、配置与测试。
- 旧 `streamvln` 路径保持可用，新能力全部走 `streamvln_actor`。
