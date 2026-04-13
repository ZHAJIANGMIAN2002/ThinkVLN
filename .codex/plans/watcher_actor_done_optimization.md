## Watcher/Actor Done-Progress 优化计划

### Summary
- 目标是提升 two-system 的稳定性与最终成功率，核心问题定义为：`subtask` 局部执行能力尚可，但 `watcher -> actor` 的 handoff 边界判断不稳定，导致 loop、切早、切晚。
- 已锁定的关键决策：
1. `watcher done` 是决定性控制信号，优先追求高 precision。
2. `actor done` 仅保留为辅助目标，不再承担最终切换判据。
3. `actor progress` 继续做连续值预测，并加入 GRU 做 episode 内时序建模。
4. `watcher system prompt` 中加入一个很短的 `meta action table`，但不引入 map，不大改现有数据格式。
- 交付范围覆盖三块：watcher done 训练、actor progress/done 改造、two-system runtime handoff 门控与调试归因。

### Key Changes
1. **Watcher：独立高精度 handoff 判断**
- 将 watcher 的 `done` 明确定义为“当前 active subtask 是否可以安全交接到下一个 subtask”，而不是“看起来快做完了”。
- watcher 训练单独建流，不复用 actor 的 `done=progress>threshold` 逻辑。
- 监督来源分两层：
  - 人工 `done` 标注（约 4k）作为高质量强监督，用来校准 precision。
  - `summary_full + GT trajectory + GT subtask split` 自动生成弱监督边界样本，用来扩充覆盖率。
- 训练样本尽量在线构造，不强制新增一套复杂 JSONL 格式；优先在 dataset 层从现有 `summary_full` 采样。
- watcher 输出仍保持现有简洁接口：`memory / done / subtask`，避免评估链路重写。

2. **Actor：强化 progress，弱化 done**
- actor 主任务仍是动作预测，不改变主输入输出范式。
- `progress` 保留连续监督，改为带 GRU 的时序 head，目标是：
  - 减少单帧抖动
  - 提升 subtask 内 progress 单调性
  - 提升边界前后的稳定性
- `done` 保留为辅助 head，但语义降级为“局部接近结束”，不作为 two-system 切换依据。
- loss 权重上继续保持 `action` 为主、`progress` 次之、`done` 最轻，避免 actor 被二值边界监督主导。

3. **Meta Action Table：短 prompt 先验**
- 在 watcher system prompt 中加入一张极短的规则表，只保留 5 类：
  - `TURN`
  - `MOVE`
  - `ENTER_EXIT`
  - `STAIRS`
  - `STOP`
- 每类只保留两条短规则：`done when` / `not done when`。
- 统一加入一条默认策略：`If ambiguous, do not hand off early.`
- 这张表的作用不是替代训练，而是：
  - 限制 watcher 的自由发挥
  - 给 watcher done 训练提供固定语义框架
  - 给后续 debug 和分类分析提供稳定类别
- actor prompt 默认不加入这张表，避免训练文本继续膨胀。

4. **利用现有数据扩充 watcher done 训练**
- 不引入 `progress / near_done / handoff_now` 这类新标签体系。
- 只训练一个 watcher 侧二分类目标：`handoff_now`。
- 基于现有 `subtask_sequence` 定义样本：
  - 正样本：每个 subtask 末尾窗口帧
  - 普通负样本：subtask 前段和中段
  - 难负样本：边界前若干帧、以及容易出现 looping 的近边界帧
- 采样时按 subtask 大类做平衡，避免 `MOVE` 类占比过高。
- memory 仍可作为上下文，但本期不要求严格标注和严格评估。

5. **Two-System Runtime：加保守 handoff 门控**
- watcher 的 `done` 不直接一票切换，增加轻量 gate，避免一帧误判造成 cascade failure。
- 默认采用偏保守策略，例如：
  - watcher 连续两次判定 `done`
  - 或 watcher 判定 `done` 且 actor progress 达到保守阈值
- actor 的 `done` 不作为必要条件，只能作为弱辅助信号或 debug 信号。
- trace/debug 输出需要明确区分：
  - watcher raw done / final done
  - actor progress
  - actor done
  - 当前 subtask class
  - gate 是否触发
- looping 分析优先分成三类：`该切未切`、`切早`、`memory 干扰`。

### Implementation Outline
1. **Watcher Done 训练链路**
- 在现有 watcher dataset / prompt builder 基础上，增加短版 `meta action table` 注入。
- 新增 watcher done 样本采样逻辑，支持：
  - 从人工标注集读取高质量正负样本
  - 从 `summary_full` 在线生成弱监督边界样本
  - 按比例混采两类来源
- watcher 训练目标保持简洁二分类，不引入新的推理 schema。

2. **Actor Progress GRU**
- 在现有 aux head 路径上增加一个轻量 GRU，用当前 pooled hidden state 序列预测 progress。
- episode 内重置逻辑必须与现有 `episode_key / subtask reset` 对齐。
- 训练接口保持兼容现有 `progress_labels / done_labels`，避免数据侧重写。

3. **Runtime Handoff Gate**
- 在 two-system runtime 中把 watcher done 和 actor progress 做一个明确的门控层。
- gate 默认保守，优先降低 premature handoff。
- 配置项保持少量、直观，可直接在 eval config 中切换。

### Config Changes
- watcher 训练/推理配置新增：
  - `use_meta_action_table`
  - `meta_action_table_version`
  - `watcher_done_positive_window`
  - `watcher_done_hard_negative_window`
  - `watcher_done_human_mix_ratio`
- actor 配置新增：
  - `progress_temporal_head`
  - `progress_gru_hidden_size`
  - `progress_gru_layers`
- two-system eval 配置新增：
  - `watcher_done_gate_mode`
  - `watcher_done_min_votes`
  - 可选 `actor_progress_gate_threshold`
- 现有数据文件格式默认保持兼容，不要求重建 actor 数据集格式。

### Test Plan
1. **数据与标签**
- 校验 watcher done 在线采样能从 `summary_full` 正确构造正负样本。
- 校验人工标注与弱监督样本混采比例、边界窗口、类别平衡逻辑。
- 校验 `meta action table` 与现有 subtask 分类映射一致。

2. **模型**
- 校验 actor GRU progress head 的 shape、reset、loss 计算与旧接口兼容。
- 校验 actor done 仍为辅助项，不破坏现有动作输出。
- 校验 watcher done 训练输出可以直接对接现有 watcher schema。

3. **Runtime**
- 校验短版 watcher system prompt 长度可控，JSON schema 不变。
- 校验 handoff gate 在 watcher done 抖动时不会立即切换。
- 校验 trace 中能完整记录 watcher/actor/gate 的边界信息。

4. **行为验收**
- two-system `valseen / valunseen` 上，watcher done precision 提升，looping 数量下降。
- 明确统计 `premature handoff` 与 `late handoff` 两类失败。
- actor 单独 close-loop 不因辅助头改造出现明显退化。

### Assumptions & Defaults
- 本期不引入 map 特征。
- 本期不新增三分类边界标签，也不重构 actor 数据格式。
- watcher done 优先级高于 recall，默认选择保守 handoff。
- `meta action table` 仅先用于 watcher；若验证有效，再考虑蒸馏回 actor 或数据标注流程。
- memory 本期不作为严格监督目标，只当弱上下文使用。
