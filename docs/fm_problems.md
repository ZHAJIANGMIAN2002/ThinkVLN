**结论**
这套 FM 实现从“学未来轨迹分布”的角度看，主体是自洽的；但从“闭环导航里用它来选下一个离散动作”的角度看，目前**不算写正确**。核心问题不在 `flow matching loss` 本身，而在**标签定义、坐标系、以及 eval 里把 waypoint 解码成动作的规则不一致**。

**关键发现**
- 最严重的问题在 [navigation_model.py](/mnt/swx/ThinkVLN/thinkvln/models/navigation_model.py#L652)。它只取 `waypoint_preds` 的第一个点 `first_wp`，然后用 `dist<=0.06 => stop`、`|dx|/|dz|` 大 => 左右转 的启发式来还原动作。这隐含假设是：`dx,dz` 是**以机器人朝向为基准的 ego-frame 控制量**。
- 但数据标签不是这个定义。[trajectory_generation_waypoint.py](/mnt/swx/ThinkVLN/thinkvln/datagen/generation/trajectory_generation_waypoint.py#L54) 记录的是 Habitat 里的**世界坐标** `[x, z]`；[fm_waypoint_dataset.py](/mnt/swx/ThinkVLN/thinkvln/dataset/fm_waypoint_dataset.py#L16) 再直接做 `future_pos - current_pos`，所以标签是**world-frame 的未来累计位移**，不是 ego-frame waypoint。
- 这会导致闭环动作解码系统性错位。我把真实标签的“第一步 waypoint”直接喂给当前 `_waypoint_to_action_logits` 规则做统计，结果是：
  - `next_action=turn_left` 的 95,288 个样本里，第一步位移全是 `[0,0]`，当前规则会全部解码成 `stop`。
  - `next_action=turn_right` 的 89,644 个样本里，第一步位移也全是 `[0,0]`，同样全部解码成 `stop`。
  - `next_action=forward` 的 330,350 个样本里，只有 41.9% 会被当前规则解成 `forward`，剩下大量被误判成左/右转，因为 world-frame 的 `dx` 往往不小。
- 所以如果你的目标是“预测未来世界坐标轨迹”，这份数据大体合理；如果目标是“靠预测结果驱动离散导航动作”，当前定义**不符合需求**。
- 次要但真实存在的不一致是训练/推理上下文不同：训练配置 [sft_training_fm.yaml](/mnt/swx/ThinkVLN/config/sft_training_fm.yaml#L8) 用 `memory_num_history_images: 0`，而闭环加载默认在 [close_eval_models.py](/mnt/swx/ThinkVLN/thinkvln/eval/close_eval_models.py#L136) 里给 FM 模型 `memory_num_history_images=6` 且 `use_memory=True`。这不是根因，但会额外拉低表现。
- 现有测试也没有覆盖到这个语义问题。[test_fm_waypoint_dataset.py](/mnt/swx/ThinkVLN/thinkvln/tests/test_fm_waypoint_dataset.py#L8) 只验证了 delta 提取和尾部补零；[test_fm_components.py](/mnt/swx/ThinkVLN/thinkvln/tests/test_fm_components.py#L8) 只验证 shape。

**模型、loss、输出到底是什么**
- 模型在 [thinkvln_fm_actor.py](/mnt/swx/ThinkVLN/thinkvln/models/thinkvln_fm_actor.py#L135) 里从最后的 query tokens 取 action queries，经过 `SharedProjector` 后做平均，得到一个条件向量 `conditioning`，shape 是 `[B, D]`。
- 训练时在 [thinkvln_fm_actor.py](/mnt/swx/ThinkVLN/thinkvln/models/thinkvln_fm_actor.py#L90) 做的是标准的直线路径 conditional flow matching 形式：`noise ~ N(0, I)`，`noisy=(1-t)noise+t*target`，目标速度场 `velocity = target - noise`，loss 是 `MSE(pred_velocity, velocity)`。从公式上看，这一段是**成立的**。
- 推理时在 [thinkvln_fm_actor.py](/mnt/swx/ThinkVLN/thinkvln/models/thinkvln_fm_actor.py#L60) 从高斯噪声开始，做若干 Euler step，最后输出 `waypoint_preds`。
- 这个输出不是 logits，也不是离散动作，而是一个连续张量 `[B, action_horizon, action_dim]`。按你的训练配置 [sft_training_fm.yaml](/mnt/swx/ThinkVLN/config/sft_training_fm.yaml#L1)，实际就是 `[B, 4, 2]`，每一行都是未来第 `k` 步相对当前位置的累计 `(dx, dz)`。

**一条真实样本**
我直接抽了 [summary_full_with_waypoints.jsonl](/mnt/swx/ThinkVLN/data/trajectory_data/R2R_back/summary_full_with_waypoints.jsonl) 第一条记录：

- `episode_key = 17DRP5sb8fy_10154`
- instruction: “Walk forward in the direction of the dining room. Veer right, and go down the hall into the bathroom. Stop in front of the sink.”

看 `frame_idx = 4`：
- `next_actions = [1, 1, 1, 2]`
- `waypoint_labels = [[0.216506, 0.125], [0.433013, 0.25], [0.649519, 0.375], [0.649519, 0.375]]`

这条标签如果理解成“未来 4 步的世界坐标累计位移”，是合理的：连续 3 个 `forward` 后位置持续推进，第 4 步是转向，所以累计位移不再增加。

再看 `frame_idx = 11`：
- `next_actions = [3, 1, 1, 2]`
- `waypoint_labels = [[0.0, 0.0], [0.216506, 0.125], [0.433013, 0.25], [0.433013, 0.25]]`

这对“轨迹预测”也合理，因为下一步先右转，位置不变，所以第一步位移就是 `[0,0]`。但对当前闭环动作规则来说，这个正确标签会被当成 `stop`，因此**不符合当前动作预测接口的需求**。

如果你下一步要把这套 FM 真正用于 VLN 闭环，我建议先二选一统一目标：要么把标签改成 **ego-frame 未来轨迹**，要么保留 world-frame 轨迹，但重写 action decoder，不再只看第一个 waypoint。