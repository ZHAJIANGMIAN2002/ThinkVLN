---

## Watcher v0 Spec

### Goal

采用最简的 `span-level` 数据设计，将 `memory` 和 `decision` 合并到同一条样本中。

每条样本对应一个 span：(span 表示起点到终点的一段 traj）

- 起点：`pivot`
- 终点：一次 rollout 结束
- 输出：`label + memory_end`

---

### Data Collection

对于每个 episode：

1. 选择 `N` 个 `pivot`
2. 用 ground-truth 使用 shortest path follower走到 `pivot`
3. 基于 `episode[0:pivot]` 调用 API，生成 `mem_start`
4. 从 `pivot` 开始调用 model，执行 `R` 次 rollout
5. 每次 rollout 最多执行 `M` steps
6. rollout 结束后：
    - 标注 `label ∈ {RESUME, PROCEED, FAIL}`
    - 调用 API 生成 `memory`

总样本数约为：

```
#samples ≈ #episodes × N × R
```

### Pivot

保留两类：

- `Boundary pivot`：子任务切换前 `1~3` 步
- `Mid-subtask pivot`：子任务前中段随机采样

推荐比例：

```
50% Boundary + 50% Mid-subtask
```

---

### Memory

### `mem_start`

定义：

> 从 episode 开始到当前 pivot 为止，对未来决策仍然相关的信息进行压缩总结
> 

保留三类信息：

- 已完成的关键子任务
- 当前未完成目标
- 最近一次失败事实（如果有）

长度控制在 `1~2` 句。

### `mem_end`

定义：

> 当前 span 结束后，供下一次 Watcher 使用的压缩记忆
> 

重点是更新状态，而非复述轨迹。

---

### Label

- `PROCEED`：当前子任务已完成，可切换到下一个子任务
- `RESUME`：当前子任务未完成，但轨迹基本正常，应继续执行
- `FAIL`：明显偏航、循环、错误 stop 或进入错误区域

---

### Data Format

使用单个 `jsonl` 文件，每行一个样本：

```json
{"ep":"E001","pivot":3,"rollout":1,"sid":4,"mem_start":"已沿走廊移动到浴室门口附近，当前目标是进入浴室。","traj":"a=[forward,forward,right,forward], p=[0.62,0.70,0.74,0.91]","label":"PROCEED","mem_end":"已进入浴室，子任务4完成，下一步需要在浴室内定位目标物体。"}
```

字段说明：

- `ep`：episode id
- `pivot`：第几个 pivot
- `rollout`：第几次 rollout
- `sid`：当前 subtask id
- `memory_start`：episode prefix 的压缩记忆
- `traj`：当前 span 的动作/progress 摘要
- `label`：`RESUME / PROCEED / FAIL`
- `memory_end`：span 结束后的压缩记忆

---

### Training Target

输入：

- `memory_start`
- `traj`

输出：

- `label`
- `memory_end`

Watcher v0 同时学习两件事：

1. 当前 span 应该标注为 `RESUME / PROCEED / FAIL`
2. 当前 span 结束后如何压缩更新 memory

---

### Default Hyperparameters

```
N = 4
R = 3
M = 6
```

建议先做小规模 sanity check：

```
50 episodes × 4 pivots × 3 rollouts = 600 samples
```

检查项：

- label 分布是否合理
- `memory_start` 是否过长
- `memory_end` 是否只是复述轨迹
- `FAIL` 样本是否足够