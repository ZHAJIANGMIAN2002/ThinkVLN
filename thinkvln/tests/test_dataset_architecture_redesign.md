# Dataset架构重设计

## 当前问题

### 问题1: 处理逻辑分散
- **Dataset**: 返回原始的dict数据，只做基本的加载和混合
- **Collator**: 负责加载图像、生成标签、padding等复杂处理

**问题**: 使得Dataset的输出不是直接可用的，需要经过Collator才能形成最终的样本

### 问题2: 图像路径错误
```python
# 当前代码 (线158-163)
episode_key = "17DRP5sb8fy_10154"  # 数据中的格式
image_path = os.path.join(image_root, episode_key, f"{frame_idx:06d}_rgb.jpg")
# 结果: /path/17DRP5sb8fy_10154/000000_rgb.jpg
# 但真实目录是: /path/17DRP5sb8fy_r2r_010154/000000_rgb.jpg

# parse_frame_key也有问题
# 输入: "17DRP5sb8fy_10154_000035"
# 输出: "17DRP5sb8fy_r2r_010154" (错误地添加了_r2r_和补零)
# 应该输出: "17DRP5sb8fy_10154" (与数据中的episode_key一致)
```

## 改进方案

### 方案1: 让Dataset返回完整处理后的样本（推荐）

**架构变化：**
```
Dataset: 
  输入: 原始JSONL数据
  输出: 完整的、可直接用于模型的数据
  - input_ids (已tokenized)
  - attention_mask
  - pixel_values (已加载的图像)
  - image_grid_thw
  - action_labels / progress_labels (for action mode)
  - labels (for CoT mode)

DataCollator:
  输入: 多个Dataset items
  输出: batched tensors
  - 处理padding到相同长度
  - 处理batch级别的mask
  - stacking tensors
```

**优点：**
1. Dataset就是可直接使用的
2. 更容易单独测试Dataset
3. Collator逻辑简化
4. 错误更容易定位

### 方案2: 调整文件路径映射

修复parse_frame_key和图像路径逻辑：
- parse_frame_key应该返回数据中的episode_key格式
- 图像路径构造需要考虑目录结构mapping

## 建议实施步骤

1. **第一步**: 修复路径问题（必须）
   - 修正parse_frame_key
   - 修正图像路径构造

2. **第二步**: 重构Dataset处理流程（可选但推荐）
   - 让Dataset.__getitem__返回完整处理的样本
   - 移除Collator中的图像加载和标签生成逻辑
   - Collator专注于batch级别处理

3. **第三步**: 更新tests
   - 测试新的路径逻辑
   - 测试完整的Dataset输出

## 实施难度评估

- **路径修复**: 简单（需要mapping表）
- **架构重设计**: 中等（需要修改多个文件，但逻辑清晰）
- **测试更新**: 简单

## 关键问题需要解决

1. 图像真实路径结构：
   - 数据中的episode_key: `17DRP5sb8fy_10154`
   - 目录中的结构: `17DRP5sb8fy_r2r_001793`
   - 需要建立映射或确认哪个是正确的

2. 所有必要的处理是否都可以在Dataset中完成：
   - 图像加载（需要处理不存在的情况）
   - Tokenization（需要processor）
   - 标签生成（已经有工具函数）

3. 内存考虑：
   - Dataset直接返回tensors会增加内存使用
   - 可以使用lazy loading或cached处理
