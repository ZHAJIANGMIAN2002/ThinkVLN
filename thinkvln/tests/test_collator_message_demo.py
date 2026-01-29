#!/usr/bin/env python3
"""
ThinkVLN 数据集处理流程详细展示
展示原始样本 -> Collator处理后的message（未经过processor）
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
from thinkvln.dataset.dataset import ThinkVLNDataset
from thinkvln.tools.dataset_utils import extract_action_chunk, parse_frame_key


def show_action_samples():
    """展示Action样本的处理流程"""
    
    print("\n" + "="*80)
    print("ACTION 样本处理流程详细展示")
    print("="*80)
    
    # 数据路径
    action_data_path = "/mnt/swx/ThinkVLN/data/trajectory_data/R2R_back/summary_full.jsonl"
    cot_data_path = "/mnt/swx/ThinkVLN/data/cot_dataset/cot_dataset_100_answer.jsonl"
    
    # 创建数据集
    dataset = ThinkVLNDataset(
        action_data_path=action_data_path,
        cot_data_path=cot_data_path,
        image_root="/mnt/nvme/swx/dataset/R2R"
    )
    
    # 找2个action样本
    action_samples_found = []
    for idx in range(len(dataset)):
        if dataset.samples[idx]['data_type'] == 'action':
            action_samples_found.append(idx)
            if len(action_samples_found) >= 2:
                break
    
    for sample_num, sample_idx in enumerate(action_samples_found, 1):
        print(f"\n{'─'*80}")
        print(f"ACTION 样本 #{sample_num} (数据集索引: {sample_idx})")
        print(f"{'─'*80}")
        
        # 获取原始样本
        raw_sample = dataset.samples[sample_idx]
        
        print(f"\n1️⃣ 原始样本数据：")
        print(f"   ├─ data_type: {raw_sample['data_type']}")
        print(f"   ├─ episode_key: {raw_sample['episode_key']}")
        print(f"   ├─ frame_idx: {raw_sample['frame_idx']}")
        print(f"   ├─ instruction: {raw_sample['instruction']}")
        print(f"   ├─ current_plan_step: {raw_sample['current_plan_step']}")
        print(f"   ├─ current_subtask_idx: {raw_sample['current_subtask_idx']}")
        print(f"   ├─ actions (全序列): {raw_sample['actions']}")
        print(f"   └─ subtask_sequence: {raw_sample['subtask_sequence']}")
        
        # 计算label
        frame_idx = raw_sample['frame_idx']
        actions = raw_sample['actions']
        subtask_seq = raw_sample['subtask_sequence']
        action_chunk, progress_chunk = extract_action_chunk(frame_idx, actions, subtask_seq, num_steps=4)
        
        print(f"\n2️⃣ Collator处理后的标签（未经processor）：")
        print(f"   ├─ 下4步动作标签: {action_chunk}")
        print(f"   ├─ 下4步进度标签: {[f'{p:.2f}' for p in progress_chunk]}")
        print(f"   ├─ 动作含义:")
        ACTION_MAPPING = {0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right"}
        for i, action in enumerate(action_chunk):
            print(f"   │  └─ 步骤{i+1}: {action} ({ACTION_MAPPING.get(action, 'unknown')})")
        
        print(f"\n3️⃣ Collator处理前的message结构（Tokenization前）：")
        prompt = f"Based on the current observation and subgoal '{raw_sample['current_plan_step']}', predict the next 4 actions."
        print(f"   ├─ Prompt: {prompt}")
        print(f"   ├─ Image: [会从 {raw_sample['episode_key']} 加载]")
        print(f"   ├─ Chat格式:")
        print(f"   │  └─ {{")
        print(f"   │     'role': 'user',")
        print(f"   │     'content': [")
        print(f"   │       {{'type': 'image', 'image': <PIL.Image>}},")
        print(f"   │       {{'type': 'text', 'text': '{prompt}'}}")
        print(f"   │     ]")
        print(f"   │  }}")
        
        print(f"\n4️⃣ 预期的Processor处理后（Tokenization）：")
        print(f"   ├─ input_ids: [<prompt tokens> + <4 query tokens>]")
        print(f"   ├─ attention_mask: [1, 1, ..., 1, 1]")
        print(f"   ├─ pixel_values: <image tensor>")
        print(f"   ├─ image_grid_thw: <grid info>")
        print(f"   └─ action_labels + progress_labels (会被使用)")


def show_cot_samples():
    """展示CoT样本的处理流程"""
    
    print("\n\n" + "="*80)
    print("COT 样本处理流程详细展示")
    print("="*80)
    
    # 数据路径
    action_data_path = "/mnt/swx/ThinkVLN/data/trajectory_data/R2R_back/summary_full.jsonl"
    cot_data_path = "/mnt/swx/ThinkVLN/data/cot_dataset/cot_dataset_100_answer.jsonl"
    
    # 创建数据集
    dataset = ThinkVLNDataset(
        action_data_path=action_data_path,
        cot_data_path=cot_data_path,
        image_root="/mnt/nvme/swx/dataset/R2R"
    )
    
    # 找2个cot样本
    cot_samples_found = []
    for idx in range(len(dataset)):
        if dataset.samples[idx]['data_type'] == 'cot':
            cot_samples_found.append(idx)
            if len(cot_samples_found) >= 2:
                break
    
    for sample_num, sample_idx in enumerate(cot_samples_found, 1):
        print(f"\n{'─'*80}")
        print(f"COT 样本 #{sample_num} (数据集索引: {sample_idx})")
        print(f"{'─'*80}")
        
        # 获取原始样本
        raw_sample = dataset.samples[sample_idx]
        
        print(f"\n1️⃣ 原始样本数据：")
        print(f"   ├─ data_type: {raw_sample['data_type']}")
        print(f"   ├─ frame_key: {raw_sample['frame_key']}")
        print(f"   ├─ episode_key: {raw_sample['episode_key']}")
        print(f"   ├─ instruction: {raw_sample['instruction']}")
        print(f"   ├─ current_plan_step: {raw_sample['current_plan_step']}")
        print(f"   ├─ answer (推理文本):")
        for line in raw_sample['answer'].split('\n'):
            if line.strip():
                print(f"   │  └─ {line}")
        
        # 解析frame_key
        episode_key, step_id = parse_frame_key(raw_sample['frame_key'])
        print(f"\n2️⃣ Frame Key解析：")
        print(f"   ├─ 原始 frame_key: {raw_sample['frame_key']}")
        print(f"   ├─ 解析后 episode_key: {episode_key}")
        print(f"   └─ 解析后 step_id: {step_id}")
        
        print(f"\n3️⃣ Collator处理前的message结构（双pass tokenization）：")
        prompt = f"Based on the current observation and subgoal '{raw_sample['current_plan_step']}', think step by step to determine the action."
        full_text = prompt + "\n" + raw_sample['answer']
        
        print(f"\n   Pass 1 - 仅Prompt（用于计算mask边界）：")
        print(f"   ├─ Chat消息:")
        print(f"   │  └─ {{")
        print(f"   │     'role': 'user',")
        print(f"   │     'content': [")
        print(f"   │       {{'type': 'image', 'image': <PIL.Image>}},")
        print(f"   │       {{'type': 'text', 'text': '{prompt[:50]}...'}}")
        print(f"   │     ]")
        print(f"   │  }}")
        print(f"   └─ 目的: 获取 prompt_len (用于后续mask)")
        
        print(f"\n   Pass 2 - 完整文本（Prompt + Answer）：")
        print(f"   ├─ Chat消息:")
        print(f"   │  └─ {{")
        print(f"   │     'role': 'user',")
        print(f"   │     'content': [")
        print(f"   │       {{'type': 'image', 'image': <PIL.Image>}},")
        print(f"   │       {{'type': 'text', 'text': prompt + '\\n' + answer}}")
        print(f"   │     ]")
        print(f"   │  }}")
        print(f"   └─ 长度: {len(full_text)} 字符")
        
        print(f"\n4️⃣ Label生成（Mask策略）：")
        print(f"   ├─ labels 初始化为 input_ids 的复制")
        print(f"   ├─ labels[:prompt_len] = -100  (mask掉prompt部分)")
        print(f"   ├─ labels[prompt_len:] = <tokens>  (保留answer部分用于学习)")
        print(f"   └─ 说明: -100是HF中的ignore_index，在计算loss时会被忽略")
        
        print(f"\n5️⃣ 预期的Processor处理后（Tokenization）：")
        print(f"   ├─ input_ids: [<tokenized full_text>]")
        print(f"   ├─ attention_mask: [1, 1, ..., 1]")
        print(f"   ├─ labels: [-100, -100, ..., <answer_tokens>, <answer_tokens>]")
        print(f"   ├─ pixel_values: <image tensor>")
        print(f"   └─ image_grid_thw: <grid info>")


def show_data_flow_diagram():
    """展示数据流图"""
    
    print("\n\n" + "="*80)
    print("数据处理流图")
    print("="*80)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ThinkVLN 混合数据集处理流程                           │
└─────────────────────────────────────────────────────────────────────────────┘

                              📥 原始JSONL数据
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
            ┌─────▼──────┐  ┌──────▼──────┐  ┌───────▼────────┐
            │ 轨迹ID       │  │ 帧Key       │  │ 其他元数据      │
            │ 动作序列     │  │ 推理文本    │  │ 指令           │
            │ 子任务标签   │  │ 答案       │  │ 计划步骤       │
            └─────┬──────┘  └──────┬──────┘  └───────┬────────┘
                  │                 │                 │
                  └─────────────────┼─────────────────┘
                                    │
                    ┌───────────────▼────────────────┐
                    │   Dataset (原始数据层)          │
                    │  ✓ 加载JSONL                   │
                    │  ✓ 混合action/CoT样本          │
                    └───────────────┬────────────────┘
                                    │
                  ┌─────────────────▼─────────────────┐
                  │                                   │
            ┌─────▼──────────────┐  ┌─────────────────▼──────┐
            │ Action数据         │  │ CoT数据                │
            │                    │  │                        │
            │ • 加载图像         │  │ • 加载图像             │
            │ • 生成动作标签     │  │ • 创建双pass messages  │
            │ • 创建prompt      │  │ • 生成mask标签         │
            │ • Tokenize        │  │ • Tokenize             │
            │                    │  │                        │
            └─────┬──────────────┘  └────────────┬───────────┘
                  │                              │
                  └──────────────┬───────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Collator (Batch处理)   │
                    │                         │
                    │ ✓ Padding到相同长度    │
                    │ ✓ Stack多个样本        │
                    │ ✓ 混合type批处理       │
                    └────────────┬────────────┘
                                 │
                        📤 处理后的Batch
                    (ready for model input)
                    
    ┌──────────────────────────────────────────────────────────────────┐
    │                     本次展示的内容                               │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  ✅ 1️⃣ 原始样本数据（直接从JSONL读取）                            │
    │  ✅ 2️⃣ Collator处理后的结构（message格式，未经processor）         │
    │  ✅ 3️⃣ 详细的处理步骤说明                                        │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    """)


def main():
    """主函数"""
    print("\n")
    print("╔═════════════════════════════════════════════════════════════════╗")
    print("║         ThinkVLN 数据集处理流程 - 详细展示                        ║")
    print("║              原始样本 vs Collator处理后的Message               ║")
    print("╚═════════════════════════════════════════════════════════════════╝")
    
    show_action_samples()
    show_cot_samples()
    show_data_flow_diagram()
    
    print("\n" + "="*80)
    print("✅ 展示完成")
    print("="*80)
    print("\n📝 关键要点：")
    print("  • Action样本: 通过extract_action_chunk生成4步动作和进度标签")
    print("  • CoT样本: 通过双pass tokenization实现selective masking")
    print("  • 两种样本都包含图像、prompt和标签")
    print("  • Collator的职责是padding和batch处理")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
