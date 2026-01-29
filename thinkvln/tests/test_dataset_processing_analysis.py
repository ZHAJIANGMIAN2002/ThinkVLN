#!/usr/bin/env python3
"""
数据集处理过程详细分析
展示原始数据和处理后的dataset item
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import tempfile
from pathlib import Path

from thinkvln.dataset.dataset import ThinkVLNDataset
from thinkvln.tools.dataset_utils import extract_action_chunk, parse_frame_key, crop_cot_answer


def create_sample_data():
    """创建示例数据文件"""
    
    # Action样本原始数据
    action_raw = {
        'episode_key': 'room_001_r2r_000042',
        'num_frames': 10,
        'instruction': '走到客厅左边的椅子旁边',
        'plan': ['先走向前方门口', '进入客厅', '转向左边寻找椅子'],
        'actions': [1, 1, 1, 2, 1, 1, 3, 0, 0, 0],
        'subtask_sequence': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
    }
    
    # CoT样本原始数据
    cot_raw = {
        'frame_key': 'room_001_042_000005',
        'episode_key': 'room_001_r2r_000042',
        'instruction': '走到客厅左边的椅子旁边',
        'plan': '1. 先走向前方门口\n2. 进入客厅\n3. 转向左边寻找椅子',
        'answer': '[causal observation] 我看到前方有一个开放的门口，右边是一堵墙。根据指令，我需要先走向前方。因此，下一步应该是向前走。',
        'ground_truth_subtask': 2,
    }
    
    return action_raw, cot_raw


def analyze_action_sample():
    """分析Action样本的处理过程"""
    
    print("\n" + "="*80)
    print("ACTION 样本处理分析")
    print("="*80)
    
    action_raw, _ = create_sample_data()
    
    print("\n1. 原始 ACTION 数据:")
    print("-" * 80)
    print(json.dumps(action_raw, indent=2, ensure_ascii=False))
    
    # 创建临时文件存储数据
    action_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    action_file.write(json.dumps(action_raw) + '\n')
    action_file.close()
    
    try:
        # 加载数据集
        dataset = ThinkVLNDataset(
            action_data_path=action_file.name,
            cot_data_path=None,
            image_root="/tmp"
        )
        
        print("\n2. 数据集统计:")
        print("-" * 80)
        print(f"总样本数: {len(dataset)}")
        print(f"  - Action样本: {len(dataset.action_samples)}")
        print(f"  - CoT样本: {len(dataset.cot_samples)}")
        print(f"\n轨迹信息:")
        print(f"  - 轨迹长度: {action_raw['num_frames']} 帧")
        print(f"  - 创建的样本数: {len(dataset.action_samples)} (每一帧一个样本)")
        
        print("\n3. 处理后的 Dataset Items (前3个示例):")
        print("-" * 80)
        
        for idx in range(min(3, len(dataset))):
            item = dataset[idx]
            print(f"\n样本 #{idx}:")
            print(f"  数据类型: {item['data_type']}")
            print(f"  episode_key: {item['episode_key']}")
            print(f"  frame_idx: {item['frame_idx']}")
            print(f"  instruction: {item['instruction']}")
            print(f"  current_plan_step: {item['current_plan_step']}")
            print(f"  current_subtask_idx: {item['current_subtask_idx']}")
            print(f"  actions (序列): {item['actions']}")
            print(f"  subtask_sequence: {item['subtask_sequence']}")
            
            # 分析该帧的动作标签
            frame_idx = item['frame_idx']
            actions = item['actions']
            subtask_seq = item['subtask_sequence']
            
            action_chunk, progress_chunk = extract_action_chunk(
                frame_idx, actions, subtask_seq, num_steps=4
            )
            
            print(f"\n  动作标签提取 (从frame {frame_idx}):")
            print(f"    - 下4步动作: {action_chunk}")
            print(f"    - 下4步进度: {[f'{p:.2f}' for p in progress_chunk]}")
        
        print("\n4. 所有样本分析:")
        print("-" * 80)
        print(f"总共生成了 {len(dataset)} 个样本")
        print("\n样本索引与帧索引的映射:")
        
        for idx in range(len(dataset)):
            item = dataset[idx]
            frame_idx = item['frame_idx']
            subtask_idx = item['current_subtask_idx']
            print(f"  样本{idx}: frame_idx={frame_idx}, subtask_idx={subtask_idx}, plan_step='{item['current_plan_step']}'")
        
    finally:
        os.unlink(action_file.name)


def analyze_cot_sample():
    """分析CoT样本的处理过程"""
    
    print("\n" + "="*80)
    print("COT 样本处理分析")
    print("="*80)
    
    _, cot_raw = create_sample_data()
    
    print("\n1. 原始 COT 数据:")
    print("-" * 80)
    print(json.dumps(cot_raw, indent=2, ensure_ascii=False))
    
    # 创建临时文件存储数据
    cot_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    cot_file.write(json.dumps(cot_raw) + '\n')
    cot_file.close()
    
    try:
        # 加载数据集
        dataset = ThinkVLNDataset(
            action_data_path=None,
            cot_data_path=cot_file.name,
            image_root="/tmp"
        )
        
        print("\n2. 数据集统计:")
        print("-" * 80)
        print(f"总样本数: {len(dataset)}")
        print(f"  - Action样本: {len(dataset.action_samples)}")
        print(f"  - CoT样本: {len(dataset.cot_samples)}")
        
        print("\n3. 处理后的 Dataset Item:")
        print("-" * 80)
        
        item = dataset[0]
        print(f"\n样本 #0:")
        print(f"  数据类型: {item['data_type']}")
        print(f"  frame_key: {item['frame_key']}")
        print(f"  episode_key: {item['episode_key']}")
        print(f"  instruction: {item['instruction']}")
        print(f"  current_plan_step: {item['current_plan_step']}")
        print(f"  answer (原始):")
        print(f"    {cot_raw['answer']}")
        print(f"  answer (处理后):")
        print(f"    {item['answer']}")
        
        # 分析答案处理
        print("\n4. 答案处理分析:")
        print("-" * 80)
        print(f"原始答案长度: {len(cot_raw['answer'])} 字符")
        print(f"处理后答案长度: {len(item['answer'])} 字符")
        
        marker = "[causal observation]"
        if marker in cot_raw['answer']:
            idx = cot_raw['answer'].index(marker)
            print(f"\n标记 '{marker}' 位置: {idx}")
            print(f"标记前内容被移除: {cot_raw['answer'][:idx]}")
            print(f"保留内容: {cot_raw['answer'][idx:idx+50]}...")
        
        # 分析frame_key解析
        print("\n5. Frame Key 解析:")
        print("-" * 80)
        frame_key = item['frame_key']
        episode_key, step_id = parse_frame_key(frame_key)
        print(f"原始 frame_key: {frame_key}")
        print(f"解析后:")
        print(f"  - episode_key: {episode_key}")
        print(f"  - step_id: {step_id}")
        
    finally:
        os.unlink(cot_file.name)


def analyze_hybrid_dataset():
    """分析混合数据集（Action + CoT）"""
    
    print("\n" + "="*80)
    print("混合数据集 (ACTION + COT) 处理分析")
    print("="*80)
    
    action_raw, cot_raw = create_sample_data()
    
    # 创建临时文件
    action_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    action_file.write(json.dumps(action_raw) + '\n')
    action_file.close()
    
    cot_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    cot_file.write(json.dumps(cot_raw) + '\n')
    cot_file.close()
    
    try:
        # 加载混合数据集
        dataset = ThinkVLNDataset(
            action_data_path=action_file.name,
            cot_data_path=cot_file.name,
            image_root="/tmp"
        )
        
        print("\n1. 数据集统计:")
        print("-" * 80)
        print(f"总样本数: {len(dataset)}")
        print(f"  - Action样本: {len(dataset.action_samples)}")
        print(f"  - CoT样本: {len(dataset.cot_samples)}")
        print(f"混合比例: {len(dataset.action_samples)} action : {len(dataset.cot_samples)} cot")
        
        print("\n2. 样本类型分布:")
        print("-" * 80)
        data_types = {}
        for sample in dataset.samples:
            dtype = sample['data_type']
            data_types[dtype] = data_types.get(dtype, 0) + 1
        
        for dtype, count in data_types.items():
            percentage = (count / len(dataset)) * 100
            print(f"  {dtype}: {count} 个 ({percentage:.1f}%)")
        
        print("\n3. 前5个样本预览:")
        print("-" * 80)
        for idx in range(min(5, len(dataset))):
            item = dataset[idx]
            dtype = item['data_type']
            if dtype == 'action':
                print(f"样本{idx}: [ACTION] frame_idx={item['frame_idx']}, plan='{item['current_plan_step']}'")
            else:
                print(f"样本{idx}: [COT] frame_key={item['frame_key']}, answer_len={len(item['answer'])}")
        
    finally:
        os.unlink(action_file.name)
        os.unlink(cot_file.name)


def save_analysis_report(output_file):
    """保存分析报告到文件"""
    
    print(f"\n正在保存分析报告到: {output_file}")
    
    action_raw, cot_raw = create_sample_data()
    
    report = {
        "title": "ThinkVLN 数据集处理完整分析",
        "description": "展示原始数据和处理后的dataset item结果",
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        
        "action_sample": {
            "raw_input": action_raw,
            "processed_items": [],
            "stats": {}
        },
        
        "cot_sample": {
            "raw_input": cot_raw,
            "processed_item": None,
            "stats": {}
        }
    }
    
    # 处理Action样本
    action_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    action_file.write(json.dumps(action_raw) + '\n')
    action_file.close()
    
    try:
        dataset = ThinkVLNDataset(
            action_data_path=action_file.name,
            cot_data_path=None,
            image_root="/tmp"
        )
        
        report["action_sample"]["stats"] = {
            "total_samples": len(dataset),
            "trajectory_frames": action_raw['num_frames'],
            "samples_per_trajectory": len(dataset)
        }
        
        for idx in range(min(len(dataset), 3)):
            item = dataset[idx]
            frame_idx = item['frame_idx']
            actions = item['actions']
            subtask_seq = item['subtask_sequence']
            
            action_chunk, progress_chunk = extract_action_chunk(
                frame_idx, actions, subtask_seq, num_steps=4
            )
            
            report["action_sample"]["processed_items"].append({
                "sample_idx": idx,
                "frame_idx": frame_idx,
                "data_type": item['data_type'],
                "episode_key": item['episode_key'],
                "instruction": item['instruction'],
                "current_plan_step": item['current_plan_step'],
                "current_subtask_idx": item['current_subtask_idx'],
                "action_labels": action_chunk,
                "progress_labels": progress_chunk,
            })
    finally:
        os.unlink(action_file.name)
    
    # 处理CoT样本
    cot_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    cot_file.write(json.dumps(cot_raw) + '\n')
    cot_file.close()
    
    try:
        dataset = ThinkVLNDataset(
            action_data_path=None,
            cot_data_path=cot_file.name,
            image_root="/tmp"
        )
        
        report["cot_sample"]["stats"] = {
            "total_samples": len(dataset),
        }
        
        item = dataset[0]
        episode_key, step_id = parse_frame_key(item['frame_key'])
        
        report["cot_sample"]["processed_item"] = {
            "data_type": item['data_type'],
            "frame_key": item['frame_key'],
            "episode_key": item['episode_key'],
            "instruction": item['instruction'],
            "current_plan_step": item['current_plan_step'],
            "answer_original": cot_raw['answer'],
            "answer_processed": item['answer'],
            "answer_processing": {
                "original_length": len(cot_raw['answer']),
                "processed_length": len(item['answer']),
                "marker_found": "[causal observation]" in cot_raw['answer'],
            },
            "frame_key_parsing": {
                "frame_key": item['frame_key'],
                "parsed_episode_key": episode_key,
                "parsed_step_id": step_id,
            }
        }
    finally:
        os.unlink(cot_file.name)
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 报告已保存: {output_file}")
    print(f"  文件大小: {os.path.getsize(output_file)} 字节")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("ThinkVLN 数据集处理详细分析")
    print("="*80)
    
    # 运行分析
    analyze_action_sample()
    analyze_cot_sample()
    analyze_hybrid_dataset()
    
    # 保存报告到文件
    output_dir = "/mnt/swx/ThinkVLN/thinkvln/tests"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dataset_processing_analysis.json")
    
    save_analysis_report(output_file)
    
    # 还保存一份易读的文本版本
    output_text_file = os.path.join(output_dir, "dataset_processing_analysis.txt")
    with open(output_text_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ThinkVLN 数据集处理完整分析报告\n")
        f.write("="*80 + "\n\n")
        
        action_raw, cot_raw = create_sample_data()
        
        f.write("1. ACTION 样本\n")
        f.write("-"*80 + "\n")
        f.write("原始数据:\n")
        f.write(json.dumps(action_raw, indent=2, ensure_ascii=False) + "\n\n")
        
        action_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        action_file.write(json.dumps(action_raw) + '\n')
        action_file.close()
        
        try:
            dataset = ThinkVLNDataset(
                action_data_path=action_file.name,
                cot_data_path=None,
                image_root="/tmp"
            )
            
            f.write(f"处理后数据集信息:\n")
            f.write(f"  - 总样本数: {len(dataset)}\n")
            f.write(f"  - 轨迹帧数: {action_raw['num_frames']}\n")
            f.write(f"  - 生成样本数/帧: {len(dataset) // action_raw['num_frames']}\n\n")
            
            f.write("处理后的Dataset Items (前3个):\n\n")
            for idx in range(min(3, len(dataset))):
                item = dataset[idx]
                frame_idx = item['frame_idx']
                actions = item['actions']
                subtask_seq = item['subtask_sequence']
                action_chunk, progress_chunk = extract_action_chunk(
                    frame_idx, actions, subtask_seq, num_steps=4
                )
                
                f.write(f"样本 #{idx}:\n")
                f.write(f"  - frame_idx: {frame_idx}\n")
                f.write(f"  - current_subtask_idx: {item['current_subtask_idx']}\n")
                f.write(f"  - current_plan_step: {item['current_plan_step']}\n")
                f.write(f"  - 下4步动作标签: {action_chunk}\n")
                f.write(f"  - 下4步进度标签: {[f'{p:.2f}' for p in progress_chunk]}\n\n")
        finally:
            os.unlink(action_file.name)
        
        f.write("\n2. COT 样本\n")
        f.write("-"*80 + "\n")
        f.write("原始数据:\n")
        f.write(json.dumps(cot_raw, indent=2, ensure_ascii=False) + "\n\n")
        
        cot_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        cot_file.write(json.dumps(cot_raw) + '\n')
        cot_file.close()
        
        try:
            dataset = ThinkVLNDataset(
                action_data_path=None,
                cot_data_path=cot_file.name,
                image_root="/tmp"
            )
            
            item = dataset[0]
            episode_key, step_id = parse_frame_key(item['frame_key'])
            
            f.write(f"处理后的Dataset Item:\n\n")
            f.write(f"  - data_type: {item['data_type']}\n")
            f.write(f"  - frame_key: {item['frame_key']}\n")
            f.write(f"  - 解析后 episode_key: {episode_key}\n")
            f.write(f"  - 解析后 step_id: {step_id}\n")
            f.write(f"  - instruction: {item['instruction']}\n")
            f.write(f"  - current_plan_step: {item['current_plan_step']}\n\n")
            
            f.write(f"答案处理:\n")
            f.write(f"  - 原始答案长度: {len(cot_raw['answer'])} 字符\n")
            f.write(f"  - 处理后答案长度: {len(item['answer'])} 字符\n\n")
            f.write(f"  原始答案:\n")
            f.write(f"    {cot_raw['answer']}\n\n")
            f.write(f"  处理后答案:\n")
            f.write(f"    {item['answer']}\n")
        finally:
            os.unlink(cot_file.name)
    
    print(f"✓ 易读版本已保存: {output_text_file}")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
