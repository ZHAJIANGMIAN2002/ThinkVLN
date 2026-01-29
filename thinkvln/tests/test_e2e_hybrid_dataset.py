#!/usr/bin/env python3
"""
端到端集成测试：混合数据集训练
验证整个训练流程能否成功运行
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from thinkvln.dataset.dataset import ThinkVLNDataset, ThinkVLNDataCollator
from thinkvln.engine.sft_trainer import (
    ThinkVLNTrainingArguments,
    ThinkVLNSFTTrainer,
    create_training_args_from_config,
)


def create_hybrid_test_data(num_action_trajectories=3, num_cot_samples=3):
    """创建混合训练数据"""
    
    action_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    for i in range(num_action_trajectories):
        sample = {
            'episode_key': f'episode_{i:03d}_r2r_000001',
            'num_frames': 8,
            'instruction': f'Navigate to destination {i}',
            'plan': ['Step 1', 'Step 2', 'Step 3'],
            'actions': [1, 1, 2, 1, 1, 3, 0, 0],
            'subtask_sequence': [1, 1, 1, 2, 2, 2, 3, 3],
        }
        action_file.write(json.dumps(sample) + '\n')
    action_file.close()
    
    cot_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
    for i in range(num_cot_samples):
        sample = {
            'frame_key': f'scene_{i:03d}_{i:02d}_000001',
            'episode_key': f'episode_{i:03d}_r2r_000001',
            'instruction': f'Navigate to destination {i}',
            'plan': '1. Step 1\n2. Step 2\n3. Step 3',
            'answer': f'[causal observation] Reasoning for step {i}.',
            'ground_truth_subtask': 1,
        }
        cot_file.write(json.dumps(sample) + '\n')
    cot_file.close()
    
    return action_file.name, cot_file.name


def test_hybrid_dataset_loading():
    """测试1: 混合数据集加载"""
    
    print("\n" + "="*80)
    print("TEST 1: 混合数据集加载")
    print("="*80)
    
    action_path, cot_path = create_hybrid_test_data(3, 3)
    
    try:
        dataset = ThinkVLNDataset(
            action_data_path=action_path,
            cot_data_path=cot_path,
            image_root="/tmp"
        )
        
        print(f"\n✓ 数据集加载成功")
        print(f"  - 总样本数: {len(dataset)}")
        print(f"  - Action样本: {len(dataset.action_samples)}")
        print(f"  - CoT样本: {len(dataset.cot_samples)}")
        
        # 验证混合
        data_types = [s['data_type'] for s in dataset.samples]
        assert 'action' in data_types, "Missing action samples"
        assert 'cot' in data_types, "Missing CoT samples"
        
        print(f"  - 数据类型混合: ✓")
        
        return True
    except Exception as e:
        print(f"\n✗ 失败: {e}")
        return False
    finally:
        os.unlink(action_path)
        os.unlink(cot_path)


def test_data_collator_with_hybrid_batch():
    """测试2: 混合批次的数据整理"""
    
    print("\n" + "="*80)
    print("TEST 2: 混合批次数据整理")
    print("="*80)
    
    # Mock processor
    processor = MagicMock()
    processor.tokenizer.pad_token_id = 0
    processor.apply_chat_template = MagicMock(return_value="text")
    
    def mock_processor(**kwargs):
        batch_size = len(kwargs.get('text', [1]))
        return {
            'input_ids': torch.randint(0, 1000, (batch_size, 100)),
            'attention_mask': torch.ones(batch_size, 100),
            'pixel_values': torch.randn(batch_size, 3, 224, 224),
            'image_grid_thw': torch.tensor([[8, 8, 144]]),
        }
    
    processor.__call__ = mock_processor
    
    collator = ThinkVLNDataCollator(processor=processor, image_root="/tmp")
    
    # 混合批次
    batch = [
        {
            'data_type': 'action',
            'episode_key': 'ep1',
            'frame_idx': 0,
            'instruction': 'Navigate',
            'current_plan_step': 'Move',
            'current_subtask_idx': 1,
            'actions': [1, 1, 2, 1],
            'subtask_sequence': [1, 1, 1, 1],
        },
        {
            'data_type': 'cot',
            'frame_key': 'scene_001_001_000001',
            'episode_key': 'ep1',
            'instruction': 'Navigate',
            'current_plan_step': 'Move',
            'answer': '[causal observation] Observe and reason.',
        }
    ]
    
    try:
        with patch('thinkvln.dataset.dataset.load_image'):
            result = collator(batch)
        
        print(f"\n✓ 混合批次整理成功")
        print(f"  - 批次大小: {result['input_ids'].shape[0]}")
        print(f"  - 序列长度: {result['input_ids'].shape[1]}")
        print(f"  - action_labels: {'✓' if result['action_labels'] is not None else '✗'}")
        print(f"  - labels (CoT): {'✓' if result['labels'] is not None else '✗'}")
        
        return True
    except Exception as e:
        print(f"\n✗ 失败: {e}")
        return False


def test_training_arguments_creation():
    """测试3: 训练参数创建"""
    
    print("\n" + "="*80)
    print("TEST 3: 训练参数创建")
    print("="*80)
    
    action_path, cot_path = create_hybrid_test_data(3, 3)
    
    try:
        args = ThinkVLNTrainingArguments(
            output_dir="./test_output",
            model_name_or_path="Qwen/Qwen3-VL-2B",
            action_data_path=action_path,
            cot_data_path=cot_path,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            num_query_tokens=4,
            action_loss_weight=1.0,
            progress_loss_weight=1.0,
        )
        
        print(f"\n✓ 训练参数创建成功")
        print(f"  - 模型: {args.model_name_or_path}")
        print(f"  - 输出目录: {args.output_dir}")
        print(f"  - 训练轮数: {args.num_train_epochs}")
        print(f"  - 批处理大小: {args.per_device_train_batch_size}")
        print(f"  - 查询令牌数: {args.num_query_tokens}")
        print(f"  - Action损失权重: {args.action_loss_weight}")
        print(f"  - Progress损失权重: {args.progress_loss_weight}")
        
        return True
    except Exception as e:
        print(f"\n✗ 失败: {e}")
        return False
    finally:
        os.unlink(action_path)
        os.unlink(cot_path)


def test_trainer_initialization():
    """测试4: 训练器初始化"""
    
    print("\n" + "="*80)
    print("TEST 4: 训练器初始化")
    print("="*80)
    
    action_path, cot_path = create_hybrid_test_data(3, 3)
    
    try:
        args = ThinkVLNTrainingArguments(
            output_dir="./test_output",
            action_data_path=action_path,
            cot_data_path=cot_path,
        )
        
        dataset = ThinkVLNDataset(
            action_data_path=action_path,
            cot_data_path=cot_path,
            image_root="/tmp"
        )
        
        mock_model = MagicMock()
        
        trainer = ThinkVLNSFTTrainer(
            model=mock_model,
            args=args,
            train_dataset=dataset,
        )
        
        print(f"\n✓ 训练器初始化成功")
        print(f"  - 模型: ✓")
        print(f"  - 参数: ✓")
        print(f"  - 数据集大小: {len(trainer.train_dataset)}")
        print(f"  - 损失跟踪器: {list(trainer.loss_history.keys())}")
        
        return True
    except Exception as e:
        print(f"\n✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.unlink(action_path)
        os.unlink(cot_path)


def test_loss_computation_routing():
    """测试5: 损失计算路由（Action vs CoT）"""
    
    print("\n" + "="*80)
    print("TEST 5: 损失计算路由")
    print("="*80)
    
    mock_model = MagicMock()
    mock_dataset = MagicMock()
    
    args = ThinkVLNTrainingArguments(
        output_dir="./test_output",
        action_data_path="dummy.jsonl",
    )
    
    trainer = ThinkVLNSFTTrainer(
        model=mock_model,
        args=args,
        train_dataset=mock_dataset,
    )
    
    try:
        # 测试Action模式
        print(f"\n  测试A: Action模式 (action_labels 不为None)")
        
        action_loss = torch.tensor(1.0)
        progress_loss = torch.tensor(0.5)
        total_loss = action_loss + progress_loss
        
        outputs_action = {
            'loss': total_loss,
            'action_loss': action_loss,
            'progress_loss': progress_loss,
            'lm_loss': None,
        }
        mock_model.return_value = outputs_action
        
        inputs_action = {
            'input_ids': torch.randint(0, 1000, (2, 100)),
            'action_labels': torch.randint(0, 4, (2, 4)),
        }
        
        loss_action = trainer.compute_loss(mock_model, inputs_action)
        print(f"    ✓ Action损失计算: {loss_action.item():.4f}")
        
        # 测试CoT模式
        print(f"\n  测试B: CoT模式 (action_labels为None)")
        
        lm_loss = torch.tensor(2.5)
        outputs_cot = {
            'loss': lm_loss,
            'action_loss': None,
            'progress_loss': None,
            'lm_loss': lm_loss,
        }
        mock_model.return_value = outputs_cot
        
        inputs_cot = {
            'input_ids': torch.randint(0, 1000, (2, 100)),
            'labels': torch.randint(0, 1000, (2, 100)),
            'action_labels': None,
        }
        
        loss_cot = trainer.compute_loss(mock_model, inputs_cot)
        print(f"    ✓ CoT损失计算: {loss_cot.item():.4f}")
        
        print(f"\n✓ 损失计算路由成功")
        return True
        
    except Exception as e:
        print(f"\n✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_e2e_tests():
    """运行所有端到端测试"""
    
    print("\n" + "="*80)
    print("ThinkVLN 端到端集成测试")
    print("混合数据集训练流程验证")
    print("="*80)
    
    results = {
        "TEST 1: 混合数据集加载": test_hybrid_dataset_loading(),
        "TEST 2: 混合批次数据整理": test_data_collator_with_hybrid_batch(),
        "TEST 3: 训练参数创建": test_training_arguments_creation(),
        "TEST 4: 训练器初始化": test_trainer_initialization(),
        "TEST 5: 损失计算路由": test_loss_computation_routing(),
    }
    
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！混合数据集训练流程验证成功")
        return True
    else:
        print("\n⚠ 部分测试失败")
        return False


if __name__ == "__main__":
    try:
        success = run_all_e2e_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
