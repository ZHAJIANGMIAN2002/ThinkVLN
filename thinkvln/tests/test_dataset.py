#!/usr/bin/env python3
"""
Unit tests for ThinkVLNDataset and ThinkVLNDataCollator
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from thinkvln.dataset.dataset import ThinkVLNDataset, ThinkVLNDataCollator, ACTION_MAPPING
from thinkvln.tools.dataset_utils import extract_action_chunk, parse_frame_key, crop_cot_answer


class TestThinkVLNDataset:
    """Test suite for ThinkVLNDataset"""
    
    @staticmethod
    def create_mock_action_data(num_samples: int = 5) -> str:
        """Create mock action data JSONL file"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        
        for i in range(num_samples):
            sample = {
                'episode_key': f'scene001_{i:05d}',  # 格式: scene_id_episode_id
                'num_frames': 10,
                'instruction': f'Navigate to room {i}',
                'plan': [f'Step {j}' for j in range(3)],
                'actions': [1, 1, 2, 1, 3, 1, 0, 0, 0, 0],
                'subtask_sequence': [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            }
            temp_file.write(json.dumps(sample) + '\n')
        
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def create_mock_cot_data(num_samples: int = 5) -> str:
        """Create mock CoT data JSONL file"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        
        for i in range(num_samples):
            sample = {
                'frame_key': f'scene001_{i:05d}_000001',  # 格式: scene_id_episode_id_step_id
                'episode_key': f'scene001_{i:05d}',
                'instruction': f'Navigate to room {i}',
                'plan': '1. Move forward\n2. Turn left\n3. Continue forward',
                'answer': '[causal observation] I see a door ahead, so I should move forward.',
                'ground_truth_subtask': 1,
            }
            temp_file.write(json.dumps(sample) + '\n')
        
        temp_file.close()
        return temp_file.name
    
    def test_dataset_initialization_action_only(self):
        """Test dataset initialization with action data only"""
        print("\n[TEST] Dataset initialization with action data only")
        
        action_data_path = self.create_mock_action_data(5)
        
        try:
            dataset = ThinkVLNDataset(
                action_data_path=action_data_path,
                cot_data_path=None,
                image_root="/tmp"
            )
            
            # Dataset creates frame-level samples from trajectories
            # 5 trajectories with 10 frames each = 50 samples
            assert len(dataset.action_samples) == 50, f"Expected 50 action samples (5 traj * 10 frames), got {len(dataset.action_samples)}"
            assert len(dataset.cot_samples) == 0, f"Expected 0 CoT samples, got {len(dataset.cot_samples)}"
            assert len(dataset.samples) == 50, f"Expected 50 total samples, got {len(dataset.samples)}"
            assert dataset.samples[0]['data_type'] == 'action'
            
            print("✓ Action-only dataset initialization passed")
        finally:
            os.unlink(action_data_path)
    
    def test_dataset_initialization_cot_only(self):
        """Test dataset initialization with CoT data only"""
        print("[TEST] Dataset initialization with CoT data only")
        
        cot_data_path = self.create_mock_cot_data(5)
        
        try:
            dataset = ThinkVLNDataset(
                action_data_path=None,
                cot_data_path=cot_data_path,
                image_root="/tmp"
            )
            
            assert len(dataset.action_samples) == 0, f"Expected 0 action samples, got {len(dataset.action_samples)}"
            assert len(dataset.cot_samples) == 5, f"Expected 5 CoT samples, got {len(dataset.cot_samples)}"
            assert len(dataset.samples) == 5, f"Expected 5 total samples, got {len(dataset.samples)}"
            assert dataset.samples[0]['data_type'] == 'cot'
            
            print("✓ CoT-only dataset initialization passed")
        finally:
            os.unlink(cot_data_path)
    
    def test_dataset_initialization_mixed(self):
        """Test dataset initialization with mixed action and CoT data"""
        print("[TEST] Dataset initialization with mixed data")
        
        action_data_path = self.create_mock_action_data(5)
        cot_data_path = self.create_mock_cot_data(5)
        
        try:
            dataset = ThinkVLNDataset(
                action_data_path=action_data_path,
                cot_data_path=cot_data_path,
                image_root="/tmp"
            )
            
            assert len(dataset.action_samples) == 50, f"Expected 50 action samples (5 traj * 10 frames)"
            assert len(dataset.cot_samples) == 5, f"Expected 5 CoT samples"
            assert len(dataset.samples) == 55, f"Expected 55 total samples, got {len(dataset.samples)}"
            
            # Check that both types are present
            data_types = [s['data_type'] for s in dataset.samples]
            assert 'action' in data_types, "Missing action samples in mixed dataset"
            assert 'cot' in data_types, "Missing CoT samples in mixed dataset"
            
            print("✓ Mixed dataset initialization passed")
        finally:
            os.unlink(action_data_path)
            os.unlink(cot_data_path)
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method"""
        print("[TEST] Dataset __getitem__")
        
        action_data_path = self.create_mock_action_data(3)
        
        try:
            dataset = ThinkVLNDataset(
                action_data_path=action_data_path,
                image_root="/tmp"
            )
            
            sample = dataset[0]
            
            assert isinstance(sample, dict), "Sample should be a dictionary"
            assert 'data_type' in sample, "Sample should contain data_type"
            assert sample['data_type'] == 'action'
            assert 'episode_key' in sample
            assert 'instruction' in sample
            assert 'actions' in sample
            
            print("✓ Dataset __getitem__ passed")
        finally:
            os.unlink(action_data_path)
    
    def test_dataset_length(self):
        """Test dataset __len__ method"""
        print("[TEST] Dataset __len__")
        
        action_data_path = self.create_mock_action_data(7)
        
        try:
            dataset = ThinkVLNDataset(
                action_data_path=action_data_path,
                image_root="/tmp"
            )
            
            # 7 trajectories * 10 frames = 70 samples
            assert len(dataset) == 70, f"Expected length 70, got {len(dataset)}"
            
            print("✓ Dataset __len__ passed")
        finally:
            os.unlink(action_data_path)
    
    def test_empty_dataset(self):
        """Test dataset with no data"""
        print("[TEST] Empty dataset")
        
        dataset = ThinkVLNDataset(
            action_data_path=None,
            cot_data_path=None,
            image_root="/tmp"
        )
        
        assert len(dataset) == 0, "Empty dataset should have length 0"
        assert len(dataset.samples) == 0
        
        print("✓ Empty dataset handling passed")


class TestThinkVLNDataCollator:
    """Test suite for ThinkVLNDataCollator"""
    
    @staticmethod
    def create_mock_processor():
        """Create mock processor for testing"""
        processor = MagicMock()
        processor.tokenizer.pad_token_id = 0
        processor.apply_chat_template = MagicMock(return_value="mock_text")
        
        # Mock processor call
        def mock_processor_call(**kwargs):
            batch_size = len(kwargs.get('text', [1]))
            seq_len = 100
            
            return {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len),
                'pixel_values': torch.randn(batch_size, 3, 224, 224),
                'image_grid_thw': torch.tensor([[8, 8, 144]]),
            }
        
        processor.side_effect = mock_processor_call
        processor.__call__ = mock_processor_call
        
        return processor
    
    def test_collator_initialization(self):
        """Test data collator initialization"""
        print("\n[TEST] Collator initialization")
        
        processor = self.create_mock_processor()
        
        collator = ThinkVLNDataCollator(
            processor=processor,
            num_query_tokens=4,
            query_token_id=151700,
            image_root="/tmp"
        )
        
        assert collator.num_query_tokens == 4
        assert collator.query_token_id == 151700
        assert collator.processor is not None
        
        print("✓ Collator initialization passed")
    
    @patch('thinkvln.dataset.dataset.load_image')
    def test_collator_process_action_sample(self, mock_load_image):
        """Test processing action samples"""
        print("[TEST] Collator process action sample")
        
        # Create mock image
        mock_image = Image.new('RGB', (224, 224), color='red')
        mock_load_image.return_value = mock_image
        
        processor = self.create_mock_processor()
        collator = ThinkVLNDataCollator(
            processor=processor,
            num_query_tokens=4,
            image_root="/tmp"
        )
        
        sample = {
            'data_type': 'action',
            'episode_key': 'scene001_00001',  # 正确的格式
            'frame_idx': 0,
            'instruction': 'Navigate forward',
            'current_plan_step': 'Move ahead',
            'current_subtask_idx': 1,
            'actions': [1, 1, 2, 1],
            'subtask_sequence': [1, 1, 1, 1],
        }
        
        processed = collator._process_action(sample)
        
        assert 'input_ids' in processed
        assert 'attention_mask' in processed
        assert 'action_labels' in processed
        assert 'progress_labels' in processed
        assert processed['data_type'] == 'action'
        
        print("✓ Action sample processing passed")
    
    @patch('thinkvln.dataset.dataset.load_image')
    def test_collator_process_cot_sample(self, mock_load_image):
        """Test processing CoT samples"""
        print("[TEST] Collator process CoT sample")
        
        # Create mock image
        mock_image = Image.new('RGB', (224, 224), color='blue')
        mock_load_image.return_value = mock_image
        
        processor = self.create_mock_processor()
        collator = ThinkVLNDataCollator(
            processor=processor,
            num_query_tokens=4,
            image_root="/tmp"
        )
        
        sample = {
            'data_type': 'cot',
            'frame_key': 'scene001_00001_000001',
            'episode_key': 'scene001_00001',
            'instruction': 'Navigate forward',
            'current_plan_step': 'Move ahead',
            'answer': '[causal observation] I see a corridor ahead.',
        }
        
        processed = collator._process_cot(sample)
        
        assert 'input_ids' in processed
        assert 'attention_mask' in processed
        assert 'labels' in processed
        assert processed['labels'] is not None
        assert processed['action_labels'] is None
        assert processed['data_type'] == 'cot'
        
        print("✓ CoT sample processing passed")
    
    @patch('thinkvln.dataset.dataset.load_image')
    def test_collator_collate_action_batch(self, mock_load_image):
        """Test collating action samples"""
        print("[TEST] Collator collate action batch")
        
        mock_image = Image.new('RGB', (224, 224), color='green')
        mock_load_image.return_value = mock_image
        
        processor = self.create_mock_processor()
        collator = ThinkVLNDataCollator(processor=processor, image_root="/tmp")
        
        batch = [
            {
                'data_type': 'action',
                'episode_key': 'scene001_00001',
                'frame_idx': i,
                'instruction': 'Navigate forward',
                'current_plan_step': 'Move ahead',
                'current_subtask_idx': 1,
                'actions': [1, 1, 2, 1],
                'subtask_sequence': [1, 1, 1, 1],
            }
            for i in range(2)
        ]
        
        result = collator(batch)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'action_labels' in result
        assert 'progress_labels' in result
        assert result['input_ids'].shape[0] == 2, "Batch size should be 2"
        
        print("✓ Action batch collation passed")


class TestDatasetUtils:
    """Test suite for dataset utility functions"""
    
    def test_extract_action_chunk_basic(self):
        """Test action chunk extraction basic case"""
        print("\n[TEST] Extract action chunk basic")
        
        frame_idx = 0
        actions = [1, 2, 3, 1, 0, 0]
        subtask_sequence = [1, 1, 1, 2, 2, 2]
        
        action_chunk, progress_chunk = extract_action_chunk(
            frame_idx, actions, subtask_sequence, num_steps=4
        )
        
        assert len(action_chunk) == 4, f"Expected 4 actions, got {len(action_chunk)}"
        assert len(progress_chunk) == 4, f"Expected 4 progress values, got {len(progress_chunk)}"
        assert all(isinstance(a, int) for a in action_chunk)
        assert all(isinstance(p, float) for p in progress_chunk)
        
        print("✓ Extract action chunk basic passed")
    
    def test_extract_action_chunk_boundary(self):
        """Test action chunk extraction at subtask boundary"""
        print("[TEST] Extract action chunk at boundary")
        
        frame_idx = 2
        actions = [1, 2, 3, 1, 0, 0]
        subtask_sequence = [1, 1, 1, 2, 2, 2]
        
        action_chunk, progress_chunk = extract_action_chunk(
            frame_idx, actions, subtask_sequence, num_steps=4
        )
        
        # Should include action at frame 3 (boundary) as stop (0)
        assert action_chunk[0] == 0, "Action at subtask boundary should be 0 (stop)"
        
        print("✓ Extract action chunk boundary handling passed")
    
    def test_extract_action_chunk_progress_values(self):
        """Test progress value computation"""
        print("[TEST] Extract action chunk progress values")
        
        frame_idx = 0
        actions = [1, 1, 1, 1, 1]
        subtask_sequence = [1, 1, 1, 1, 1]
        
        _, progress_chunk = extract_action_chunk(
            frame_idx, actions, subtask_sequence, num_steps=4
        )
        
        # Progress should increase from 0 to 1 within subtask
        assert progress_chunk[0] < progress_chunk[-1], "Progress should increase"
        assert all(0.0 <= p <= 1.0 for p in progress_chunk), "Progress should be in [0, 1]"
        
        print("✓ Extract action chunk progress values passed")
    
    def test_parse_frame_key(self):
        """Test frame key parsing"""
        print("[TEST] Parse frame key")
        
        frame_key = "17DRP5sb8fy_10154_000035"
        episode_key, step_id = parse_frame_key(frame_key)
        
        assert "17DRP5sb8fy" in episode_key
        assert step_id == 35, f"Expected step_id 35, got {step_id}"
        
        print("✓ Parse frame key passed")
    
    def test_crop_cot_answer(self):
        """Test CoT answer cropping"""
        print("[TEST] Crop CoT answer")
        
        answer = "Some text here [causal observation] The actual reasoning starts here."
        cropped = crop_cot_answer(answer)
        
        assert cropped.startswith("[causal observation]")
        assert "actual reasoning" in cropped
        
        # Test answer without marker
        answer_no_marker = "Just some text without marker"
        cropped_no_marker = crop_cot_answer(answer_no_marker)
        assert cropped_no_marker == answer_no_marker
        
        print("✓ Crop CoT answer passed")
    
    def test_action_mapping(self):
        """Test action mapping consistency"""
        print("[TEST] Action mapping")
        
        assert ACTION_MAPPING[0] == "stop"
        assert ACTION_MAPPING[1] == "forward"
        assert ACTION_MAPPING[2] == "turn_left"
        assert ACTION_MAPPING[3] == "turn_right"
        
        print("✓ Action mapping passed")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("ThinkVLN Dataset and Collator Tests")
    print("="*80)
    
    test_suite = TestThinkVLNDataset()
    test_suite.test_dataset_initialization_action_only()
    test_suite.test_dataset_initialization_cot_only()
    test_suite.test_dataset_initialization_mixed()
    test_suite.test_dataset_getitem()
    test_suite.test_dataset_length()
    test_suite.test_empty_dataset()
    
    collator_suite = TestThinkVLNDataCollator()
    collator_suite.test_collator_initialization()
    collator_suite.test_collator_process_action_sample()
    collator_suite.test_collator_process_cot_sample()
    collator_suite.test_collator_collate_action_batch()
    
    utils_suite = TestDatasetUtils()
    utils_suite.test_extract_action_chunk_basic()
    utils_suite.test_extract_action_chunk_boundary()
    utils_suite.test_extract_action_chunk_progress_values()
    utils_suite.test_parse_frame_key()
    utils_suite.test_crop_cot_answer()
    utils_suite.test_action_mapping()
    
    print("\n" + "="*80)
    print("🎉 All dataset tests passed!")
    print("="*80)


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
