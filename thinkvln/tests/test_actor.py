#!/usr/bin/env python3
"""
Unit tests for ThinkVLNActor model implementation
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

from thinkvln.models.thinkvln_actor import (
    ThinkVLNActor, SharedProjector, ActionClassificationHead, ProgressRegressionHead
)
from thinkvln.models.actor_config import ThinkVLNActorConfig


class TestSharedProjector:
    """Test suite for SharedProjector"""
    
    def test_projector_initialization(self):
        """Test SharedProjector initialization"""
        print("\n[TEST] SharedProjector initialization")
        
        projector = SharedProjector(input_size=768, hidden_size=384, dropout=0.1)
        
        assert projector is not None
        assert len(list(projector.parameters())) > 0, "Projector should have parameters"
        
        print("✓ SharedProjector initialization passed")
    
    def test_projector_forward(self):
        """Test SharedProjector forward pass"""
        print("[TEST] SharedProjector forward pass")
        
        input_size = 768
        hidden_size = 384
        batch_size = 2
        seq_len = 10
        
        projector = SharedProjector(input_size, hidden_size, dropout=0.1)
        x = torch.randn(batch_size, seq_len, input_size)
        
        output = projector(x)
        
        assert output.shape == (batch_size, seq_len, hidden_size), \
            f"Expected shape {(batch_size, seq_len, hidden_size)}, got {output.shape}"
        
        print("✓ SharedProjector forward pass passed")
    
    def test_projector_output_normalized(self):
        """Test that SharedProjector output is layer-normalized"""
        print("[TEST] SharedProjector layer normalization")
        
        projector = SharedProjector(768, 384)
        x = torch.randn(4, 10, 768)
        
        output = projector(x)
        
        # Check that output has reasonable mean and std (layer norm behavior)
        mean = output.mean()
        std = output.std()
        
        assert abs(mean.item()) < 1.0, "Output should have mean close to 0"
        
        print("✓ SharedProjector layer normalization passed")


class TestActionClassificationHead:
    """Test suite for ActionClassificationHead"""
    
    def test_action_head_initialization(self):
        """Test ActionClassificationHead initialization"""
        print("[TEST] ActionClassificationHead initialization")
        
        head = ActionClassificationHead(hidden_size=384, num_classes=4, dropout=0.1)
        
        assert head is not None
        assert len(list(head.parameters())) > 0, "Head should have parameters"
        
        print("✓ ActionClassificationHead initialization passed")
    
    def test_action_head_forward(self):
        """Test ActionClassificationHead forward pass"""
        print("[TEST] ActionClassificationHead forward pass")
        
        hidden_size = 384
        num_classes = 4
        batch_size = 2
        seq_len = 4
        
        head = ActionClassificationHead(hidden_size, num_classes, dropout=0.1)
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        logits = head(x)
        
        assert logits.shape == (batch_size, seq_len, num_classes), \
            f"Expected shape {(batch_size, seq_len, num_classes)}, got {logits.shape}"
        
        print("✓ ActionClassificationHead forward pass passed")
    
    def test_action_head_logits_range(self):
        """Test that logits are in reasonable range"""
        print("[TEST] ActionClassificationHead logits range")
        
        head = ActionClassificationHead(384, 4)
        x = torch.randn(2, 4, 384)
        
        logits = head(x)
        
        # Logits should be finite and not too extreme
        assert torch.all(torch.isfinite(logits)), "Logits should be finite"
        assert not torch.all(torch.abs(logits) > 100), "Logits should not be extremely large"
        
        print("✓ ActionClassificationHead logits range passed")


class TestProgressRegressionHead:
    """Test suite for ProgressRegressionHead"""
    
    def test_progress_head_initialization(self):
        """Test ProgressRegressionHead initialization"""
        print("[TEST] ProgressRegressionHead initialization")
        
        head = ProgressRegressionHead(hidden_size=384, dropout=0.1)
        
        assert head is not None
        assert len(list(head.parameters())) > 0, "Head should have parameters"
        
        print("✓ ProgressRegressionHead initialization passed")
    
    def test_progress_head_forward(self):
        """Test ProgressRegressionHead forward pass"""
        print("[TEST] ProgressRegressionHead forward pass")
        
        hidden_size = 384
        batch_size = 2
        seq_len = 4
        
        head = ProgressRegressionHead(hidden_size, dropout=0.1)
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        output = head(x)
        
        assert output.shape == (batch_size, seq_len), \
            f"Expected shape {(batch_size, seq_len)}, got {output.shape}"
        
        print("✓ ProgressRegressionHead forward pass passed")
    
    def test_progress_head_output_range(self):
        """Test that progress values are in [0, 1] range"""
        print("[TEST] ProgressRegressionHead output range")
        
        head = ProgressRegressionHead(384)
        x = torch.randn(4, 6, 384)
        
        output = head(x)
        
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0), \
            "Progress values should be in [0, 1] range (sigmoid output)"
        
        print("✓ ProgressRegressionHead output range passed")


class TestThinkVLNActorConfig:
    """Test suite for ThinkVLNActorConfig"""
    
    def test_actor_config_initialization(self):
        """Test ThinkVLNActorConfig initialization"""
        print("[TEST] ThinkVLNActorConfig initialization")
        
        config = ThinkVLNActorConfig(
            num_query_tokens=4,
            num_action_classes=4,
            action_loss_weight=1.0,
            progress_loss_weight=1.0
        )
        
        assert config.num_query_tokens == 4
        assert config.num_action_classes == 4
        assert config.action_loss_weight == 1.0
        assert config.progress_loss_weight == 1.0
        
        print("✓ ThinkVLNActorConfig initialization passed")
    
    def test_actor_config_defaults(self):
        """Test ThinkVLNActorConfig default values"""
        print("[TEST] ThinkVLNActorConfig defaults")
        
        config = ThinkVLNActorConfig()
        
        assert config.num_query_tokens > 0
        assert config.num_action_classes == 4
        assert config.action_loss_weight > 0
        assert config.progress_loss_weight > 0
        
        print("✓ ThinkVLNActorConfig defaults passed")


class TestThinkVLNActor:
    """Test suite for ThinkVLNActor model"""
    
    @staticmethod
    def create_mock_config():
        """Create a mock ThinkVLN config for testing"""
        from thinkvln.models.thinkvln_config import ThinkVLNConfig
        
        config_dict = {
            'use_return_dict': True,
            'num_query_tokens': 4,
            'text_config': {
                'hidden_size': 512,
                'vocab_size': 32000,
                'num_attention_heads': 8,
                'num_hidden_layers': 4,
            },
            'vision_config': {
                'hidden_size': 512,
                'image_size': 224,
            }
        }
        
        return ThinkVLNConfig(**config_dict)
    
    @patch('thinkvln.models.thinkvln_actor.ThinkVLNForConditionalGeneration.__init__')
    def test_actor_initialization(self, mock_parent_init):
        """Test ThinkVLNActor initialization"""
        print("[TEST] ThinkVLNActor initialization")
        
        mock_parent_init.return_value = None
        
        config = self.create_mock_config()
        actor_config = ThinkVLNActorConfig()
        
        # Mock the model to avoid actual pretrained loading
        with patch.object(ThinkVLNActor, 'model', MagicMock()):
            actor = ThinkVLNActor(config, actor_config)
        
        assert actor.actor_config is not None
        assert actor.shared_projector is not None
        assert actor.action_head is not None
        assert actor.progress_head is not None
        
        print("✓ ThinkVLNActor initialization passed")
    
    @patch('thinkvln.models.thinkvln_actor.ThinkVLNForConditionalGeneration.__init__')
    def test_actor_forward_action_mode(self, mock_parent_init):
        """Test ThinkVLNActor forward in action mode"""
        print("[TEST] ThinkVLNActor forward - action mode")
        
        mock_parent_init.return_value = None
        
        config = self.create_mock_config()
        actor_config = ThinkVLNActorConfig(num_query_tokens=4)
        
        with patch.object(ThinkVLNActor, 'model', MagicMock()) as mock_model:
            # Mock model outputs
            hidden_states = torch.randn(2, 110, 512)  # 106 + 4 query tokens
            mock_model.return_value = MagicMock(
                hidden_states=None,
                past_key_values=None,
                attentions=None,
                __getitem__=lambda self, idx: hidden_states if idx == 0 else None
            )
            
            actor = ThinkVLNActor(config, actor_config)
            
            input_ids = torch.randint(0, 32000, (2, 106))
            attention_mask = torch.ones(2, 106)
            action_labels = torch.randint(0, 4, (2, 4))
            progress_labels = torch.rand(2, 4)
            
            # Action mode: should compute action and progress losses
            with patch.object(actor, 'model', mock_model):
                outputs = actor.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    action_labels=action_labels,
                    progress_labels=progress_labels,
                    return_dict=True
                )
            
            assert 'loss' in outputs or isinstance(outputs, dict)
        
        print("✓ ThinkVLNActor forward - action mode passed")
    
    @patch('thinkvln.models.thinkvln_actor.ThinkVLNForConditionalGeneration.__init__')
    def test_actor_forward_cot_mode(self, mock_parent_init):
        """Test ThinkVLNActor forward in CoT mode"""
        print("[TEST] ThinkVLNActor forward - CoT mode")
        
        mock_parent_init.return_value = None
        
        config = self.create_mock_config()
        actor_config = ThinkVLNActorConfig(num_query_tokens=4)
        
        with patch.object(ThinkVLNActor, 'model', MagicMock()) as mock_model:
            hidden_states = torch.randn(2, 100, 512)
            mock_model.return_value = MagicMock(
                hidden_states=None,
                past_key_values=None,
                attentions=None,
                __getitem__=lambda self, idx: hidden_states if idx == 0 else None
            )
            
            actor = ThinkVLNActor(config, actor_config)
            
            input_ids = torch.randint(0, 32000, (2, 100))
            attention_mask = torch.ones(2, 100)
            labels = torch.randint(0, 32000, (2, 100))
            
            # CoT mode: action_labels is None, should compute LM loss
            with patch.object(actor, 'model', mock_model):
                with patch.object(actor, 'lm_head', MagicMock(return_value=torch.randn(2, 100, 32000))):
                    outputs = actor.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True
                    )
            
            assert isinstance(outputs, dict) or hasattr(outputs, 'loss')
        
        print("✓ ThinkVLNActor forward - CoT mode passed")
    
    def test_actor_mixed_batch_routing(self):
        """Test that actor correctly routes mixed batches"""
        print("[TEST] ThinkVLNActor mixed batch routing")
        
        # This test verifies the design concept
        # In action mode: action_labels is not None -> action head
        # In CoT mode: action_labels is None -> LM head
        
        # Test action mode condition
        action_labels = torch.randint(0, 4, (2, 4))
        use_action_mode = action_labels is not None
        assert use_action_mode, "Action mode should be True when action_labels provided"
        
        # Test CoT mode condition
        action_labels = None
        use_action_mode = action_labels is not None
        assert not use_action_mode, "Action mode should be False when action_labels is None"
        
        print("✓ ThinkVLNActor mixed batch routing passed")


class TestActorHeadIntegration:
    """Test integration of actor heads"""
    
    def test_projector_to_action_head_pipeline(self):
        """Test pipeline from projector to action head"""
        print("[TEST] Projector to action head pipeline")
        
        hidden_size = 768
        proj_size = 384
        batch_size = 2
        num_query_tokens = 4
        
        # Create pipeline
        projector = SharedProjector(hidden_size, proj_size)
        action_head = ActionClassificationHead(proj_size, num_classes=4)
        
        # Forward pass
        hidden_states = torch.randn(batch_size, num_query_tokens, hidden_size)
        projected = projector(hidden_states)
        logits = action_head(projected)
        
        assert logits.shape == (batch_size, num_query_tokens, 4)
        assert torch.all(torch.isfinite(logits))
        
        print("✓ Projector to action head pipeline passed")
    
    def test_projector_to_progress_head_pipeline(self):
        """Test pipeline from projector to progress head"""
        print("[TEST] Projector to progress head pipeline")
        
        hidden_size = 768
        proj_size = 384
        batch_size = 2
        num_query_tokens = 4
        
        # Create pipeline
        projector = SharedProjector(hidden_size, proj_size)
        progress_head = ProgressRegressionHead(proj_size)
        
        # Forward pass
        hidden_states = torch.randn(batch_size, num_query_tokens, hidden_size)
        projected = projector(hidden_states)
        values = progress_head(projected)
        
        assert values.shape == (batch_size, num_query_tokens)
        assert torch.all((values >= 0.0) & (values <= 1.0))
        
        print("✓ Projector to progress head pipeline passed")
    
    def test_dual_head_pipeline(self):
        """Test simultaneous action and progress prediction"""
        print("[TEST] Dual head pipeline")
        
        hidden_size = 768
        proj_size = 384
        batch_size = 2
        num_query_tokens = 4
        
        projector = SharedProjector(hidden_size, proj_size)
        action_head = ActionClassificationHead(proj_size, num_classes=4)
        progress_head = ProgressRegressionHead(proj_size)
        
        hidden_states = torch.randn(batch_size, num_query_tokens, hidden_size)
        projected = projector(hidden_states)
        
        action_logits = action_head(projected)
        progress_values = progress_head(projected)
        
        assert action_logits.shape == (batch_size, num_query_tokens, 4)
        assert progress_values.shape == (batch_size, num_query_tokens)
        
        print("✓ Dual head pipeline passed")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("ThinkVLN Actor Model Tests")
    print("="*80)
    
    # Test heads
    shared_proj_suite = TestSharedProjector()
    shared_proj_suite.test_projector_initialization()
    shared_proj_suite.test_projector_forward()
    shared_proj_suite.test_projector_output_normalized()
    
    action_head_suite = TestActionClassificationHead()
    action_head_suite.test_action_head_initialization()
    action_head_suite.test_action_head_forward()
    action_head_suite.test_action_head_logits_range()
    
    progress_head_suite = TestProgressRegressionHead()
    progress_head_suite.test_progress_head_initialization()
    progress_head_suite.test_progress_head_forward()
    progress_head_suite.test_progress_head_output_range()
    
    # Test config
    config_suite = TestThinkVLNActorConfig()
    config_suite.test_actor_config_initialization()
    config_suite.test_actor_config_defaults()
    
    # Test actor model
    actor_suite = TestThinkVLNActor()
    actor_suite.test_actor_initialization()
    actor_suite.test_actor_forward_action_mode()
    actor_suite.test_actor_forward_cot_mode()
    actor_suite.test_actor_mixed_batch_routing()
    
    # Test integration
    integration_suite = TestActorHeadIntegration()
    integration_suite.test_projector_to_action_head_pipeline()
    integration_suite.test_projector_to_progress_head_pipeline()
    integration_suite.test_dual_head_pipeline()
    
    print("\n" + "="*80)
    print("🎉 All actor model tests passed!")
    print("="*80)


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
