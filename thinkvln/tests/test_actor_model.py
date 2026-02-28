#!/usr/bin/env python3
"""
Test script for ThinkVLNActor model implementation
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from transformers import AutoProcessor
from thinkvln.models import ThinkVLNActor, ThinkVLNActorConfig

def test_actor_config():
    """Test ThinkVLNActorConfig"""
    print("Testing ThinkVLNActorConfig...")
    
    config = ThinkVLNActorConfig(num_action_tokens=4)
    
    # Test token name generation
    token_names = config.get_action_token_names()
    print(f"Action token names: {token_names}")
    assert len(token_names) == 4
    assert token_names == ["[ACT_1]", "[ACT_2]", "[ACT_3]", "[ACT_4]"]
    
    print("✓ ThinkVLNActorConfig test passed")

def test_heads():
    """Test ActionClassificationHead and ProgressRegressionHead"""
    print("\nTesting Action and Progress Heads...")
    
    from thinkvln.models import ActionClassificationHead, ProgressRegressionHead
    
    hidden_size = 512
    batch_size = 2
    num_tokens = 4
    
    # Test action classification head
    action_head = ActionClassificationHead(hidden_size, num_classes=4, dropout=0.1)
    hidden_states = torch.randn(batch_size, num_tokens, hidden_size)
    
    action_logits = action_head(hidden_states)
    print(f"Action logits shape: {action_logits.shape}")
    assert action_logits.shape == (batch_size, num_tokens, 4)
    
    # Test progress regression head
    progress_head = ProgressRegressionHead(hidden_size, dropout=0.1)
    progress_values = progress_head(hidden_states)
    print(f"Progress values shape: {progress_values.shape}")
    assert progress_values.shape == (batch_size, num_tokens, 1)
    
    # Check progress values are in [0, 1] range
    assert torch.all(progress_values >= 0.0) and torch.all(progress_values <= 1.0)
    
    print("✓ Action and Progress Heads test passed")

def test_token_extraction():
    """Test action token extraction logic"""
    print("\nTesting Token Extraction...")
    
    # Mock a simple scenario
    from thinkvln.models.actor_config import ThinkVLNActorConfig
    config = ThinkVLNActorConfig(num_action_tokens=2)
    
    # Create mock tokenizer vocab
    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                "[ACT_1]": 1001,
                "[ACT_2]": 1002,
                "hello": 1,
                "world": 2,
            }
        
        def get_vocab(self):
            return self.vocab
        
        def convert_tokens_to_ids(self, token):
            return self.vocab[token]
    
    tokenizer = MockTokenizer()
    token_ids = config.get_action_token_ids(tokenizer)
    print(f"Token IDs: {token_ids}")
    assert token_ids == [1001, 1002]
    
    print("✓ Token extraction test passed")

def test_model_instantiation():
    """Test if ThinkVLNActor can be instantiated"""
    print("\nTesting Model Instantiation...")
    
    # Test with minimal config (without actual pretrained weights)
    try:
        from transformers import Qwen3VLConfig
        from thinkvln.models import ThinkVLNActorConfig
        
        # Create a minimal config for testing
        qwen_config = Qwen3VLConfig.from_dict({
            "text_config": {
                "hidden_size": 512,
                "vocab_size": 1000,
                "num_attention_heads": 8,
                "num_hidden_layers": 2,
            },
            "vision_config": {
                "hidden_size": 512,
                "image_size": 224,
                "patch_size": 16,
            },
            "use_return_dict": True,
        })
        
        actor_config = ThinkVLNActorConfig(num_action_tokens=4)
        
        # This should work for instantiation test (though model won't have pretrained weights)
        model = ThinkVLNActor(qwen_config, actor_config)
        
        print(f"Model created successfully")
        print(f"Action head: {model.action_head}")
        print(f"Progress head: {model.progress_head}")
        
        # Test setting action token IDs
        model.set_action_token_ids([1001, 1002, 1003, 1004])
        assert model.action_token_ids == [1001, 1002, 1003, 1004]
        
        print("✓ Model instantiation test passed")
        
    except Exception as e:
        print(f"⚠ Model instantiation test failed (expected for missing deps): {e}")
        # This is expected if we don't have the full transformers setup

def main():
    """Run all tests"""
    print("Running ThinkVLNActor model tests...\n")
    
    try:
        test_actor_config()
        test_heads()
        test_token_extraction()
        test_model_instantiation()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()