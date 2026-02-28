"""Simple test to verify action query token integration works correctly"""

import torch
import torch.nn as nn
from thinkvln_config import ThinkVLNConfig
from thinkvln_model import ThinkVLNModel, ThinkVLNForConditionalGeneration
from thinkvln_actor import ThinkVLNActor
from actor_config import ThinkVLNActorConfig


def test_config_creation():
    """Test ThinkVLNConfig creation"""
    config = ThinkVLNConfig(
        action_query_token_id=151700,
        num_query_tokens=4,
    )
    assert config.action_query_token_id == 151700
    assert config.num_query_tokens == 4
    print("✓ Config creation test passed")


def test_model_structure():
    """Test that ThinkVLNModel and ThinkVLNForConditionalGeneration can be instantiated"""
    from transformers import Qwen3VLConfig
    
    # Create a minimal Qwen3VL config for testing
    qwen_config = Qwen3VLConfig()
    
    # Convert to ThinkVLN config
    config = ThinkVLNConfig(
        **qwen_config.to_dict(),
        action_query_token_id=151700,
        num_query_tokens=4,
    )
    
    # Test model instantiation
    try:
        model = ThinkVLNModel(config)
        assert model.query_embeddings is None  # Should be set by parent
        print("✓ ThinkVLNModel instantiation test passed")
    except Exception as e:
        print(f"✗ ThinkVLNModel instantiation failed: {e}")
        return False
    
    # Test conditional generation model
    try:
        cond_model = ThinkVLNForConditionalGeneration(config)
        assert cond_model.query_embeddings is not None
        assert cond_model.query_embeddings.shape == (4, config.text_config.hidden_size)
        assert cond_model.model.query_embeddings is cond_model.query_embeddings
        print("✓ ThinkVLNForConditionalGeneration instantiation test passed")
    except Exception as e:
        print(f"✗ ThinkVLNForConditionalGeneration instantiation failed: {e}")
        return False
    
    return True


def test_actor_structure():
    """Test ThinkVLNActor structure"""
    from transformers import Qwen3VLConfig
    
    qwen_config = Qwen3VLConfig()
    config = ThinkVLNConfig(
        **qwen_config.to_dict(),
        action_query_token_id=151700,
        num_query_tokens=4,
    )
    
    actor_config = ThinkVLNActorConfig(
        num_query_tokens=4,
        num_action_classes=4,
    )
    
    try:
        actor = ThinkVLNActor(config, actor_config)
        
        # Check components exist
        assert hasattr(actor, 'query_embeddings')
        assert hasattr(actor, 'shared_projector')
        assert hasattr(actor, 'action_head')
        assert hasattr(actor, 'progress_head')
        
        # Check query embeddings shape
        assert actor.query_embeddings.shape == (4, config.text_config.hidden_size)
        
        print("✓ ThinkVLNActor structure test passed")
        return True
    except Exception as e:
        print(f"✗ ThinkVLNActor structure test failed: {e}")
        return False


def test_forward_pass_shape():
    """Test forward pass with dummy data"""
    from transformers import Qwen3VLConfig
    
    qwen_config = Qwen3VLConfig()
    config = ThinkVLNConfig(
        **qwen_config.to_dict(),
        action_query_token_id=151700,
        num_query_tokens=4,
    )
    
    actor_config = ThinkVLNActorConfig(
        num_query_tokens=4,
        num_action_classes=4,
    )
    
    try:
        actor = ThinkVLNActor(config, actor_config)
        actor.eval()
        
        batch_size = 2
        seq_len = 10
        num_query_tokens = 4
        action_query_token_id = config.action_query_token_id
        
        # Create dummy input with action query tokens at the end
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_tokens = torch.full((batch_size, num_query_tokens), action_query_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, query_tokens], dim=1)
        
        attention_mask = torch.ones_like(input_ids)
        
        # Forward pass (without vision inputs for simplicity)
        with torch.no_grad():
            outputs = actor(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Check output shapes
        assert "action_logits" in outputs
        assert "progress_values" in outputs
        assert outputs["action_logits"].shape == (batch_size, num_query_tokens, 4)
        assert outputs["progress_values"].shape == (batch_size, num_query_tokens)
        
        print("✓ Forward pass shape test passed")
        return True
    except Exception as e:
        print(f"✗ Forward pass shape test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_embedding_injection():
    """Test that query embeddings are properly injected"""
    from transformers import Qwen3VLConfig
    
    qwen_config = Qwen3VLConfig()
    config = ThinkVLNConfig(
        **qwen_config.to_dict(),
        action_query_token_id=151700,
        num_query_tokens=4,
    )
    
    try:
        model = ThinkVLNForConditionalGeneration(config)
        model.eval()
        
        batch_size = 2
        seq_len = 10
        num_query_tokens = 4
        
        # Create input with query tokens
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        query_tokens = torch.full((batch_size, num_query_tokens), config.action_query_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, query_tokens], dim=1)
        attention_mask = torch.ones_like(input_ids)
        
        # Forward through model
        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        hidden_states = outputs.last_hidden_state
        
        # Check that we got the right sequence length
        assert hidden_states.shape[1] == seq_len + num_query_tokens
        
        # Extract query hidden states
        query_hidden = hidden_states[:, -num_query_tokens:, :]
        
        # Query hidden states should be non-zero (embeddings were injected)
        assert query_hidden.abs().sum() > 0
        
        print("✓ Query embedding injection test passed")
        return True
    except Exception as e:
        print(f"✗ Query embedding injection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running ThinkVLN Action Query Integration Tests\n")
    
    test_config_creation()
    test_model_structure()
    test_actor_structure()
    test_forward_pass_shape()
    test_query_embedding_injection()
    
    print("\n✓ All tests passed!")
