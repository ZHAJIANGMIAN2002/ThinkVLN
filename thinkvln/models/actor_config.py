"""Configuration class for ThinkVLNActor model"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ThinkVLNActorConfig:
    """Configuration for ThinkVLNActor model"""
    
    # Action space configuration
    num_action_classes: int = 4  # forward, turn_left, turn_right, stop
    
    # Progress regression configuration
    progress_min: float = 0.0
    progress_max: float = 1.0
    
    # Learnable query embeddings configuration (replaces action token search)
    num_query_tokens: int = 4  # Number of learnable query tokens for action/progress prediction
    
    # Shared projector configuration for semantic->control signal conversion
    projector_hidden_size: Optional[int] = None  # If None, uses hidden_size // 2
    projector_dropout: float = 0.1
    
    # Head architecture configuration
    action_head_dropout: float = 0.1
    progress_head_dropout: float = 0.1
    
    # Loss weights
    action_loss_weight: float = 1.0
    progress_loss_weight: float = 1.0
    done_loss_weight: float = 1.0
    use_huber_loss_for_progress: bool = False  # Huber (SmoothL1) more robust than MSE for outliers
    
    # Training configuration
    freeze_llm: bool = False  # Whether to freeze the LLM during training

    def get_action_token_names(self) -> List[str]:
        """DEPRECATED: Get list of action token names. Use learnable query embeddings instead."""
        import warnings
        warnings.warn("get_action_token_names is deprecated. Use learnable query embeddings instead.", 
                     DeprecationWarning, stacklevel=2)
        return [f"[{self.action_token_prefix}{i+1}]" for i in range(self.num_action_tokens)]
    
    def get_action_token_ids(self, tokenizer) -> List[int]:
        """DEPRECATED: Get token IDs for action tokens from tokenizer. Use learnable query embeddings instead."""
        import warnings
        warnings.warn("get_action_token_ids is deprecated. Use learnable query embeddings instead.", 
                     DeprecationWarning, stacklevel=2)
        token_names = self.get_action_token_names()
        token_ids = []
        for token_name in token_names:
            if token_name in tokenizer.get_vocab():
                token_ids.append(tokenizer.convert_tokens_to_ids(token_name))
            else:
                raise ValueError(f"Action token {token_name} not found in tokenizer vocabulary. "
                               f"Please add it using tokenizer.add_tokens([{token_name}])")
        return token_ids
