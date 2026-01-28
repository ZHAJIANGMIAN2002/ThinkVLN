"""ThinkVLN configuration classes extending Qwen3VL"""

from transformers import Qwen3VLConfig
from typing import Optional


class ThinkVLNConfig(Qwen3VLConfig):
    """Configuration class for ThinkVLN models with action query token support"""
    
    model_type = "thinkvln"
    
    def __init__(
        self,
        action_query_token_id: int = 151700,
        num_query_tokens: int = 4,
        **kwargs
    ):
        """
        Initialize ThinkVLN configuration.
        
        Args:
            action_query_token_id: Token ID for action query placeholder (default: 151700, an unused ID)
            num_query_tokens: Number of learnable query tokens for action/progress prediction
            **kwargs: Additional arguments passed to Qwen3VLConfig
        """
        super().__init__(**kwargs)
        self.action_query_token_id = action_query_token_id
        self.num_query_tokens = num_query_tokens
