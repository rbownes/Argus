"""
Base adapter interface for LLM providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import os
import logging

class ModelAdapter(ABC):
    """Base adapter interface for LLM providers."""
    
    def __init__(self, provider_config: Dict[str, Any]):
        """
        Initialize the adapter with provider configuration.
        
        Args:
            provider_config: Provider configuration dictionary
        """
        self.provider_config = provider_config
        self.logger = logging.getLogger(f"model_adapter.{provider_config['id']}")
        self.api_key = os.environ.get(provider_config['env_var_key'])
        
        if not self.api_key and provider_config.get('requires_api_key', True):
            self.logger.warning(f"API key not found in environment variable {provider_config['env_var_key']}")
            
    @abstractmethod
    async def complete(
        self, 
        model_id: str,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion for the given messages.
        
        Args:
            model_id: ID of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            model_config: Model-specific configuration
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with completion results
        """
        pass
        
    @classmethod
    def format_messages(cls, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format messages for the provider's expected structure.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted messages
        """
        # Default implementation passes through messages unchanged
        return messages
