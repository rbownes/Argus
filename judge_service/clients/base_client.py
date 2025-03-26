"""
Base client interface for LLM providers.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import os
import logging
import uuid
from datetime import datetime

class BaseProviderClient(ABC):
    """Base client interface for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, api_key_env_var: Optional[str] = None):
        """
        Initialize the provider client.
        
        Args:
            api_key: API key for the provider (optional)
            api_key_env_var: Environment variable name for API key (optional)
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Set up API key
        self.api_key = api_key
        if not self.api_key and api_key_env_var:
            self.api_key = os.environ.get(api_key_env_var)
            
        if not self.api_key:
            self.logger.warning(f"API key not found. Provider client may not function properly.")
            
    @abstractmethod
    async def generate_content(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using the provider's API.
        
        Args:
            model_id: ID of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with response content:
            {
                "id": str,  # Unique ID for this completion
                "model_id": str,  # ID of the model used
                "content": str,  # Generated content
                "usage": {  # Token usage information
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int
                },
                "created_at": datetime,  # Timestamp of creation
                "provider": str  # Provider identifier
            }
        """
        pass

    @abstractmethod
    def format_messages(self, messages: List[Dict[str, str]]) -> Any:
        """
        Format messages for the provider's expected structure.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted messages in the provider's expected format
        """
        pass
        
    def create_response_id(self) -> str:
        """
        Create a unique ID for a completion response.
        
        Returns:
            UUID string
        """
        return str(uuid.uuid4())
        
    def create_response_object(
        self,
        model_id: str,
        content: str,
        usage: Dict[str, int],
        provider: str,
        response_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized response object.
        
        Args:
            model_id: ID of the model used
            content: Generated content
            usage: Token usage information
            provider: Provider identifier
            response_id: Optional response ID (will generate if not provided)
            
        Returns:
            Standardized response dictionary
        """
        return {
            "id": response_id or self.create_response_id(),
            "model_id": model_id,
            "content": content,
            "usage": usage,
            "created_at": datetime.utcnow(),
            "provider": provider
        }
