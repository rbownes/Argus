"""
Adapter for LiteLLM as a fallback integration.
"""
from .base import ModelAdapter
from typing import Dict, List, Optional, Any
import litellm
import logging
import uuid
import asyncio

class LiteLLMAdapter(ModelAdapter):
    """Adapter for LiteLLM integration."""
    
    async def complete(
        self, 
        model_id: str,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion using LiteLLM.
        
        Args:
            model_id: ID of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            model_config: Model-specific configuration
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with completion results
        """
        try:
            # Configure LiteLLM
            litellm.verbose = False
            
            # Format model string for LiteLLM based on provider
            provider_id = self.provider_config.get('litellm_provider') or self.provider_config.get('id')
            
            # Check if this is a Gemini model but we should use the Gemini adapter instead
            if (provider_id.lower() == "gemini" or "gemini" in model_id.lower()) and self.provider_config.get('adapter') == 'gemini':
                self.logger.error(f"Attempted to use LiteLLM with Gemini model {model_id}, but this provider is configured to use the dedicated Gemini adapter")
                raise ValueError(f"Gemini models should use the dedicated Gemini adapter, not LiteLLM. Please check your provider configuration.")
                
            # Special handling for Gemini models that are explicitly configured to use LiteLLM
            if provider_id.lower() == "google" or "gemini" in model_id.lower():
                litellm_model = f"google/{model_id}"
            else:
                litellm_model = f"{provider_id}/{model_id}"
                
            self.logger.info(f"Using LiteLLM to complete with model: {litellm_model}")
            
            # Run query through LiteLLM
            response = await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                temperature=temperature or model_config.get("temperature_default", 0.7),
                max_tokens=max_tokens or model_config.get("max_tokens_default", 1024)
            )
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Extract token usage
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return {
                "id": str(uuid.uuid4()),
                "model_id": model_id,
                "provider_id": self.provider_config["id"],
                "content": content,
                "usage": usage,
                "raw_response": response
            }
        except Exception as e:
            self.logger.error(f"LiteLLM completion error: {str(e)}")
            # Include model and provider details in error message to help with debugging
            error_msg = (f"LiteLLM error with model {model_id} from provider "
                        f"{self.provider_config['id']}: {str(e)}")
            raise ValueError(error_msg)
