"""
LLM Service implementation using LiteLLM.
"""
import asyncio
from typing import Dict, List, Any, Optional
import time

from llm_eval.core.models import Prompt, LLMResponse, LLMProvider
from llm_eval.core.utils import Result, generate_id, measure_latency
from .interface import LLMServiceInterface

# Import litellm conditionally to handle environments where it's not installed
try:
    import litellm
except ImportError:
    litellm = None


class LiteLLMService(LLMServiceInterface):
    """
    LLM Service implementation using LiteLLM.
    
    LiteLLM provides a unified interface to multiple LLM providers.
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the LiteLLM service.
        
        Args:
            api_keys: Dictionary mapping provider names to API keys.
                     Example: {"openai": "sk-...", "anthropic": "sk-..."}
        """
        if litellm is None:
            raise ImportError(
                "LiteLLM is not installed. Install it with 'pip install litellm'"
            )
        
        self.api_keys = api_keys or {}
        
        # Set up API keys
        for provider, key in self.api_keys.items():
            if provider == "openai":
                litellm.openai_api_key = key
            elif provider == "anthropic":
                litellm.anthropic_api_key = key
            elif provider == "cohere":
                litellm.cohere_api_key = key
            # Add other providers as needed
    
    def _get_provider_from_model(self, model_name: str) -> LLMProvider:
        """
        Determine the provider from the model name.
        
        Args:
            model_name: The name of the model.
            
        Returns:
            The provider enum value.
        """
        model_name = model_name.lower()
        
        if any(prefix in model_name for prefix in ["gpt", "text-davinci", "openai"]):
            return LLMProvider.OPENAI
        elif any(prefix in model_name for prefix in ["claude", "anthropic"]):
            return LLMProvider.ANTHROPIC
        elif "cohere" in model_name:
            return LLMProvider.COHERE
        elif "mistral" in model_name:
            return LLMProvider.MISTRAL
        elif any(prefix in model_name for prefix in ["palm", "gemini", "bard"]):
            return LLMProvider.GOOGLE
        else:
            return LLMProvider.OTHER
    
    async def query_model(
        self,
        model_name: str,
        prompt: Prompt,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Result[LLMResponse]:
        """Query a single LLM model with a prompt."""
        try:
            provider = self._get_provider_from_model(model_name)
            
            # Prepare parameters
            params = parameters or {}
            default_params = {
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            # Merge default and provided parameters
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
            
            # Query the model and measure latency
            start_time = time.time()
            response = await litellm.acompletion(
                model=model_name,
                messages=[{"role": "user", "content": prompt.text}],
                **params
            )
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Extract and process the response
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            # Create the response object
            llm_response = LLMResponse(
                id=generate_id(),
                prompt_id=prompt.id or generate_id(),
                prompt_text=prompt.text,
                model_name=model_name,
                provider=provider,
                response_text=response_text,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                    **{f"parameter_{k}": v for k, v in params.items()}
                }
            )
            
            return Result.ok(llm_response)
        except Exception as e:
            return Result.err(e)
    
    async def batch_query(
        self,
        model_names: List[str],
        prompts: List[Prompt],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Result[List[LLMResponse]]:
        """Query multiple LLM models with multiple prompts."""
        try:
            results = []
            
            # Create a task for each model-prompt pair
            tasks = []
            for model_name in model_names:
                for prompt in prompts:
                    tasks.append(self.query_model(model_name, prompt, parameters))
            
            # Run tasks concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            for response in responses:
                if isinstance(response, Result):
                    if response.is_ok:
                        results.append(response.unwrap())
                    else:
                        # Log the error but continue processing
                        print(f"Error querying model: {response.error}")
                else:
                    # Handle exceptions from asyncio.gather
                    print(f"Exception during batch query: {response}")
            
            return Result.ok(results)
        except Exception as e:
            return Result.err(e)
    
    async def get_available_models(self) -> Result[List[Dict[str, Any]]]:
        """Get a list of available LLM models."""
        try:
            # This is a simplified implementation
            # In a real implementation, you might query the providers' APIs
            
            available_models = [
                {
                    "name": "gpt-4-0125-preview",
                    "provider": LLMProvider.OPENAI,
                    "description": "GPT-3.5 Turbo by OpenAI",
                    "supported": "openai_api_key" in self.api_keys
                },
                {
                    "name": "gpt-4",
                    "provider": LLMProvider.OPENAI,
                    "description": "GPT-4 by OpenAI",
                    "supported": "openai_api_key" in self.api_keys
                },
                {
                    "name": "claude-instant-1",
                    "provider": LLMProvider.ANTHROPIC,
                    "description": "Claude Instant by Anthropic",
                    "supported": "anthropic_api_key" in self.api_keys
                },
                {
                    "name": "claude-2",
                    "provider": LLMProvider.ANTHROPIC,
                    "description": "Claude 2 by Anthropic",
                    "supported": "anthropic_api_key" in self.api_keys
                }
                # Add more models as needed
            ]
            
            return Result.ok(available_models)
        except Exception as e:
            return Result.err(e)
