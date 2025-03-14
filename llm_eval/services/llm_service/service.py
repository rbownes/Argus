"""
LLM query service implementation.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
from uuid import UUID

import litellm
from pydantic import BaseModel

from llm_eval.core.models import ModelConfig, QueryPrompt, ModelResponse


class LLMQueryService:
    """Service for querying LLM models."""
    
    def __init__(self):
        """Initialize the LLM query service."""
        self.litellm = litellm
    
    async def query_model(
        self, 
        model_config: ModelConfig, 
        prompt: QueryPrompt,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """
        Query a single LLM model with a prompt.
        
        Args:
            model_config: Configuration for the model to query
            prompt: The prompt to send to the model
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            ModelResponse object containing the response and metadata
        """
        start_time = time.time()
        
        # Set API key if provided
        if model_config.api_key:
            litellm.api_key = model_config.api_key
        
        # Prepare model string in LiteLLM format
        model_string = f"{model_config.provider}/{model_config.model_id}"
        
        # Query the model
        try:
            response = await litellm.acompletion(
                model=model_string,
                messages=[{"role": "user", "content": prompt.text}],
                temperature=temperature,
                max_tokens=max_tokens,
                **model_config.parameters
            )
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract response text and metadata
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else None
            
            # Create and return ModelResponse
            return ModelResponse(
                prompt_id=prompt.id,
                model_config=model_config,
                content=response_text,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                metadata={
                    "provider_metadata": response.model_dump() if hasattr(response, 'model_dump') else {}
                }
            )
            
        except Exception as e:
            # Handle errors
            latency_ms = int((time.time() - start_time) * 1000)
            return ModelResponse(
                prompt_id=prompt.id,
                model_config=model_config,
                content=f"Error: {str(e)}",
                latency_ms=latency_ms,
                metadata={"error": str(e)}
            )
    
    async def query_multiple_models(
        self, 
        model_configs: List[ModelConfig], 
        prompts: List[QueryPrompt],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> List[ModelResponse]:
        """
        Query multiple LLM models with multiple prompts.
        
        Args:
            model_configs: List of model configurations
            prompts: List of prompts to send to each model
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of ModelResponse objects
        """
        tasks = []
        for model_config in model_configs:
            for prompt in prompts:
                tasks.append(
                    self.query_model(
                        model_config=model_config,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                )
        
        return await asyncio.gather(*tasks)
