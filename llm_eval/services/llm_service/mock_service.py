"""
Mock LLM Service for testing.
"""
import asyncio
import random
import time
from typing import Dict, List, Any, Optional

from llm_eval.core.models import Prompt, LLMResponse, LLMProvider
from llm_eval.core.utils import Result, generate_id
from .interface import LLMServiceInterface


class MockLLMService(LLMServiceInterface):
    """
    Mock LLM Service for testing.
    
    This service generates deterministic responses based on the prompt
    and model name, making it suitable for testing.
    """
    
    def __init__(self, response_delay_ms: int = 500):
        """
        Initialize the mock service.
        
        Args:
            response_delay_ms: Simulated response delay in milliseconds.
        """
        self.response_delay_ms = response_delay_ms
        
        # Predefined model information
        self.models = [
            {
                "name": "mock-gpt-3.5",
                "provider": LLMProvider.OPENAI,
                "description": "Mock GPT-3.5 for testing",
                "supported": True
            },
            {
                "name": "mock-gpt-4",
                "provider": LLMProvider.OPENAI,
                "description": "Mock GPT-4 for testing",
                "supported": True
            },
            {
                "name": "mock-claude",
                "provider": LLMProvider.ANTHROPIC,
                "description": "Mock Claude for testing",
                "supported": True
            }
        ]
    
    def _get_provider_from_model(self, model_name: str) -> LLMProvider:
        """Get the provider enum from the model name."""
        model_name = model_name.lower()
        
        for model in self.models:
            if model["name"].lower() == model_name:
                return model["provider"]
        
        # Default for unknown models
        if "gpt" in model_name:
            return LLMProvider.OPENAI
        elif "claude" in model_name:
            return LLMProvider.ANTHROPIC
        else:
            return LLMProvider.OTHER
    
    def _generate_mock_response(self, model_name: str, prompt_text: str) -> str:
        """
        Generate a deterministic mock response.
        
        Args:
            model_name: The name of the model.
            prompt_text: The prompt text.
            
        Returns:
            A mock response.
        """
        # Simple deterministic response based on model and prompt
        if "mock-gpt-3.5" in model_name.lower():
            return f"This is a response from GPT-3.5 to: {prompt_text}"
        elif "mock-gpt-4" in model_name.lower():
            return f"This is a more sophisticated response from GPT-4 to: {prompt_text}"
        elif "mock-claude" in model_name.lower():
            return f"Claude here! Responding to: {prompt_text}"
        else:
            return f"Generic response from {model_name} to: {prompt_text}"
    
    async def query_model(
        self,
        model_name: str,
        prompt: Prompt,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Result[LLMResponse]:
        """Query a single LLM model with a prompt."""
        try:
            # Simulate API delay
            await asyncio.sleep(self.response_delay_ms / 1000)
            
            # Generate a deterministic response
            response_text = self._generate_mock_response(model_name, prompt.text)
            
            # Simulate tokens used based on output length (rough approximation)
            tokens_used = len(response_text.split()) + len(prompt.text.split())
            
            # Create a mock LLM response
            llm_response = LLMResponse(
                id=generate_id(),
                prompt_id=prompt.id or generate_id(),
                prompt_text=prompt.text,
                model_name=model_name,
                provider=self._get_provider_from_model(model_name),
                response_text=response_text,
                tokens_used=tokens_used,
                latency_ms=self.response_delay_ms + random.randint(-100, 100),
                metadata={
                    "finish_reason": "stop",
                    "mock": True,
                    **(parameters or {})
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
            responses = await asyncio.gather(*tasks)
            
            # Process responses
            for response in responses:
                if response.is_ok:
                    results.append(response.unwrap())
            
            return Result.ok(results)
        except Exception as e:
            return Result.err(e)
    
    async def get_available_models(self) -> Result[List[Dict[str, Any]]]:
        """Get a list of available LLM models."""
        try:
            return Result.ok(self.models)
        except Exception as e:
            return Result.err(e)
