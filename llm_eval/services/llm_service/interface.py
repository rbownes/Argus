"""
Interface for the LLM Service.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from llm_eval.core.models import Prompt, LLMResponse
from llm_eval.core.utils import Result


class LLMServiceInterface(ABC):
    """Interface for services that query LLMs."""
    
    @abstractmethod
    async def query_model(
        self,
        model_name: str,
        prompt: Prompt,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Result[LLMResponse]:
        """
        Query a single LLM model with a prompt.
        
        Args:
            model_name: Name of the model to query.
            prompt: The prompt to send to the model.
            parameters: Optional parameters for the query (temperature, etc.).
            
        Returns:
            Result containing the model's response.
        """
        pass
    
    @abstractmethod
    async def batch_query(
        self,
        model_names: List[str],
        prompts: List[Prompt],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Result[List[LLMResponse]]:
        """
        Query multiple LLM models with multiple prompts.
        
        Args:
            model_names: Names of the models to query.
            prompts: The prompts to send to the models.
            parameters: Optional parameters for the queries.
            
        Returns:
            Result containing a list of model responses.
        """
        pass
    
    @abstractmethod
    async def get_available_models(self) -> Result[List[Dict[str, Any]]]:
        """
        Get a list of available LLM models.
        
        Returns:
            Result containing information about available models.
        """
        pass
