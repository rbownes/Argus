"""
Interface for the Storage Service.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, TypeVar

from llm_eval.core.models import (
    Prompt, 
    LLMResponse, 
    EvaluationResult,
    BatchQueryRequest,
    BatchQueryResponse
)
from llm_eval.core.utils import Result

T = TypeVar('T', bound=Union[Prompt, LLMResponse, EvaluationResult])


class StorageServiceInterface(ABC):
    """Interface for services that handle data storage and retrieval."""
    
    # Structured Data Storage
    
    @abstractmethod
    async def store_prompt(self, prompt: Prompt) -> Result[Prompt]:
        """
        Store a prompt in the database.
        
        Args:
            prompt: The prompt to store.
            
        Returns:
            Result containing the stored prompt with its ID.
        """
        pass
    
    @abstractmethod
    async def store_response(self, response: LLMResponse) -> Result[LLMResponse]:
        """
        Store an LLM response in the database.
        
        Args:
            response: The response to store.
            
        Returns:
            Result containing the stored response with its ID.
        """
        pass
    
    @abstractmethod
    async def store_evaluation(self, evaluation: EvaluationResult) -> Result[EvaluationResult]:
        """
        Store an evaluation result in the database.
        
        Args:
            evaluation: The evaluation result to store.
            
        Returns:
            Result containing the stored evaluation with its ID.
        """
        pass
    
    @abstractmethod
    async def store_batch(
        self, 
        batch_request: BatchQueryRequest,
        batch_response: BatchQueryResponse
    ) -> Result[str]:
        """
        Store a batch query request and response.
        
        Args:
            batch_request: The batch query request.
            batch_response: The batch query response.
            
        Returns:
            Result containing the batch ID.
        """
        pass
    
    # Data Retrieval
    
    @abstractmethod
    async def get_prompt(self, prompt_id: str) -> Result[Prompt]:
        """
        Get a prompt by ID.
        
        Args:
            prompt_id: The ID of the prompt to retrieve.
            
        Returns:
            Result containing the prompt if found.
        """
        pass
    
    @abstractmethod
    async def get_response(self, response_id: str) -> Result[LLMResponse]:
        """
        Get an LLM response by ID.
        
        Args:
            response_id: The ID of the response to retrieve.
            
        Returns:
            Result containing the response if found.
        """
        pass
    
    @abstractmethod
    async def get_evaluation(self, evaluation_id: str) -> Result[EvaluationResult]:
        """
        Get an evaluation result by ID.
        
        Args:
            evaluation_id: The ID of the evaluation to retrieve.
            
        Returns:
            Result containing the evaluation if found.
        """
        pass
    
    @abstractmethod
    async def get_batch(self, batch_id: str) -> Result[BatchQueryResponse]:
        """
        Get a batch query response by ID.
        
        Args:
            batch_id: The ID of the batch to retrieve.
            
        Returns:
            Result containing the batch response if found.
        """
        pass
    
    # Querying
    
    @abstractmethod
    async def query_responses(
        self,
        model_name: Optional[str] = None,
        prompt_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Result[List[LLMResponse]]:
        """
        Query LLM responses with filters.
        
        Args:
            model_name: Filter by model name if provided.
            prompt_id: Filter by prompt ID if provided.
            start_time: Filter by start time if provided.
            end_time: Filter by end time if provided.
            limit: Maximum number of responses to return.
            offset: Number of responses to skip.
            
        Returns:
            Result containing a list of matching responses.
        """
        pass
    
    @abstractmethod
    async def query_evaluations(
        self,
        response_id: Optional[str] = None,
        evaluation_type: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Result[List[EvaluationResult]]:
        """
        Query evaluation results with filters.
        
        Args:
            response_id: Filter by response ID if provided.
            evaluation_type: Filter by evaluation type if provided.
            min_score: Filter by minimum score if provided.
            max_score: Filter by maximum score if provided.
            limit: Maximum number of evaluations to return.
            offset: Number of evaluations to skip.
            
        Returns:
            Result containing a list of matching evaluations.
        """
        pass
    
    # Vector Storage
    
    @abstractmethod
    async def store_embedding(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> Result[str]:
        """
        Store a text embedding in the vector database.
        
        Args:
            text: The original text.
            embedding: The vector embedding of the text.
            metadata: Additional metadata to store with the embedding.
            
        Returns:
            Result containing the embedding ID.
        """
        pass
    
    @abstractmethod
    async def query_embeddings(
        self,
        query_embedding: List[float],
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        """
        Query the vector database for similar embeddings.
        
        Args:
            query_embedding: The query vector embedding.
            filter_metadata: Filter by metadata if provided.
            limit: Maximum number of results to return.
            
        Returns:
            Result containing a list of similar embeddings with metadata.
        """
        pass
    
    @abstractmethod
    async def query_embeddings_by_text(
        self,
        query_text: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        """
        Query the vector database for similar embeddings using text.
        
        Args:
            query_text: The query text to embed and search with.
            filter_metadata: Filter by metadata if provided.
            limit: Maximum number of results to return.
            
        Returns:
            Result containing a list of similar embeddings with metadata.
        """
        pass
    
    # Comparison
    
    @abstractmethod
    async def compare_models(
        self,
        model_names: List[str],
        evaluation_type: Optional[str] = None,
        prompt_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Result[Dict[str, Any]]:
        """
        Compare the performance of multiple models.
        
        Args:
            model_names: The names of the models to compare.
            evaluation_type: Filter by evaluation type if provided.
            prompt_ids: Filter by prompt IDs if provided.
            start_time: Filter by start time if provided.
            end_time: Filter by end time if provided.
            
        Returns:
            Result containing comparison data.
        """
        pass
