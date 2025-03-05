"""
Combined storage service implementation.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from llm_eval.core.models import (
    Prompt, 
    LLMResponse, 
    EvaluationResult,
    BatchQueryRequest,
    BatchQueryResponse
)
from llm_eval.core.utils import Result, generate_id
from .interface import StorageServiceInterface
from .postgres_storage import PostgresStorage
from .chroma_storage import ChromaStorage


class StorageService(StorageServiceInterface):
    """
    Storage service implementation combining PostgreSQL and ChromaDB.
    
    This service handles both structured data and vector embeddings.
    """
    
    def __init__(
        self,
        # PostgreSQL settings
        pg_host: str = "localhost",
        pg_port: int = 5432,
        pg_user: str = "postgres",
        pg_password: str = "postgres",
        pg_database: str = "llm_eval",
        # ChromaDB settings
        collection_name: str = "llm_eval_embeddings",
        persist_directory: Optional[str] = None,
        chroma_host: Optional[str] = None,
        chroma_port: Optional[int] = None
    ):
        """
        Initialize the storage service.
        
        Args:
            pg_host: PostgreSQL host.
            pg_port: PostgreSQL port.
            pg_user: PostgreSQL user.
            pg_password: PostgreSQL password.
            pg_database: PostgreSQL database name.
            collection_name: ChromaDB collection name.
            persist_directory: Directory to persist ChromaDB data (for local mode).
            chroma_host: ChromaDB server host (for client mode).
            chroma_port: ChromaDB server port (for client mode).
        """
        # Initialize PostgreSQL storage
        self.pg_storage = PostgresStorage(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            database=pg_database
        )
        
        # Initialize ChromaDB storage
        self.chroma_storage = ChromaStorage(
            collection_name=collection_name,
            persist_directory=persist_directory,
            host=chroma_host,
            port=chroma_port
        )
    
    async def initialize(self) -> None:
        """Initialize the storage service."""
        await self.pg_storage.initialize()
    
    async def close(self) -> None:
        """Close the storage service."""
        await self.pg_storage.close()
    
    # Structured Data Storage
    
    async def store_prompt(self, prompt: Prompt) -> Result[Prompt]:
        """Store a prompt in the database."""
        return await self.pg_storage.store_prompt(prompt)
    
    async def store_response(self, response: LLMResponse) -> Result[LLMResponse]:
        """Store an LLM response in the database."""
        return await self.pg_storage.store_response(response)
    
    async def store_evaluation(self, evaluation: EvaluationResult) -> Result[EvaluationResult]:
        """Store an evaluation result in the database."""
        return await self.pg_storage.store_evaluation(evaluation)
    
    async def store_batch(
        self, 
        batch_request: BatchQueryRequest,
        batch_response: BatchQueryResponse
    ) -> Result[str]:
        """Store a batch query request and response."""
        return await self.pg_storage.store_batch(batch_request, batch_response)
    
    # Data Retrieval
    
    async def get_prompt(self, prompt_id: str) -> Result[Prompt]:
        """Get a prompt by ID."""
        return await self.pg_storage.get_prompt(prompt_id)
    
    async def get_response(self, response_id: str) -> Result[LLMResponse]:
        """Get an LLM response by ID."""
        return await self.pg_storage.get_response(response_id)
    
    async def get_evaluation(self, evaluation_id: str) -> Result[EvaluationResult]:
        """Get an evaluation result by ID."""
        return await self.pg_storage.get_evaluation(evaluation_id)
    
    async def get_batch(self, batch_id: str) -> Result[BatchQueryResponse]:
        """Get a batch query response by ID."""
        return await self.pg_storage.get_batch(batch_id)
    
    # Querying
    
    async def query_responses(
        self,
        model_name: Optional[str] = None,
        prompt_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Result[List[LLMResponse]]:
        """Query LLM responses with filters."""
        return await self.pg_storage.query_responses(
            model_name=model_name,
            prompt_id=prompt_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset
        )
    
    async def query_evaluations(
        self,
        response_id: Optional[str] = None,
        evaluation_type: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Result[List[EvaluationResult]]:
        """Query evaluation results with filters."""
        return await self.pg_storage.query_evaluations(
            response_id=response_id,
            evaluation_type=evaluation_type,
            min_score=min_score,
            max_score=max_score,
            limit=limit,
            offset=offset
        )
    
    # Vector Storage
    
    async def store_embedding(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> Result[str]:
        """Store a text embedding in the vector database."""
        # Generate an ID for the embedding
        embedding_id = generate_id()
        
        # Store the embedding
        return await self.chroma_storage.store_embedding(
            id=embedding_id,
            text=text,
            embedding=embedding,
            metadata=metadata
        )
    
    async def query_embeddings(
        self,
        query_embedding: List[float],
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        """Query the vector database for similar embeddings."""
        return await self.chroma_storage.query_embeddings(
            query_embedding=query_embedding,
            filter_metadata=filter_metadata,
            limit=limit
        )
    
    async def query_embeddings_by_text(
        self,
        query_text: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        """
        Query the vector database for similar embeddings using text.
        
        Note: In a real implementation, this would embed the query text
        using a text embedding model. This is a simplified version.
        """
        # In a real implementation, embed the query text here
        # For simplicity, we'll return an empty result
        return Result.ok([])
    
    # Comparison
    
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
        
        This is a simplified implementation. In a real system,
        you'd implement more sophisticated comparison logic.
        """
        try:
            comparison = {}
            
            # For each model, get all relevant evaluations
            for model_name in model_names:
                # Get responses for this model
                responses_result = await self.query_responses(
                    model_name=model_name,
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000  # Increased limit for analysis
                )
                
                if responses_result.is_err:
                    return responses_result
                
                responses = responses_result.unwrap()
                
                # Filter by prompt IDs if provided
                if prompt_ids:
                    prompt_id_set = set(prompt_ids)
                    responses = [r for r in responses if r.prompt_id in prompt_id_set]
                
                if not responses:
                    comparison[model_name] = {
                        "count": 0,
                        "evaluations": {}
                    }
                    continue
                
                # Get evaluations for these responses
                response_ids = [r.id for r in responses]
                all_evaluations = []
                
                for response_id in response_ids:
                    evals_result = await self.query_evaluations(
                        response_id=response_id,
                        evaluation_type=evaluation_type
                    )
                    
                    if evals_result.is_ok:
                        all_evaluations.extend(evals_result.unwrap())
                
                # Group evaluations by type
                eval_by_type = {}
                for eval_result in all_evaluations:
                    eval_type = eval_result.evaluation_type
                    if eval_type not in eval_by_type:
                        eval_by_type[eval_type] = []
                    
                    eval_by_type[eval_type].append(eval_result.score)
                
                # Calculate statistics for each evaluation type
                eval_stats = {}
                for eval_type, scores in eval_by_type.items():
                    if not scores:
                        continue
                    
                    eval_stats[eval_type] = {
                        "count": len(scores),
                        "average": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores)
                    }
                
                comparison[model_name] = {
                    "count": len(responses),
                    "evaluations": eval_stats
                }
            
            return Result.ok(comparison)
        except Exception as e:
            return Result.err(e)
