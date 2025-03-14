"""
Client library for interacting with the LLM Evaluation API.
"""
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from uuid import UUID

import httpx

from llm_eval.core.models import (
    ModelConfig, ThemeCategory, EvaluationMetric, 
    BatchQueryRequest, BatchQueryResponse
)


class LLMEvalClient:
    """Client for interacting with the LLM Evaluation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
    
    async def create_evaluation(
        self,
        models: List[ModelConfig],
        themes: List[ThemeCategory],
        evaluator_ids: Optional[List[str]] = None,
        metrics: Optional[List[EvaluationMetric]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        Create a new evaluation run.
        
        Args:
            models: List of model configurations
            themes: List of themes to evaluate
            evaluator_ids: List of evaluator IDs to use
            metrics: List of metrics to evaluate
            metadata: Optional metadata for the run
            
        Returns:
            UUID of the created run
        """
        request = BatchQueryRequest(
            models=models,
            themes=themes,
            evaluator_ids=evaluator_ids or ["rule_based_evaluator"],
            metrics=metrics,
            metadata=metadata or {}
        )
        
        response = await self.client.post(
            "/api/v1/evaluations",
            json=request.model_dump()
        )
        response.raise_for_status()
        
        data = response.json()
        return UUID(data["run_id"])
    
    async def get_evaluation_status(
        self,
        run_id: UUID
    ) -> Dict[str, Any]:
        """
        Get the status of an evaluation run.
        
        Args:
            run_id: UUID of the run
            
        Returns:
            Status information
        """
        response = await self.client.get(f"/api/v1/evaluations/{run_id}")
        response.raise_for_status()
        
        return response.json()
    
    async def get_evaluation_results(
        self,
        run_id: UUID
    ) -> Dict[str, Any]:
        """
        Get the results of an evaluation run.
        
        Args:
            run_id: UUID of the run
            
        Returns:
            Evaluation results
        """
        response = await self.client.get(f"/api/v1/evaluations/{run_id}/results")
        response.raise_for_status()
        
        return response.json()
    
    async def get_model_performance(
        self,
        model_provider: str,
        model_id: str,
        start_time: Optional[Union[datetime, str]] = None,
        end_time: Optional[Union[datetime, str]] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a specific model.
        
        Args:
            model_provider: Model provider (e.g., "openai")
            model_id: Model ID (e.g., "gpt-4")
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            Performance metrics
        """
        # Convert datetime objects to ISO strings
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
            
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
            
        request_data = {
            "model_provider": model_provider,
            "model_id": model_id
        }
        
        if start_time:
            request_data["start_time"] = start_time
            
        if end_time:
            request_data["end_time"] = end_time
            
        response = await self.client.post(
            "/api/v1/performance",
            json=request_data
        )
        response.raise_for_status()
        
        return response.json()
    
    async def search_semantically_similar(
        self,
        query_text: str,
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar responses.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of similar responses with metadata
        """
        request_data = {
            "query_text": query_text,
            "n_results": n_results,
            "filter_metadata": filter_metadata
        }
        
        response = await self.client.post(
            "/api/v1/semantic_search",
            json=request_data
        )
        response.raise_for_status()
        
        return response.json()
    
    async def get_model_responses(
        self,
        model_provider: str,
        model_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get responses from a specific model.
        
        Args:
            model_provider: Model provider (e.g., "openai")
            model_id: Model ID (e.g., "gpt-4")
            limit: Maximum number of responses to return
            offset: Offset for pagination
            
        Returns:
            List of responses with metadata
        """
        response = await self.client.get(
            f"/api/v1/models/{model_provider}/{model_id}/responses",
            params={"limit": limit, "offset": offset}
        )
        response.raise_for_status()
        
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        await self.close()
