"""
In-memory implementation of the Storage Service.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
import statistics
from collections import defaultdict

from llm_eval.core.models import (
    Prompt, 
    LLMResponse, 
    EvaluationResult,
    BatchQueryRequest,
    BatchQueryResponse,
    EvaluationType
)
from llm_eval.core.utils import Result, generate_id
from .interface import StorageServiceInterface


class InMemoryStorageService(StorageServiceInterface):
    """
    In-memory implementation of the Storage Service.
    
    This implementation stores all data in memory and is intended
    for testing or small-scale deployments.
    """
    
    def __init__(self):
        """Initialize the service with empty data stores."""
        self._prompts: Dict[str, Prompt] = {}
        self._responses: Dict[str, LLMResponse] = {}
        self._evaluations: Dict[str, EvaluationResult] = {}
        self._batches: Dict[str, Tuple[BatchQueryRequest, BatchQueryResponse]] = {}
        self._embeddings: List[Dict[str, Any]] = []
    
    async def store_prompt(self, prompt: Prompt) -> Result[Prompt]:
        """Store a prompt in memory."""
        try:
            # Generate an ID if one isn't provided
            if not prompt.id:
                prompt.id = generate_id()
            
            # Store the prompt
            self._prompts[prompt.id] = prompt
            return Result.ok(prompt)
        except Exception as e:
            return Result.err(e)
    
    async def store_response(self, response: LLMResponse) -> Result[LLMResponse]:
        """Store an LLM response in memory."""
        try:
            # Generate an ID if one isn't provided
            if not response.id:
                response.id = generate_id()
            
            # Store the response
            self._responses[response.id] = response
            return Result.ok(response)
        except Exception as e:
            return Result.err(e)
    
    async def store_evaluation(self, evaluation: EvaluationResult) -> Result[EvaluationResult]:
        """Store an evaluation result in memory."""
        try:
            # Generate an ID if one isn't provided
            if not evaluation.id:
                evaluation.id = generate_id()
            
            # Store the evaluation
            self._evaluations[evaluation.id] = evaluation
            return Result.ok(evaluation)
        except Exception as e:
            return Result.err(e)
    
    async def store_batch(
        self, 
        batch_request: BatchQueryRequest,
        batch_response: BatchQueryResponse
    ) -> Result[str]:
        """Store a batch query request and response."""
        try:
            batch_id = batch_response.batch_id
            
            # Store the batch
            self._batches[batch_id] = (batch_request, batch_response)
            
            # Also store individual responses
            for response in batch_response.responses:
                await self.store_response(response)
            
            # Store evaluations if present
            if batch_response.evaluations:
                for evaluation in batch_response.evaluations:
                    await self.store_evaluation(evaluation)
            
            return Result.ok(batch_id)
        except Exception as e:
            return Result.err(e)
    
    async def get_prompt(self, prompt_id: str) -> Result[Prompt]:
        """Get a prompt by ID."""
        try:
            if prompt_id not in self._prompts:
                return Result.err(KeyError(f"Prompt with ID {prompt_id} not found"))
            return Result.ok(self._prompts[prompt_id])
        except Exception as e:
            return Result.err(e)
    
    async def get_response(self, response_id: str) -> Result[LLMResponse]:
        """Get an LLM response by ID."""
        try:
            if response_id not in self._responses:
                return Result.err(KeyError(f"Response with ID {response_id} not found"))
            return Result.ok(self._responses[response_id])
        except Exception as e:
            return Result.err(e)
    
    async def get_evaluation(self, evaluation_id: str) -> Result[EvaluationResult]:
        """Get an evaluation result by ID."""
        try:
            if evaluation_id not in self._evaluations:
                return Result.err(KeyError(f"Evaluation with ID {evaluation_id} not found"))
            return Result.ok(self._evaluations[evaluation_id])
        except Exception as e:
            return Result.err(e)
    
    async def get_batch(self, batch_id: str) -> Result[BatchQueryResponse]:
        """Get a batch query response by ID."""
        try:
            if batch_id not in self._batches:
                return Result.err(KeyError(f"Batch with ID {batch_id} not found"))
            return Result.ok(self._batches[batch_id][1])
        except Exception as e:
            return Result.err(e)
    
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
        try:
            responses = list(self._responses.values())
            
            # Apply filters
            if model_name:
                responses = [r for r in responses if r.model_name == model_name]
            
            if prompt_id:
                responses = [r for r in responses if r.prompt_id == prompt_id]
            
            if start_time:
                responses = [r for r in responses if r.created_at >= start_time]
            
            if end_time:
                responses = [r for r in responses if r.created_at <= end_time]
            
            # Sort by creation time (newest first)
            responses.sort(key=lambda r: r.created_at, reverse=True)
            
            # Apply pagination
            paginated = responses[offset:offset + limit]
            
            return Result.ok(paginated)
        except Exception as e:
            return Result.err(e)
    
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
        try:
            evaluations = list(self._evaluations.values())
            
            # Apply filters
            if response_id:
                evaluations = [e for e in evaluations if e.response_id == response_id]
            
            if evaluation_type:
                evaluations = [e for e in evaluations if e.evaluation_type == evaluation_type]
            
            if min_score is not None:
                evaluations = [e for e in evaluations if e.score >= min_score]
            
            if max_score is not None:
                evaluations = [e for e in evaluations if e.score <= max_score]
            
            # Sort by score (highest first)
            evaluations.sort(key=lambda e: e.score, reverse=True)
            
            # Apply pagination
            paginated = evaluations[offset:offset + limit]
            
            return Result.ok(paginated)
        except Exception as e:
            return Result.err(e)
    
    async def store_embedding(
        self,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> Result[str]:
        """Store a text embedding in memory."""
        try:
            embedding_id = generate_id()
            
            # Store the embedding
            embedding_data = {
                "id": embedding_id,
                "text": text,
                "embedding": embedding,
                "metadata": metadata,
                "created_at": datetime.now()
            }
            
            self._embeddings.append(embedding_data)
            
            return Result.ok(embedding_id)
        except Exception as e:
            return Result.err(e)
    
    def _vector_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: The first vector.
            vec2: The second vector.
            
        Returns:
            The cosine similarity (higher is more similar).
        """
        # Simple dot product for cosine similarity
        # In a real implementation, use numpy or other optimized libraries
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same dimension")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    async def query_embeddings(
        self,
        query_embedding: List[float],
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        """Query for similar embeddings."""
        try:
            results = []
            
            # Filter embeddings by metadata if provided
            embeddings = self._embeddings
            if filter_metadata:
                embeddings = []
                for emb in self._embeddings:
                    metadata = emb["metadata"]
                    match = True
                    for key, value in filter_metadata.items():
                        if key not in metadata or metadata[key] != value:
                            match = False
                            break
                    if match:
                        embeddings.append(emb)
            
            # Calculate similarity for each embedding
            for emb in embeddings:
                similarity = self._vector_similarity(query_embedding, emb["embedding"])
                results.append({
                    "id": emb["id"],
                    "text": emb["text"],
                    "metadata": emb["metadata"],
                    "similarity": similarity
                })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda r: r["similarity"], reverse=True)
            
            # Apply limit
            results = results[:limit]
            
            return Result.ok(results)
        except Exception as e:
            return Result.err(e)
    
    async def query_embeddings_by_text(
        self,
        query_text: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        """
        Query for similar embeddings using text.
        
        Note: In a real implementation, this would embed the query text.
        Here we use a simple keyword match as a placeholder.
        """
        try:
            results = []
            query_text = query_text.lower()
            
            # Filter embeddings by metadata if provided
            embeddings = self._embeddings
            if filter_metadata:
                embeddings = []
                for emb in self._embeddings:
                    metadata = emb["metadata"]
                    match = True
                    for key, value in filter_metadata.items():
                        if key not in metadata or metadata[key] != value:
                            match = False
                            break
                    if match:
                        embeddings.append(emb)
            
            # Calculate simple text similarity for each embedding
            for emb in embeddings:
                # Simple keyword-based similarity (placeholder)
                text = emb["text"].lower()
                keywords = set(query_text.split())
                text_keywords = set(text.split())
                common_keywords = keywords.intersection(text_keywords)
                
                if not keywords:
                    similarity = 0
                else:
                    similarity = len(common_keywords) / len(keywords)
                
                results.append({
                    "id": emb["id"],
                    "text": emb["text"],
                    "metadata": emb["metadata"],
                    "similarity": similarity
                })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda r: r["similarity"], reverse=True)
            
            # Apply limit
            results = results[:limit]
            
            return Result.ok(results)
        except Exception as e:
            return Result.err(e)
    
    async def compare_models(
        self,
        model_names: List[str],
        evaluation_type: Optional[str] = None,
        prompt_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Result[Dict[str, Any]]:
        """Compare the performance of multiple models."""
        try:
            comparison = {}
            
            # Get all relevant evaluations
            evaluations = list(self._evaluations.values())
            
            # Filter evaluations by type if provided
            if evaluation_type:
                evaluations = [e for e in evaluations if e.evaluation_type == evaluation_type]
            
            # Get responses for each evaluation
            evaluation_responses = {}
            for evaluation in evaluations:
                response_id = evaluation.response_id
                if response_id in self._responses:
                    evaluation_responses[evaluation.id] = self._responses[response_id]
            
            # Filter by prompt IDs if provided
            if prompt_ids:
                prompt_id_set = set(prompt_ids)
                filtered_evaluations = []
                for evaluation in evaluations:
                    response = evaluation_responses.get(evaluation.id)
                    if response and response.prompt_id in prompt_id_set:
                        filtered_evaluations.append(evaluation)
                evaluations = filtered_evaluations
            
            # Filter by time range if provided
            if start_time or end_time:
                filtered_evaluations = []
                for evaluation in evaluations:
                    response = evaluation_responses.get(evaluation.id)
                    if not response:
                        continue
                    
                    include = True
                    if start_time and response.created_at < start_time:
                        include = False
                    if end_time and response.created_at > end_time:
                        include = False
                    
                    if include:
                        filtered_evaluations.append(evaluation)
                
                evaluations = filtered_evaluations
            
            # Group evaluations by model
            model_evaluations = defaultdict(list)
            for evaluation in evaluations:
                response = evaluation_responses.get(evaluation.id)
                if response and response.model_name in model_names:
                    model_evaluations[response.model_name].append(evaluation)
            
            # Calculate statistics for each model
            for model_name in model_names:
                model_evals = model_evaluations.get(model_name, [])
                
                if not model_evals:
                    comparison[model_name] = {
                        "count": 0,
                        "average_score": None,
                        "median_score": None,
                        "min_score": None,
                        "max_score": None
                    }
                    continue
                
                scores = [e.score for e in model_evals]
                
                comparison[model_name] = {
                    "count": len(scores),
                    "average_score": statistics.mean(scores) if scores else None,
                    "median_score": statistics.median(scores) if scores else None,
                    "min_score": min(scores) if scores else None,
                    "max_score": max(scores) if scores else None
                }
                
                # Group by evaluation type if no specific type was requested
                if not evaluation_type:
                    by_type = defaultdict(list)
                    for eval_result in model_evals:
                        by_type[eval_result.evaluation_type].append(eval_result.score)
                    
                    type_stats = {}
                    for eval_type, type_scores in by_type.items():
                        type_stats[eval_type] = {
                            "count": len(type_scores),
                            "average_score": statistics.mean(type_scores),
                            "median_score": statistics.median(type_scores),
                            "min_score": min(type_scores),
                            "max_score": max(type_scores)
                        }
                    
                    comparison[model_name]["by_type"] = type_stats
            
            return Result.ok(comparison)
        except Exception as e:
            return Result.err(e)
