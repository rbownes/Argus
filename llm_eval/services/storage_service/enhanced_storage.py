"""
Enhanced storage service with automatic embedding.
"""
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np

from llm_eval.core.models import (
    Prompt, 
    LLMResponse, 
    EvaluationResult,
    BatchQueryRequest,
    BatchQueryResponse
)
from llm_eval.core.utils import Result, generate_id
from llm_eval.services.storage_service.interface import StorageServiceInterface
from llm_eval.services.storage_service.postgres_storage import PostgresStorage
from llm_eval.services.storage_service.chroma_storage import ChromaStorage
from llm_eval.services.embedding_service.text_embedding import TextEmbeddingService


class EmbeddingCache:
    """Cache for embeddings to avoid re-embedding similar texts."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize the embedding cache."""
        self.cache = {}
        self.text_to_id = {}
        self.max_size = max_size
        self.access_count = {}
    
    def add(self, id: str, text: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Add an embedding to the cache."""
        # Remove oldest entry if at capacity
        if len(self.cache) >= self.max_size:
            oldest_id = min(self.access_count.items(), key=lambda x: x[1])[0]
            self._remove(oldest_id)
        
        # Add to cache
        self.cache[id] = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        }
        
        # Update text to ID mapping
        self.text_to_id[text] = id
        
        # Initialize access count
        self.access_count[id] = 1
    
    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """Get an embedding from the cache by ID."""
        if id in self.cache:
            self.access_count[id] += 1
            return self.cache[id]
        return None
    
    def find_by_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Find an embedding by text."""
        if text in self.text_to_id:
            id = self.text_to_id[text]
            return self.get(id)
        return None
    
    def _remove(self, id: str) -> None:
        """Remove an entry from the cache."""
        if id in self.cache:
            text = self.cache[id]["text"]
            if self.text_to_id.get(text) == id:
                del self.text_to_id[text]
            del self.cache[id]
            del self.access_count[id]


class EnhancedStorageService(StorageServiceInterface):
    """Enhanced storage service with automatic embedding."""
    
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
        chroma_port: Optional[int] = None,
        # Embedding settings
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None
    ):
        """Initialize the enhanced storage service."""
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
        
        # Initialize embedding service
        self.embedding_service = TextEmbeddingService(
            model_name=embedding_model,
            device=device
        )
        
        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache(max_size=10000)
    
    async def initialize(self) -> None:
        """Initialize the storage service."""
        await self.pg_storage.initialize()
    
    async def close(self) -> None:
        """Close the storage service."""
        await self.pg_storage.close()
    
    async def _embed_and_store(
        self, 
        id: str, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> Result[str]:
        """Embed text and store it in the vector database."""
        try:
            # Check cache for exact text match
            cached_embedding = self.embedding_cache.find_by_text(text)
            if cached_embedding:
                embedding = cached_embedding["embedding"]
                metadata.update({
                    "embedding_model": self.embedding_service.model_name,
                    "embedding_latency_ms": 0,  # Cached
                    "cached": True
                })
            else:
                # Generate new embedding
                embedding, latency = self.embedding_service.embed_text(text)
                metadata.update({
                    "embedding_model": self.embedding_service.model_name,
                    "embedding_latency_ms": latency,
                    "cached": False
                })
                
                # Add to cache
                self.embedding_cache.add(
                    id=id,
                    text=text,
                    embedding=embedding,
                    metadata=metadata
                )
            
            # Store in ChromaDB
            result = await self.chroma_storage.store_embedding(
                id=id,
                text=text,
                embedding=embedding,
                metadata=metadata
            )
            
            return result
        except Exception as e:
            return Result.err(e)
    
    # Override storage methods to add embedding
    
    async def store_prompt(self, prompt: Prompt) -> Result[Prompt]:
        """Store a prompt and its embedding."""
        # Store in PostgreSQL
        result = await self.pg_storage.store_prompt(prompt)
        
        if result.is_err:
            return result
        
        # Embed and store in ChromaDB
        stored_prompt = result.unwrap()
        embedding_result = await self._embed_and_store(
            id=f"prompt:{stored_prompt.id}",
            text=stored_prompt.text,
            metadata={
                "type": "prompt",
                "id": stored_prompt.id,
                "category": stored_prompt.category,
                "tags": stored_prompt.tags,
                "created_at": stored_prompt.created_at.isoformat(),
            }
        )
        
        # Even if embedding fails, return the stored prompt
        return result
    
    async def store_response(self, response: LLMResponse) -> Result[LLMResponse]:
        """Store a response and its embedding."""
        # Store in PostgreSQL
        result = await self.pg_storage.store_response(response)
        
        if result.is_err:
            return result
        
        # Embed and store in ChromaDB
        stored_response = result.unwrap()
        embedding_result = await self._embed_and_store(
            id=f"response:{stored_response.id}",
            text=stored_response.response_text,
            metadata={
                "type": "response",
                "id": stored_response.id,
                "prompt_id": stored_response.prompt_id,
                "model_name": stored_response.model_name,
                "provider": stored_response.provider,
                "tokens_used": stored_response.tokens_used,
                "latency_ms": stored_response.latency_ms,
                "created_at": stored_response.created_at.isoformat(),
            }
        )
        
        # Even if embedding fails, return the stored response
        return result
    
    # Forward other methods to PostgreSQL storage
    
    async def store_evaluation(self, evaluation: EvaluationResult) -> Result[EvaluationResult]:
        return await self.pg_storage.store_evaluation(evaluation)
    
    async def store_batch(self, batch_request: BatchQueryRequest, batch_response: BatchQueryResponse) -> Result[str]:
        return await self.pg_storage.store_batch(batch_request, batch_response)
    
    async def get_prompt(self, prompt_id: str) -> Result[Prompt]:
        return await self.pg_storage.get_prompt(prompt_id)
    
    async def get_response(self, response_id: str) -> Result[LLMResponse]:
        return await self.pg_storage.get_response(response_id)
    
    async def get_evaluation(self, evaluation_id: str) -> Result[EvaluationResult]:
        return await self.pg_storage.get_evaluation(evaluation_id)
    
    async def get_batch(self, batch_id: str) -> Result[BatchQueryResponse]:
        return await self.pg_storage.get_batch(batch_id)
    
    async def query_responses(self, *args, **kwargs) -> Result[List[LLMResponse]]:
        return await self.pg_storage.query_responses(*args, **kwargs)
    
    async def query_evaluations(self, *args, **kwargs) -> Result[List[EvaluationResult]]:
        return await self.pg_storage.query_evaluations(*args, **kwargs)
        
    # Implement vector search methods
    
    async def store_embedding(self, text: str, embedding: List[float], metadata: Dict[str, Any]) -> Result[str]:
        """Store a text embedding in the vector database."""
        embedding_id = generate_id()
        return await self._embed_and_store(embedding_id, text, metadata)
    
    async def query_embeddings(
        self,
        query_embedding: List[float],
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        """Query the vector database with a precomputed embedding."""
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
        """Query the vector database using natural language text."""
        try:
            # Generate embedding for the query text
            query_embedding, _ = self.embedding_service.embed_text(query_text)
            
            # Use the embedding to search
            return await self.query_embeddings(
                query_embedding=query_embedding,
                filter_metadata=filter_metadata,
                limit=limit
            )
        except Exception as e:
            return Result.err(e)
    
    async def compare_responses_semantically(
        self,
        prompt_id: str,
        model_names: List[str],
        limit: int = 10
    ) -> Result[Dict[str, Any]]:
        """
        Compare responses from different models semantically.
        
        Args:
            prompt_id: ID of the prompt to compare responses for.
            model_names: Names of the models to compare.
            limit: Maximum number of responses to consider per model.
            
        Returns:
            Result containing semantic similarity scores between models.
        """
        try:
            # Get responses for each model
            model_responses = {}
            for model_name in model_names:
                responses_result = await self.query_responses(
                    model_name=model_name,
                    prompt_id=prompt_id,
                    limit=limit
                )
                
                if responses_result.is_err:
                    return responses_result
                
                responses = responses_result.unwrap()
                if responses:
                    model_responses[model_name] = responses
            
            # Check if we have enough models with responses
            if len(model_responses) < 2:
                return Result.err(ValueError(
                    f"Need at least 2 models with responses for prompt {prompt_id}"
                ))
            
            # Calculate pairwise similarities
            similarity_scores = {}
            for i, model1 in enumerate(model_responses.keys()):
                for j, model2 in enumerate(model_responses.keys()):
                    if i >= j:  # Only calculate upper triangle
                        continue
                    
                    # Get responses
                    responses1 = model_responses[model1]
                    responses2 = model_responses[model2]
                    
                    # Encode all responses
                    texts1 = [r.response_text for r in responses1]
                    texts2 = [r.response_text for r in responses2]
                    
                    # Calculate pairwise similarities
                    embeddings1, _ = self.embedding_service.embed_batch(texts1)
                    embeddings2, _ = self.embedding_service.embed_batch(texts2)
                    
                    # Calculate cosine similarities
                    similarities = []
                    for emb1 in embeddings1:
                        emb1_np = np.array(emb1)
                        for emb2 in embeddings2:
                            emb2_np = np.array(emb2)
                            # Normalized dot product for cosine similarity
                            sim = np.dot(emb1_np, emb2_np) / (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
                            similarities.append(float(sim))
                    
                    # Store average similarity
                    pair_key = f"{model1}:{model2}"
                    similarity_scores[pair_key] = {
                        "average": sum(similarities) / len(similarities) if similarities else 0,
                        "min": min(similarities) if similarities else 0,
                        "max": max(similarities) if similarities else 0,
                        "count": len(similarities)
                    }
            
            # Create comparison result
            comparison = {
                "prompt_id": prompt_id,
                "model_counts": {model: len(responses) for model, responses in model_responses.items()},
                "similarity_scores": similarity_scores
            }
            
            return Result.ok(comparison)
        except Exception as e:
            return Result.err(e)
            
    async def detect_semantic_drift(
        self,
        model_name: str,
        reference_time: datetime,
        current_time: datetime,
        top_n: int = 20,
        similarity_threshold: float = 0.8
    ) -> Result[Dict[str, Any]]:
        """
        Detect semantic drift in model responses over time.
        
        Args:
            model_name: Name of the model to check for drift.
            reference_time: Reference point in time.
            current_time: Current point in time.
            top_n: Number of top prompts to analyze.
            similarity_threshold: Threshold for considering responses semantically different.
            
        Returns:
            Result containing drift analysis.
        """
        try:
            # Get reference responses
            reference_responses_result = await self.query_responses(
                model_name=model_name,
                start_time=reference_time,
                end_time=reference_time + (current_time - reference_time) / 2,
                limit=1000
            )
            
            if reference_responses_result.is_err:
                return reference_responses_result
            
            reference_responses = reference_responses_result.unwrap()
            
            # Get current responses
            current_responses_result = await self.query_responses(
                model_name=model_name,
                start_time=reference_time + (current_time - reference_time) / 2,
                end_time=current_time,
                limit=1000
            )
            
            if current_responses_result.is_err:
                return current_responses_result
            
            current_responses = current_responses_result.unwrap()
            
            # Find common prompts
            reference_prompt_ids = {r.prompt_id for r in reference_responses}
            current_prompt_ids = {r.prompt_id for r in current_responses}
            common_prompt_ids = reference_prompt_ids.intersection(current_prompt_ids)
            
            # Group responses by prompt ID
            reference_by_prompt = {}
            for response in reference_responses:
                if response.prompt_id in common_prompt_ids:
                    if response.prompt_id not in reference_by_prompt:
                        reference_by_prompt[response.prompt_id] = []
                    reference_by_prompt[response.prompt_id].append(response)
            
            current_by_prompt = {}
            for response in current_responses:
                if response.prompt_id in common_prompt_ids:
                    if response.prompt_id not in current_by_prompt:
                        current_by_prompt[response.prompt_id] = []
                    current_by_prompt[response.prompt_id].append(response)
            
            # Compare responses for each prompt
            drift_analysis = {}
            
            for prompt_id in common_prompt_ids:
                if prompt_id in reference_by_prompt and prompt_id in current_by_prompt:
                    # Get responses
                    ref_responses = reference_by_prompt[prompt_id]
                    cur_responses = current_by_prompt[prompt_id]
                    
                    # Skip if not enough responses
                    if not ref_responses or not cur_responses:
                        continue
                    
                    # Embed responses
                    ref_texts = [r.response_text for r in ref_responses]
                    cur_texts = [r.response_text for r in cur_responses]
                    
                    ref_embeddings, _ = self.embedding_service.embed_batch(ref_texts)
                    cur_embeddings, _ = self.embedding_service.embed_batch(cur_texts)
                    
                    # Calculate average reference embedding
                    avg_ref_embedding = np.mean(np.array(ref_embeddings), axis=0)
                    
                    # Calculate average current embedding
                    avg_cur_embedding = np.mean(np.array(cur_embeddings), axis=0)
                    
                    # Calculate similarity
                    similarity = np.dot(avg_ref_embedding, avg_cur_embedding) / (
                        np.linalg.norm(avg_ref_embedding) * np.linalg.norm(avg_cur_embedding)
                    )
                    
                    # Check if there's significant drift
                    has_drift = similarity < similarity_threshold
                    
                    # Get prompt text
                    prompt_result = await self.get_prompt(prompt_id)
                    prompt_text = prompt_result.unwrap().text if prompt_result.is_ok else "Unknown"
                    
                    # Store analysis
                    drift_analysis[prompt_id] = {
                        "prompt_text": prompt_text,
                        "similarity": float(similarity),
                        "has_drift": has_drift,
                        "reference_count": len(ref_responses),
                        "current_count": len(cur_responses)
                    }
            
            # Sort by similarity (ascending) to find most drifted first
            sorted_drift = sorted(
                drift_analysis.items(),
                key=lambda x: x[1]["similarity"]
            )
            
            # Take top N drifted prompts
            top_drifted = {k: v for k, v in sorted_drift[:top_n]}
            
            # Calculate overall drift metrics
            all_similarities = [v["similarity"] for v in drift_analysis.values()]
            avg_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 1.0
            drift_ratio = sum(1 for v in drift_analysis.values() if v["has_drift"]) / len(drift_analysis) if drift_analysis else 0
            
            result = {
                "model_name": model_name,
                "reference_time": reference_time.isoformat(),
                "current_time": current_time.isoformat(),
                "average_similarity": avg_similarity,
                "drift_ratio": drift_ratio,
                "similarity_threshold": similarity_threshold,
                "total_prompts_analyzed": len(drift_analysis),
                "top_drifted_prompts": top_drifted
            }
            
            return Result.ok(result)
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
        """
        Compare performance between different models based on evaluations.
        
        Args:
            model_names: List of model names to compare.
            evaluation_type: Filter by evaluation type if provided.
            prompt_ids: Filter by prompt IDs if provided.
            start_time: Filter by start time if provided.
            end_time: Filter by end time if provided.
            
        Returns:
            Result containing comparison metrics between models.
        """
        try:
            # Get evaluations for each model
            model_evaluations = {}
            for model_name in model_names:
                # Query responses for this model
                responses_result = await self.query_responses(
                    model_name=model_name,
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000  # Reasonable limit for comparison
                )
                
                if responses_result.is_err:
                    return responses_result
                
                responses = responses_result.unwrap()
                response_ids = [r.id for r in responses]
                
                # Query evaluations for these responses
                evaluations_result = await self.query_evaluations(
                    evaluation_type=evaluation_type,
                    response_ids=response_ids
                )
                
                if evaluations_result.is_err:
                    return evaluations_result
                
                model_evaluations[model_name] = evaluations_result.unwrap()
            
            # Calculate comparison metrics
            comparison = {
                "model_metrics": {},
                "evaluation_counts": {},
                "average_scores": {},
                "score_distributions": {}
            }
            
            for model_name, evaluations in model_evaluations.items():
                if not evaluations:
                    comparison["model_metrics"][model_name] = {
                        "total_evaluations": 0,
                        "average_score": None
                    }
                    continue
                
                scores = [e.score for e in evaluations if e.score is not None]
                
                comparison["model_metrics"][model_name] = {
                    "total_evaluations": len(evaluations),
                    "average_score": sum(scores) / len(scores) if scores else None,
                    "min_score": min(scores) if scores else None,
                    "max_score": max(scores) if scores else None
                }
                
                # Count evaluations by type
                eval_types = {}
                for eval in evaluations:
                    eval_types[eval.evaluation_type] = eval_types.get(eval.evaluation_type, 0) + 1
                comparison["evaluation_counts"][model_name] = eval_types
                
                # Calculate score distribution
                if scores:
                    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    hist = np.histogram(scores, bins=bins)[0].tolist()
                    comparison["score_distributions"][model_name] = {
                        "bins": bins,
                        "counts": hist
                    }
            
            return Result.ok(comparison)
            
        except Exception as e:
            return Result.err(str(e))