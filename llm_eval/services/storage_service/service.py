"""
Storage service implementation for PostgreSQL and ChromaDB.
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import UUID

import chromadb
import psycopg
from chromadb.utils import embedding_functions
from pydantic import BaseModel

from llm_eval.core.models import ModelResponse, EvaluationResult, EvaluationRun


class StorageService:
    """Service for storing and retrieving evaluation data."""
    
    def __init__(
        self,
        postgres_url: str,
        chroma_path: str = "./chroma_db",
        embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    ):
        """
        Initialize storage connections.
        
        Args:
            postgres_url: PostgreSQL connection URL
            chroma_path: Path for ChromaDB storage
            embedding_model_name: Name of the embedding model to use
        """
        # Initialize PostgreSQL connection
        self.pg_conn = psycopg.connect(postgres_url)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        # Create response collection if it doesn't exist
        self.response_collection = self.chroma_client.get_or_create_collection(
            name="model_responses",
            embedding_function=self.embedding_function
        )
    
    async def store_run(self, run: EvaluationRun) -> None:
        """Store an evaluation run in PostgreSQL."""
        with self.pg_conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO evaluation_runs 
                (id, models, themes, start_time, end_time, status, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(run.id),
                    json.dumps([m.model_dump() for m in run.models]),
                    json.dumps([t.value for t in run.themes]),
                    run.start_time,
                    run.end_time,
                    run.status,
                    json.dumps(run.metadata)
                )
            )
        self.pg_conn.commit()
    
    async def store_response(self, response: ModelResponse) -> str:
        """
        Store a model response in both PostgreSQL and ChromaDB.
        
        Returns:
            ChromaDB document ID
        """
        # Store in PostgreSQL
        with self.pg_conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO model_responses
                (id, prompt_id, model_provider, model_id, content, timestamp, latency_ms, tokens_used, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(response.id),
                    str(response.prompt_id),
                    response.model_config.provider.value,
                    response.model_config.model_id,
                    response.content,
                    response.timestamp,
                    response.latency_ms,
                    response.tokens_used,
                    json.dumps(response.metadata)
                )
            )
        self.pg_conn.commit()
        
        # Store in ChromaDB
        doc_id = str(response.id)
        self.response_collection.add(
            documents=[response.content],
            metadatas=[{
                "id": str(response.id),
                "prompt_id": str(response.prompt_id),
                "model_provider": response.model_config.provider.value,
                "model_id": response.model_config.model_id,
                "timestamp": response.timestamp.isoformat(),
                "latency_ms": response.latency_ms,
                "tokens_used": response.tokens_used
            }],
            ids=[doc_id]
        )
        
        return doc_id
    
    async def store_evaluation_result(self, result: EvaluationResult) -> None:
        """Store an evaluation result in PostgreSQL."""
        with self.pg_conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO evaluation_results
                (id, response_id, run_id, evaluator_id, scores, timestamp, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(result.id),
                    str(result.response_id),
                    str(result.run_id),
                    result.evaluator_id,
                    json.dumps([s.model_dump() for s in result.scores]),
                    result.timestamp,
                    json.dumps(result.metadata)
                )
            )
        self.pg_conn.commit()
    
    async def query_semantically_similar(
        self,
        query_text: str,
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for semantically similar responses.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of similar responses with metadata
        """
        results = self.response_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata
        )
        
        return [
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]
    
    async def get_model_performance(
        self,
        model_provider: str,
        model_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a specific model.
        
        Args:
            model_provider: Model provider (e.g., "openai")
            model_id: Model ID (e.g., "gpt-4")
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            Dictionary of performance metrics
        """
        time_filter = ""
        params = [model_provider, model_id]
        
        if start_time:
            time_filter += " AND r.timestamp >= %s"
            params.append(start_time)
        
        if end_time:
            time_filter += " AND r.timestamp <= %s"
            params.append(end_time)
        
        with self.pg_conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT 
                    e.evaluator_id,
                    m.metric,
                    AVG(m.score) as avg_score,
                    COUNT(*) as count
                FROM 
                    model_responses r
                JOIN 
                    evaluation_results e ON r.id = e.response_id
                JOIN 
                    (
                        SELECT 
                            result_id,
                            json_array_elements(scores::json)->>'metric' as metric,
                            (json_array_elements(scores::json)->>'score')::float as score
                        FROM 
                            evaluation_results
                    ) m ON e.id = m.result_id
                WHERE 
                    r.model_provider = %s
                    AND r.model_id = %s
                    {time_filter}
                GROUP BY 
                    e.evaluator_id, m.metric
                """,
                params
            )
            
            results = cursor.fetchall()
        
        # Organize results by evaluator and metric
        performance = {}
        for row in results:
            evaluator_id, metric, avg_score, count = row
            
            if evaluator_id not in performance:
                performance[evaluator_id] = {}
                
            performance[evaluator_id][metric] = {
                "avg_score": float(avg_score),
                "count": int(count)
            }
        
        return performance
    
    async def get_responses_by_model(
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
        with self.pg_conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    id, prompt_id, content, timestamp, latency_ms, tokens_used, metadata
                FROM 
                    model_responses
                WHERE 
                    model_provider = %s
                    AND model_id = %s
                ORDER BY 
                    timestamp DESC
                LIMIT %s OFFSET %s
                """,
                (model_provider, model_id, limit, offset)
            )
            
            results = cursor.fetchall()
        
        responses = []
        for row in results:
            id_str, prompt_id, content, timestamp, latency_ms, tokens_used, metadata = row
            
            responses.append({
                "id": id_str,
                "prompt_id": prompt_id,
                "content": content,
                "timestamp": timestamp.isoformat() if timestamp else None,
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
                "metadata": json.loads(metadata) if metadata else {}
            })
        
        return responses
    
    async def get_evaluation_results_by_run(
        self,
        run_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation results for a specific run.
        
        Args:
            run_id: UUID of the evaluation run
            
        Returns:
            List of evaluation results with scores
        """
        with self.pg_conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    id, response_id, evaluator_id, scores, timestamp, metadata
                FROM 
                    evaluation_results
                WHERE 
                    run_id = %s
                ORDER BY 
                    timestamp
                """,
                (str(run_id),)
            )
            
            results = cursor.fetchall()
        
        evaluation_results = []
        for row in results:
            id_str, response_id, evaluator_id, scores, timestamp, metadata = row
            
            evaluation_results.append({
                "id": id_str,
                "response_id": response_id,
                "evaluator_id": evaluator_id,
                "scores": json.loads(scores) if scores else [],
                "timestamp": timestamp.isoformat() if timestamp else None,
                "metadata": json.loads(metadata) if metadata else {}
            })
        
        return evaluation_results
