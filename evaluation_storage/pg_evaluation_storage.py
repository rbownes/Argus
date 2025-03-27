"""
Evaluation storage implementation using PostgreSQL with pgvector extension.
"""
import uuid
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from shared.db import Database

class PgEvaluationStorage:
    """Storage for evaluation metrics using PostgreSQL with pgvector extension."""
    
    def __init__(self, db_url: str):
        """
        Initialize PostgreSQL connection and embedding model.
        
        Args:
            db_url: Database connection URL
        """
        self.logger = logging.getLogger("pg_evaluation_storage")
        self.db = Database(db_url)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
            self.logger.info(f"Initialized embedding model with dimension {self.embedding_dimension}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables and extensions if they don't exist."""
        try:
            with self.db.engine.connect() as connection:
                # Enable pgvector extension
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                
                # Create evaluation_metrics table with vector support
                connection.execute(text(f"""
                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    id UUID PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    embedding VECTOR({self.embedding_dimension}),
                    metric_type TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """))
                
                # Create index for similarity search
                connection.execute(text(f"""
                CREATE INDEX IF NOT EXISTS evaluation_metrics_embedding_idx
                ON evaluation_metrics
                USING ivfflat (embedding vector_cosine_ops);
                """))
                
                connection.commit()
                
                self.logger.info("Evaluation metrics tables and extensions initialized")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {str(e)}")
            raise

    def store_evaluation_metric(self, prompt: str, metric_type: str, metadata: Optional[Dict] = None) -> str:
        """
        Store an evaluation metric with its type and optional metadata.
        
        Args:
            prompt: The evaluation prompt text
            metric_type: Type or category of the evaluation metric
            metadata: Additional metadata for the metric
            
        Returns:
            The metric ID (UUID)
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(prompt).tolist()
            
            # Generate UUID
            metric_id = str(uuid.uuid4())
            
            # Prepare metadata
            metric_metadata = {
                "metric_type": metric_type,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            # Store in PostgreSQL
            with self.db.engine.connect() as connection:
                # Ensure metadata is properly serialized as a JSON string
                serialized_metadata = json.dumps(metric_metadata)
                
                connection.execute(
                    text("""
                    INSERT INTO evaluation_metrics (id, prompt, embedding, metric_type, metadata, created_at)
                    VALUES (:id, :prompt, :embedding, :metric_type, :metadata, NOW())
                    """),
                    {
                        "id": metric_id,
                        "prompt": prompt,
                        "embedding": embedding,
                        "metric_type": metric_type,
                        "metadata": serialized_metadata
                    }
                )
                connection.commit()
            
            self.logger.info(f"Stored evaluation metric with ID {metric_id} and type '{metric_type}'")
            return metric_id
        except Exception as e:
            self.logger.error(f"Failed to store evaluation metric: {str(e)}")
            raise

    def get_metrics_by_type(self, metric_type: str, limit: int = 10, skip: int = 0) -> List[Dict]:
        """
        Retrieve evaluation metrics by type with pagination.
        
        Args:
            metric_type: Type to filter by
            limit: Maximum number of results to return
            skip: Number of results to skip for pagination
            
        Returns:
            List of metrics with their metadata
        """
        try:
            with self.db.engine.connect() as connection:
                result = connection.execute(
                    text("""
                    SELECT id, prompt, metadata, created_at
                    FROM evaluation_metrics
                    WHERE metric_type = :metric_type
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :skip
                    """),
                    {
                        "metric_type": metric_type,
                        "limit": limit,
                        "skip": skip
                    }
                )
                
                metrics = []
                for row in result:
                    # Ensure metadata is properly deserialized from JSON
                    metadata = row.metadata
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    
                    metrics.append({
                        "id": str(row.id),
                        "prompt": row.prompt,
                        "metadata": metadata,
                    })
                
                self.logger.info(f"Retrieved {len(metrics)} metrics with type '{metric_type}'")
                return metrics
        except Exception as e:
            self.logger.error(f"Failed to get metrics by type: {str(e)}")
            raise

    def count_metrics_by_type(self, metric_type: str) -> int:
        """
        Count the number of metrics for a type.
        
        Args:
            metric_type: Type to count metrics for
            
        Returns:
            Number of metrics for the type
        """
        try:
            with self.db.engine.connect() as connection:
                result = connection.execute(
                    text("""
                    SELECT COUNT(*) as count
                    FROM evaluation_metrics
                    WHERE metric_type = :metric_type
                    """),
                    {"metric_type": metric_type}
                )
                
                count = result.fetchone().count
                self.logger.info(f"Counted {count} metrics with type '{metric_type}'")
                return count
        except Exception as e:
            self.logger.error(f"Failed to count metrics by type: {str(e)}")
            raise

    def search_similar_metrics(self, prompt: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar evaluation metrics using semantic similarity.
        
        Args:
            prompt: Prompt text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of similar metrics with their metadata and distance score
        """
        try:
            # Generate embedding for the prompt
            embedding = self.embedding_model.encode(prompt).tolist()
            
            with self.db.engine.connect() as connection:
                # Build the SQL query with the embedding vector directly embedded in the query
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                
                # Create the SQL query with the vector data directly in the query string
                query_sql = f"""
                SELECT "id", "prompt", "metadata", 
                       1 - ("embedding" <=> '{embedding_str}'::vector) as similarity
                FROM evaluation_metrics
                ORDER BY "embedding" <=> '{embedding_str}'::vector
                LIMIT :limit
                """
                
                result = connection.execute(
                    text(query_sql),
                    {
                        "limit": limit
                    }
                )
                
                metrics = []
                for row in result:
                    # Handle metadata properly - it could already be a dict
                    metadata = row.metadata
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                        
                    metrics.append({
                        "id": str(row.id),
                        "prompt": row.prompt,
                        "metadata": metadata,
                        "distance": 1.0 - float(row.similarity)
                    })
                
                self.logger.info(f"Found {len(metrics)} similar metrics")
                return metrics
        except Exception as e:
            self.logger.error(f"Failed to search similar metrics: {str(e)}")
            raise
            
    def get_metric_by_id(self, metric_id: str) -> Optional[Dict]:
        """
        Retrieve a metric by its ID.
        
        Args:
            metric_id: ID of the metric to retrieve
            
        Returns:
            Metric data or None if not found
        """
        try:
            with self.db.engine.connect() as connection:
                result = connection.execute(
                    text("""
                    SELECT id, prompt, metadata
                    FROM evaluation_metrics
                    WHERE id = :id
                    """),
                    {"id": metric_id}
                )
                
                row = result.fetchone()
                if not row:
                    return None
                
                # Ensure metadata is properly deserialized from JSON
                metadata = row.metadata
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                    
                return {
                    "id": str(row.id),
                    "prompt": row.prompt,
                    "metadata": metadata
                }
        except Exception as e:
            self.logger.error(f"Failed to get metric by ID: {str(e)}")
            raise
