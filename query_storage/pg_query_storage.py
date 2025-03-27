"""
Query storage implementation using PostgreSQL with pgvector extension.
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

class PgQueryStorage:
    """Storage for queries using PostgreSQL with pgvector extension."""
    
    def __init__(self, db_url: str):
        """
        Initialize PostgreSQL connection and embedding model.
        
        Args:
            db_url: Database connection URL
        """
        self.logger = logging.getLogger("pg_query_storage")
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
                
                # Create queries table with vector support
                connection.execute(text(f"""
                CREATE TABLE IF NOT EXISTS queries (
                    id UUID PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    embedding VECTOR({self.embedding_dimension}),
                    theme TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """))
                
                # Create index for similarity search
                # Note: This may take time on a large table
                connection.execute(text(f"""
                CREATE INDEX IF NOT EXISTS queries_embedding_idx
                ON queries
                USING ivfflat (embedding vector_cosine_ops);
                """))
                
                connection.commit()
                
                self.logger.info("Query storage tables and extensions initialized")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {str(e)}")
            raise

    def store_query(self, query: str, theme: str, metadata: Optional[Dict] = None) -> str:
        """
        Store a query with its theme and optional metadata.
        
        Args:
            query: The query text
            theme: Theme or category of the query
            metadata: Additional metadata for the query
            
        Returns:
            The query ID (UUID)
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(query).tolist()
            
            # Generate UUID
            query_id = str(uuid.uuid4())
            
            # Prepare metadata
            query_metadata = {
                "theme": theme,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            # Store in PostgreSQL
            with self.db.engine.connect() as connection:
                # Ensure metadata is properly serialized as a JSON string
                serialized_metadata = json.dumps(query_metadata)
                
                connection.execute(
                    text("""
                    INSERT INTO queries (id, query_text, embedding, theme, metadata, created_at)
                    VALUES (:id, :query, :embedding, :theme, :metadata, NOW())
                    """),
                    {
                        "id": query_id,
                        "query": query,
                        "embedding": embedding,
                        "theme": theme,
                        "metadata": serialized_metadata
                    }
                )
                connection.commit()
            
            self.logger.info(f"Stored query with ID {query_id} and theme '{theme}'")
            return query_id
        except Exception as e:
            self.logger.error(f"Failed to store query: {str(e)}")
            raise

    def get_queries_by_theme(self, theme: str, limit: int = 10, skip: int = 0) -> List[Dict]:
        """
        Retrieve queries by theme with pagination.
        
        Args:
            theme: Theme to filter by
            limit: Maximum number of results to return
            skip: Number of results to skip for pagination
            
        Returns:
            List of queries with their metadata
        """
        try:
            with self.db.engine.connect() as connection:
                result = connection.execute(
                    text("""
                    SELECT id, query_text, metadata, created_at
                    FROM queries
                    WHERE theme = :theme
                    ORDER BY created_at DESC
                    LIMIT :limit OFFSET :skip
                    """),
                    {
                        "theme": theme,
                        "limit": limit,
                        "skip": skip
                    }
                )
                
                queries = []
                for row in result:
                    # Ensure metadata is properly deserialized from JSON
                    metadata = row.metadata
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                    
                    queries.append({
                        "id": str(row.id),
                        "query": row.query_text,
                        "metadata": metadata,
                    })
                
                self.logger.info(f"Retrieved {len(queries)} queries with theme '{theme}'")
                return queries
        except Exception as e:
            self.logger.error(f"Failed to get queries by theme: {str(e)}")
            raise

    def count_queries_by_theme(self, theme: str) -> int:
        """
        Count the number of queries for a theme.
        
        Args:
            theme: Theme to count queries for
            
        Returns:
            Number of queries for the theme
        """
        try:
            with self.db.engine.connect() as connection:
                result = connection.execute(
                    text("""
                    SELECT COUNT(*) as count
                    FROM queries
                    WHERE theme = :theme
                    """),
                    {"theme": theme}
                )
                
                count = result.fetchone().count
                self.logger.info(f"Counted {count} queries with theme '{theme}'")
                return count
        except Exception as e:
            self.logger.error(f"Failed to count queries by theme: {str(e)}")
            raise

    def search_similar_queries(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar queries using semantic similarity.
        
        Args:
            query: Query text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of similar queries with their metadata and distance score
        """
        try:
            # Generate embedding for the query
            embedding = self.embedding_model.encode(query).tolist()
            
            with self.db.engine.connect() as connection:
                # Build the SQL query with the embedding vector directly embedded in the query
                embedding_str = f"[{','.join(str(x) for x in embedding)}]"
                
                # Create the SQL query with the vector data directly in the query string - fix column references
                query_sql = f"""
                SELECT "id", "query_text", "metadata", 
                       1 - ("embedding" <=> '{embedding_str}'::vector) as similarity
                FROM queries
                ORDER BY "embedding" <=> '{embedding_str}'::vector
                LIMIT :limit
                """
                
                result = connection.execute(
                    text(query_sql),
                    {
                        "limit": limit
                    }
                )
                
                queries = []
                for row in result:
                    # Handle metadata properly - it could already be a dict
                    metadata = row.metadata
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                        
                    queries.append({
                        "id": str(row.id),
                        "query": row.query_text,
                        "metadata": metadata,
                        "distance": 1.0 - float(row.similarity)
                    })
                
                self.logger.info(f"Found {len(queries)} similar queries")
                return queries
        except Exception as e:
            self.logger.error(f"Failed to search similar queries: {str(e)}")
            raise
            
    def get_query_by_id(self, query_id: str) -> Optional[Dict]:
        """
        Retrieve a query by its ID.
        
        Args:
            query_id: ID of the query to retrieve
            
        Returns:
            Query data or None if not found
        """
        try:
            with self.db.engine.connect() as connection:
                result = connection.execute(
                    text("""
                    SELECT id, query_text, metadata
                    FROM queries
                    WHERE id = :id
                    """),
                    {"id": query_id}
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
                    "query": row.query_text,
                    "metadata": metadata
                }
        except Exception as e:
            self.logger.error(f"Failed to get query by ID: {str(e)}")
            raise
