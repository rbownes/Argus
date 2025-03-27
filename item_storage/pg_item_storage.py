"""
Item storage implementation using PostgreSQL with pgvector extension.
"""
import uuid
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sqlalchemy import text, func
from shared.db import Database

logger = logging.getLogger("pg_item_storage")

class PgItemStorage:
    """Storage for items using PostgreSQL with pgvector extension."""
    
    def __init__(self, db_url: str, table_name: str = "items"):
        """
        Initialize PostgreSQL connection and embedding model.
        
        Args:
            db_url: Database connection URL
            table_name: Name of the table to use (e.g., "queries" or "evaluation_metrics")
        """
        self.db = Database(db_url)
        self.table_name = table_name
        self.logger = logging.getLogger(f"pg_item_storage.{table_name}")

        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
            self.logger.info(f"Initialized embedding model for {table_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
        
        # Create tables if they don't exist
        self._create_table()
    
    def _create_table(self):
        """Create necessary table and extensions if they don't exist."""
        try:
            with self.db.engine.connect() as connection:
                # Enable pgvector extension
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                
                # Create items table with vector support
                # Use f-string for table name (generally safe if table_name comes from config)
                connection.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY,
                    item_text TEXT NOT NULL,
                    embedding VECTOR({self.embedding_dimension}),
                    item_type TEXT NOT NULL, -- Corresponds to theme or metric_type
                    metadata JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """))
                
                # Create index for similarity search
                connection.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                ON {self.table_name} USING ivfflat (embedding vector_cosine_ops);
                """))
                
                # Create index on item_type for faster filtering
                connection.execute(text(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_type_idx ON {self.table_name} (item_type);
                """))
                
                connection.commit()
                
                self.logger.info(f"Table '{self.table_name}' initialized.")
        except Exception as e:
            self.logger.error(f"Failed to create table {self.table_name}: {str(e)}")
            raise

    def store_item(self, item_text: str, item_type: str, metadata: Optional[Dict] = None) -> str:
        """
        Store an item with its type and optional metadata.
        
        Args:
            item_text: The item text content
            item_type: Type or category of the item (e.g., 'query', 'evaluation_prompt')
            metadata: Additional metadata for the item
            
        Returns:
            The item ID (UUID)
        """
        item_id = str(uuid.uuid4())
        embedding = self.embedding_model.encode(item_text).tolist()
        item_metadata = {
            "item_type": item_type, # Store type in metadata too for convenience
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        serialized_metadata = json.dumps(item_metadata)

        try:
            with self.db.engine.connect() as connection:
                connection.execute(
                    text(f"""
                    INSERT INTO {self.table_name} (id, item_text, embedding, item_type, metadata, created_at)
                    VALUES (:id, :item_text, :embedding, :item_type, :metadata, NOW())
                    """),
                    {
                        "id": item_id, 
                        "item_text": item_text, 
                        "embedding": embedding,
                        "item_type": item_type, 
                        "metadata": serialized_metadata
                    }
                )
                connection.commit()
            self.logger.info(f"Stored item {item_id} of type '{item_type}' in '{self.table_name}'")
            return item_id
        except Exception as e:
            self.logger.error(f"Failed to store item in {self.table_name}: {str(e)}")
            raise

    def get_items_by_type(self, item_type: str, limit: int = 10, skip: int = 0) -> List[Dict]:
        """
        Retrieve items by type with pagination.
        
        Args:
            item_type: Type to filter by
            limit: Maximum number of results to return
            skip: Number of results to skip for pagination
            
        Returns:
            List of items with their metadata
        """
        try:
            with self.db.engine.connect() as connection:
                result = connection.execute(
                    text(f"""
                    SELECT id, item_text, metadata, created_at
                    FROM {self.table_name}
                    WHERE item_type = :item_type
                    ORDER BY created_at DESC LIMIT :limit OFFSET :skip
                    """),
                    {"item_type": item_type, "limit": limit, "skip": skip}
                )
                items = []
                for row in result:
                    metadata = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
                    items.append({
                        "id": str(row.id),
                        "item": row.item_text, # Use generic 'item' key
                        "metadata": metadata,
                    })
                return items
        except Exception as e:
            self.logger.error(f"Failed to get items by type from {self.table_name}: {str(e)}")
            raise

    def count_items_by_type(self, item_type: str) -> int:
        """
        Count the number of items for a type.
        
        Args:
            item_type: Type to count items for
            
        Returns:
            Number of items for the type
        """
        try:
            with self.db.engine.connect() as connection:
                result = connection.execute(
                    text(f"SELECT COUNT(*) as count FROM {self.table_name} WHERE item_type = :item_type"),
                    {"item_type": item_type}
                ).scalar_one_or_none()
                return result or 0
        except Exception as e:
            self.logger.error(f"Failed to count items by type from {self.table_name}: {str(e)}")
            raise

    def search_similar_items(self, query_text: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar items using semantic similarity.
        
        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of similar items with their metadata and distance score
        """
        embedding = self.embedding_model.encode(query_text).tolist()
        embedding_str = f"[{','.join(map(str, embedding))}]"
        try:
            with self.db.engine.connect() as connection:
                query_sql = f"""
                SELECT id, item_text, metadata,
                       1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM {self.table_name}
                ORDER BY embedding <=> '{embedding_str}'::vector LIMIT :limit
                """
                result = connection.execute(text(query_sql), {"limit": limit})
                items = []
                for row in result:
                    metadata = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
                    items.append({
                        "id": str(row.id),
                        "item": row.item_text,
                        "metadata": metadata,
                        "distance": 1.0 - float(row.similarity)
                    })
                return items
        except Exception as e:
            self.logger.error(f"Failed to search similar items in {self.table_name}: {str(e)}")
            raise

    def get_item_by_id(self, item_id: str) -> Optional[Dict]:
        """
        Retrieve an item by its ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            Item data or None if not found
        """
        try:
            with self.db.engine.connect() as connection:
                result = connection.execute(
                    text(f"SELECT id, item_text, metadata FROM {self.table_name} WHERE id = :id"),
                    {"id": item_id}
                ).first()
                if not result: return None
                metadata = json.loads(result.metadata) if isinstance(result.metadata, str) else result.metadata
                return {"id": str(result.id), "item": result.item_text, "metadata": metadata}
        except Exception as e:
            self.logger.error(f"Failed to get item by ID from {self.table_name}: {str(e)}")
            raise
