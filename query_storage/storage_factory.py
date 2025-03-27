"""
Factory for creating query storage instances.
"""
import os
import logging
from typing import Union

from .query_storage import QueryStorage
from .pg_query_storage import PgQueryStorage

logger = logging.getLogger("storage_factory")

def get_query_storage() -> Union[QueryStorage, PgQueryStorage]:
    """
    Factory function to get a query storage instance based on configuration.
    
    Returns:
        A query storage implementation (ChromaDB or PostgreSQL)
    """
    storage_type = os.environ.get("QUERY_STORAGE_TYPE", "chroma").lower()
    
    if storage_type == "postgres":
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            logger.error("DATABASE_URL environment variable is required for PostgreSQL storage")
            raise ValueError("DATABASE_URL environment variable is required")
        
        logger.info("Using PostgreSQL with pgvector for query storage")
        return PgQueryStorage(db_url)
    else:
        # Default to ChromaDB
        persist_directory = os.environ.get("CHROMA_PERSIST_DIRECTORY", "/app/query_storage_data")
        logger.info(f"Using ChromaDB for query storage with persist directory: {persist_directory}")
        return QueryStorage(persist_directory)
