"""
Factory for creating evaluation storage instances.
"""
import os
import logging
import importlib.util
from typing import Union

from .evaluation_storage import EvaluationStorage

logger = logging.getLogger("evaluation_storage_factory")

def get_evaluation_storage() -> Union['EvaluationStorage', 'PgEvaluationStorage']:
    """
    Factory function to get an evaluation storage instance based on configuration.

    Returns:
        An evaluation storage implementation (ChromaDB or PostgreSQL)
    """
    storage_type = os.environ.get("EVALUATION_STORAGE_TYPE", "chroma").lower()

    if storage_type == "postgres":
        # Check if sentence_transformers is available
        if importlib.util.find_spec("sentence_transformers") is None:
            logger.error("sentence_transformers package is not installed")
            logger.warning("Falling back to ChromaDB storage")
            storage_type = "chroma"
        else:
            # Import here to avoid errors if the package is not installed
            from .pg_evaluation_storage import PgEvaluationStorage
            
            db_url = os.environ.get("DATABASE_URL")
            if not db_url:
                logger.error("DATABASE_URL environment variable is required for PostgreSQL storage")
                logger.warning("Falling back to ChromaDB storage")
                storage_type = "chroma"
            else:
                logger.info("Using PostgreSQL with pgvector for evaluation storage")
                return PgEvaluationStorage(db_url)
    
    # Default to ChromaDB
    persist_directory = os.environ.get("EVALUATION_PERSIST_DIRECTORY", "/app/evaluation_storage_data")
    logger.info(f"Using ChromaDB for evaluation storage with persist directory: {persist_directory}")
    return EvaluationStorage(persist_directory)
