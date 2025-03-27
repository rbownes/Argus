"""
Factory for creating judge storage instances.
"""
import os
import logging
import importlib.util
from typing import Union

from .judge_storage import JudgeStorage

logger = logging.getLogger("judge_storage_factory")

def get_judge_storage() -> Union['JudgeStorage', 'PgJudgeStorage']:
    """
    Factory function to get a judge storage instance based on configuration.
    
    Returns:
        A judge storage implementation (ChromaDB or PostgreSQL)
    """
    storage_type = os.environ.get("JUDGE_STORAGE_TYPE", "chroma").lower()
    
    # Get PostgreSQL URL, which is used by both implementations but differently
    postgres_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/panopticon")
    
    if storage_type == "postgres":
        # Check if sentence_transformers is available
        if importlib.util.find_spec("sentence_transformers") is None:
            logger.error("sentence_transformers package is not installed")
            logger.warning("Falling back to ChromaDB storage")
            storage_type = "chroma"
        else:
            # Import here to avoid errors if the package is not installed
            from .pg_judge_storage import PgJudgeStorage
            
            try:
                logger.info("Using PostgreSQL with pgvector for both LLM outputs and evaluation results")
                return PgJudgeStorage(postgres_url=postgres_url)
            except Exception as e:
                logger.error(f"Failed to initialize PostgreSQL judge storage: {str(e)}")
                logger.warning("Falling back to ChromaDB storage")
                storage_type = "chroma"
    
    # Default to ChromaDB for LLM outputs and PostgreSQL for evaluation results
    persist_directory = os.environ.get("JUDGE_PERSIST_DIRECTORY", "/app/judge_service_data")
    logger.info(f"Using ChromaDB for LLM outputs with persist directory: {persist_directory}")
    logger.info("Using PostgreSQL for evaluation results")
    return JudgeStorage(persist_directory=persist_directory, postgres_url=postgres_url)
