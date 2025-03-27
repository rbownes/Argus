"""
Factory for creating judge storage instances.
"""
import os
import logging

from .pg_judge_storage import PgJudgeStorage

logger = logging.getLogger("judge_storage_factory")

def get_judge_storage() -> PgJudgeStorage:
    """
    Factory function to get a judge storage instance.
    
    Returns:
        A PgJudgeStorage instance for judge storage
    """
    postgres_url = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/panopticon")
    try:
        logger.info("Using PostgreSQL with pgvector for Judge storage")
        return PgJudgeStorage(postgres_url=postgres_url)
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL judge storage: {str(e)}")
        raise  # No fallback - fail if PostgreSQL cannot be used
