"""
Factory for creating item storage instances.
"""
import os
import logging

from .pg_item_storage import PgItemStorage

logger = logging.getLogger("item_storage_factory")

def get_item_storage() -> PgItemStorage:
    """
    Factory function to get an item storage instance.

    Returns:
        A PgItemStorage instance configured with the appropriate database URL and table name
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable is required")
        raise ValueError("DATABASE_URL environment variable is required")

    table_name = os.environ.get("ITEM_STORAGE_TABLE", "items")  # Configurable table name
    logger.info(f"Using PostgreSQL for item storage (table: {table_name})")
    return PgItemStorage(db_url, table_name=table_name)
