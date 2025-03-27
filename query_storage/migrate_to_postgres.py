"""
Migration utility to transfer data from ChromaDB to PostgreSQL.
"""
import argparse
import os
import logging
from typing import Dict, List, Any
import time

from query_storage import QueryStorage
from pg_query_storage import PgQueryStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migration")

def migrate_data(chroma_dir: str, db_url: str, batch_size: int = 50) -> Dict[str, Any]:
    """
    Migrate data from ChromaDB to PostgreSQL.
    
    Args:
        chroma_dir: Directory where ChromaDB data is stored
        db_url: PostgreSQL connection URL
        batch_size: Number of records to process in each batch
        
    Returns:
        Statistics about the migration
    """
    start_time = time.time()
    stats = {
        "total_queries": 0,
        "successful_migrations": 0,
        "failed_migrations": 0,
        "themes": set()
    }
    
    # Initialize both storage systems
    logger.info(f"Initializing ChromaDB storage from {chroma_dir}")
    chroma_storage = QueryStorage(persist_directory=chroma_dir)
    
    logger.info(f"Initializing PostgreSQL storage with connection to {db_url}")
    pg_storage = PgQueryStorage(db_url=db_url)
    
    # Get all themes from ChromaDB
    # Since we don't have a direct "get all themes" method, we'll query with an empty string
    # to get some results, and analyze their metadata to extract themes
    
    # This is a workaround, in a real system with known themes you'd iterate through them
    sample_queries = chroma_storage.search_similar_queries("", limit=1000)
    themes = set()
    for item in sample_queries:
        if "theme" in item["metadata"]:
            themes.add(item["metadata"]["theme"])
    
    logger.info(f"Found {len(themes)} themes: {', '.join(themes)}")
    stats["themes"] = themes
    
    # Migrate data theme by theme
    for theme in themes:
        logger.info(f"Processing theme: {theme}")
        offset = 0
        
        while True:
            # Get a batch of queries for the current theme
            queries = chroma_storage.get_queries_by_theme(theme, limit=batch_size, skip=offset)
            if not queries:
                logger.info(f"No more queries for theme {theme}")
                break
                
            logger.info(f"Processing {len(queries)} queries for theme {theme}, offset {offset}")
            
            # Migrate each query
            for query_data in queries:
                try:
                    # Extract metadata excluding theme (which is handled separately)
                    metadata = {k: v for k, v in query_data["metadata"].items() if k != "theme"}
                    
                    # Store in PostgreSQL
                    pg_storage.store_query(
                        query=query_data["query"],
                        theme=theme,
                        metadata=metadata
                    )
                    
                    stats["successful_migrations"] += 1
                except Exception as e:
                    logger.error(f"Failed to migrate query {query_data['id']}: {str(e)}")
                    stats["failed_migrations"] += 1
            
            stats["total_queries"] += len(queries)
            offset += batch_size
    
    # Compute elapsed time
    end_time = time.time()
    stats["elapsed_time_seconds"] = end_time - start_time
    
    return stats

def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(description="Migrate data from ChromaDB to PostgreSQL")
    parser.add_argument("--chroma-dir", default="/app/query_storage_data",
                       help="Directory where ChromaDB data is stored")
    parser.add_argument("--db-url", required=True,
                       help="PostgreSQL connection URL")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Number of records to process in each batch")
    
    args = parser.parse_args()
    
    logger.info("Starting migration from ChromaDB to PostgreSQL")
    stats = migrate_data(
        chroma_dir=args.chroma_dir,
        db_url=args.db_url,
        batch_size=args.batch_size
    )
    
    # Print summary
    logger.info("Migration complete!")
    logger.info(f"Total queries processed: {stats['total_queries']}")
    logger.info(f"Successfully migrated: {stats['successful_migrations']}")
    logger.info(f"Failed migrations: {stats['failed_migrations']}")
    logger.info(f"Themes: {', '.join(stats['themes'])}")
    logger.info(f"Total time: {stats['elapsed_time_seconds']:.2f} seconds")

if __name__ == "__main__":
    main()
