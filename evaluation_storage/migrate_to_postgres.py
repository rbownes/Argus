"""
Migration script to transfer evaluation metrics from ChromaDB to PostgreSQL.
"""
import os
import sys
import argparse
import logging
import json
from tqdm import tqdm
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migrate_to_postgres")

# Add project root to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation_storage.evaluation_storage import EvaluationStorage
from evaluation_storage.pg_evaluation_storage import PgEvaluationStorage

def migrate_evaluation_metrics(chroma_dir: str, postgres_url: str, batch_size: int = 100):
    """
    Migrate evaluation metrics from ChromaDB to PostgreSQL.
    
    Args:
        chroma_dir: ChromaDB persist directory
        postgres_url: PostgreSQL connection URL
        batch_size: Number of records to process in each batch
    """
    logger.info(f"Initializing ChromaDB source (directory: {chroma_dir})")
    source = EvaluationStorage(persist_directory=chroma_dir)
    
    logger.info(f"Initializing PostgreSQL target (URL: {postgres_url})")
    target = PgEvaluationStorage(db_url=postgres_url)
    
    # Get all unique metric types to process in batches
    logger.info("Fetching all unique metric types from ChromaDB collection")
    # We don't have a direct method to get all metric types, so we'll need to query with empty text
    # and extract unique metric types from the results
    all_metrics_sample = source.collection.query(
        query_texts=[""],
        n_results=10000  # Set a high number to get as many records as possible
    )
    
    metric_types = set()
    if all_metrics_sample["metadatas"] and all_metrics_sample["metadatas"][0]:
        for metadata in all_metrics_sample["metadatas"][0]:
            if metadata and "metric_type" in metadata:
                metric_types.add(metadata["metric_type"])
    
    logger.info(f"Found {len(metric_types)} unique metric types")
    
    total_migrated = 0
    
    # Process each metric type
    for metric_type in metric_types:
        logger.info(f"Processing metric type: {metric_type}")
        
        # Get metrics for this type
        metrics = []
        
        # Query for metrics with this type
        results = source.collection.query(
            query_texts=[""],
            where={"metric_type": metric_type},
            n_results=10000  # Set a high number to get all records
        )
        
        if not results["ids"][0]:
            logger.warning(f"No metrics found for type '{metric_type}'")
            continue
        
        metric_ids = results["ids"][0]
        logger.info(f"Found {len(metric_ids)} metrics with type '{metric_type}'")
        
        # Process in batches
        for i in range(0, len(metric_ids), batch_size):
            batch_ids = metric_ids[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_ids)} metrics")
            
            # Get full data for each metric in batch
            for metric_id in tqdm(batch_ids):
                result = source.collection.get(ids=[metric_id])
                
                if result["ids"]:
                    prompt = result["documents"][0]
                    metadata = result["metadatas"][0]
                    
                    # Store in PostgreSQL
                    target.store_evaluation_metric(
                        prompt=prompt,
                        metric_type=metadata["metric_type"],
                        metadata={k: v for k, v in metadata.items() if k != "metric_type"}
                    )
                    
                    total_migrated += 1
    
    logger.info(f"Migration completed. Total metrics migrated: {total_migrated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate evaluation metrics from ChromaDB to PostgreSQL")
    parser.add_argument(
        "--chroma-dir", 
        type=str, 
        default="/app/evaluation_storage_data",
        help="ChromaDB persist directory"
    )
    parser.add_argument(
        "--postgres-url", 
        type=str, 
        default=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/panopticon"),
        help="PostgreSQL connection URL"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=100,
        help="Batch size for processing records"
    )
    
    args = parser.parse_args()
    
    migrate_evaluation_metrics(
        chroma_dir=args.chroma_dir,
        postgres_url=args.postgres_url,
        batch_size=args.batch_size
    )
