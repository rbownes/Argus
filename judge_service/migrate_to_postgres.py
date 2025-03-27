"""
Migration script to transfer LLM outputs from ChromaDB to PostgreSQL.
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

from judge_service.judge_storage import JudgeStorage
from judge_service.pg_judge_storage import PgJudgeStorage

def migrate_llm_outputs(chroma_dir: str, postgres_url: str, batch_size: int = 100):
    """
    Migrate LLM outputs from ChromaDB to PostgreSQL.
    
    Args:
        chroma_dir: ChromaDB persist directory
        postgres_url: PostgreSQL connection URL
        batch_size: Number of records to process in each batch
    """
    logger.info(f"Initializing ChromaDB source (directory: {chroma_dir})")
    source = JudgeStorage(persist_directory=chroma_dir, postgres_url=postgres_url)
    
    logger.info(f"Initializing PostgreSQL target (URL: {postgres_url})")
    target = PgJudgeStorage(postgres_url=postgres_url)
    
    # Get all unique themes to process in batches
    logger.info("Fetching all unique themes from ChromaDB collection")
    # We don't have a direct method to get all themes, so we'll need to query with empty text
    # and extract unique themes from the results
    all_outputs_sample = source.collection.query(
        query_texts=[""],
        n_results=10000  # Set a high number to get as many records as possible
    )
    
    themes = set()
    models = set()
    if all_outputs_sample["metadatas"] and all_outputs_sample["metadatas"][0]:
        for metadata in all_outputs_sample["metadatas"][0]:
            if metadata:
                if "theme" in metadata:
                    themes.add(metadata["theme"])
                if "model_id" in metadata:
                    models.add(metadata["model_id"])
    
    logger.info(f"Found {len(themes)} unique themes")
    logger.info(f"Found {len(models)} unique models")
    
    total_migrated = 0
    
    # First, process by theme
    for theme in themes:
        logger.info(f"Processing theme: {theme}")
        
        # For each theme, get model IDs
        for model_id in models:
            logger.info(f"Processing model: {model_id}")
            
            # Get outputs for this theme and model
            outputs = []
            
            # Query for outputs with this theme and model
            results = source.collection.query(
                query_texts=[""],
                where={"theme": theme, "model_id": model_id},
                n_results=10000  # Set a high number to get all records
            )
            
            if not results["ids"][0]:
                logger.warning(f"No outputs found for theme '{theme}' and model '{model_id}'")
                continue
            
            output_ids = results["ids"][0]
            logger.info(f"Found {len(output_ids)} outputs with theme '{theme}' and model '{model_id}'")
            
            # Process in batches
            for i in range(0, len(output_ids), batch_size):
                batch_ids = output_ids[i:i+batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_ids)} outputs")
                
                # Get full data for each output in batch
                for output_id in tqdm(batch_ids):
                    result = source.collection.get(ids=[output_id])
                    
                    if result["ids"]:
                        output_text = result["documents"][0]
                        metadata = result["metadatas"][0]
                        
                        # Extract required fields
                        model_id = metadata.get("model_id")
                        theme = metadata.get("theme")
                        query = metadata.get("query")
                        
                        if not (model_id and theme and query):
                            logger.warning(f"Skipping output {output_id} due to missing required metadata")
                            continue
                        
                        # Prepare metadata without the fields that are stored in the table columns
                        filtered_metadata = {
                            k: v for k, v in metadata.items() 
                            if k not in ["model_id", "theme", "query"]
                        }
                        
                        # Store in PostgreSQL
                        with target.db.engine.connect() as connection:
                            # Generate embedding for the output text
                            embedding = target.embedding_model.encode(output_text).tolist()
                            
                            # Ensure metadata is properly serialized as a JSON string
                            serialized_metadata = json.dumps(filtered_metadata)
                            
                            from sqlalchemy import text
                            
                            # Check if the output already exists in the database
                            exists_result = connection.execute(
                                text("SELECT COUNT(*) FROM llm_outputs WHERE id = :id"),
                                {"id": output_id}
                            ).scalar()
                            
                            if exists_result > 0:
                                logger.warning(f"Output {output_id} already exists in the database, skipping")
                                continue
                            
                            connection.execute(
                                text("""
                                INSERT INTO llm_outputs (id, output_text, embedding, model_id, theme, query, metadata, created_at)
                                VALUES (:id, :output_text, :embedding, :model_id, :theme, :query, :metadata, NOW())
                                """),
                                {
                                    "id": output_id,
                                    "output_text": output_text,
                                    "embedding": embedding,
                                    "model_id": model_id,
                                    "theme": theme,
                                    "query": query,
                                    "metadata": serialized_metadata
                                }
                            )
                            connection.commit()
                            total_migrated += 1
    
    logger.info(f"Migration completed. Total outputs migrated: {total_migrated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate LLM outputs from ChromaDB to PostgreSQL")
    parser.add_argument(
        "--chroma-dir", 
        type=str, 
        default="/app/judge_service_data",
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
    
    migrate_llm_outputs(
        chroma_dir=args.chroma_dir,
        postgres_url=args.postgres_url,
        batch_size=args.batch_size
    )
