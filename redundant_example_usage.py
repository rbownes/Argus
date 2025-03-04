"""
Example usage of the LLM querying and embedding modules.

This script demonstrates how to use the llm_query and embedding_store modules
to query multiple LLM models with diverse prompts, store their responses in a 
vector database, and retrieve them based on model, time, and semantic similarity.
"""

import os
import sys
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from argus.llm_query import query_llm_models, get_batch_summary
    from argus.embedding_store import (
        embed_and_store_model_outputs, 
        query_vector_database,
        get_responses_by_model,
        get_responses_by_time_range,
        get_responses_by_batch_id,
        list_available_batches
    )
    from argus.diverse_queries import get_queries_by_theme, get_themes, get_all_queries
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)


def run_test_queries(num_queries_per_theme: int = 2, themes: List[str] = None) -> None:
    """
    Run test queries against specified LLM models and store the results.
    
    Args:
        num_queries_per_theme: Number of queries to select from each theme
        themes: List of themes to include. If None, uses all available themes.
    """
    # Define the models to query
    models = [
        "gpt-3.5-turbo",  # OpenAI
        # "claude-3-opus-20240229"  # Anthropic
    ]
    
    # Get available themes
    available_themes = themes or get_themes()
    logger.info(f"Using themes: {available_themes}")
    
    # Collect prompts from each theme
    prompts = []
    for theme in available_themes:
        theme_queries = get_queries_by_theme(theme)
        # Take the specified number of queries from each theme
        selected_queries = theme_queries[:num_queries_per_theme]
        prompts.extend(selected_queries)
    
    logger.info(f"Selected {len(prompts)} prompts from {len(available_themes)} themes")
    
    # Query the models
    logger.info(f"Querying models: {models}")
    response_data = query_llm_models(
        models=models,
        prompts=prompts,
        max_tokens=600,  # Limit token length for testing
        temperature=0.7,
        custom_metadata={"purpose": "model_comparison", "test_run": True}
    )
    
    # Get summary statistics
    summary = get_batch_summary(response_data)
    logger.info(f"Query batch summary: {summary}")
    
    # Embed and store the responses
    collection, embedding_model, batch_id, timestamp = embed_and_store_model_outputs(
        model_outputs=response_data,  # Pass structured LLMResponse objects
        embedding_model_name="BAAI/bge-base-en-v1.5",
        persist_directory="./vector_db",
        collection_name="llm_responses_demo",
        additional_metadata={"run_type": "demo", "environment": "testing"}
    )
    
    logger.info(f"Stored responses in batch {batch_id} at {timestamp}")
    
    # Return the collection and batch_id for demonstration queries
    return collection, batch_id, timestamp


def demonstrate_retrieval(collection, batch_id: str) -> None:
    """
    Demonstrate various ways to retrieve responses from the vector database.
    
    Args:
        collection: ChromaDB collection containing the embedded responses
        batch_id: Batch ID to use for retrieval demonstrations
    """
    logger.info("\n--- RETRIEVAL DEMONSTRATIONS ---\n")
    
    # 1. Semantic search across all responses
    logger.info("Demonstration 1: Semantic search")
    results = query_vector_database(
        collection=collection,
        query_text="Explain artificial intelligence concepts",
        n_results=3
    )
    
    logger.info(f"Found {len(results)} semantically similar responses")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: from {result['metadata'].get('model_name')}")
        logger.info(f"Distance: {result['distance']}")
        logger.info(f"Text snippet: {result['text'][:100]}...\n")
    
    # 2. Model-specific retrieval
    logger.info("\nDemonstration 2: Model-specific retrieval")
    gpt_results = get_responses_by_model(
        collection=collection,
        model_name="gpt-3.5-turbo",
        query_text="ethical implications",
        n_results=2
    )
    
    logger.info(f"Found {len(gpt_results)} GPT responses about ethical implications")
    for i, result in enumerate(gpt_results):
        logger.info(f"Result {i+1}:")
        logger.info(f"Prompt: {result['metadata'].get('prompt')}")
        logger.info(f"Text snippet: {result['text'][:100]}...\n")
    
    # 3. Batch-specific retrieval 
    logger.info("\nDemonstration 3: Batch-specific retrieval")
    batch_results = get_responses_by_batch_id(
        collection=collection,
        batch_id=batch_id,
        model_name="claude-3-opus-20240229",  # Filter to Claude responses
        n_results=2
    )
    
    logger.info(f"Found {len(batch_results)} Claude responses in batch {batch_id}")
    for i, result in enumerate(batch_results):
        logger.info(f"Result {i+1}:")
        logger.info(f"Prompt: {result['metadata'].get('prompt')}")
        logger.info(f"Text snippet: {result['text'][:100]}...\n")
    
    # 4. Time-based retrieval
    logger.info("\nDemonstration 4: Time-based retrieval")
    one_hour_ago = datetime.now() - timedelta(hours=1)
    time_results = get_responses_by_time_range(
        collection=collection,
        start_time=one_hour_ago,
        model_name=None,  # All models
        query_text="technology impact",
        n_results=2
    )
    
    logger.info(f"Found {len(time_results)} recent responses about technology impact")
    for i, result in enumerate(time_results):
        logger.info(f"Result {i+1} from {result['metadata'].get('model_name')}:")
        logger.info(f"Distance: {result['distance']}")
        logger.info(f"Text snippet: {result['text'][:100]}...\n")
    
    # 5. List all batches
    logger.info("\nDemonstration 5: List all available batches")
    batches = list_available_batches(collection)
    
    logger.info(f"Found {len(batches)} batches in the collection")
    for i, batch in enumerate(batches[:3]):  # Show up to 3 batches
        logger.info(f"Batch {i+1}: {batch['batch_id']}")
        logger.info(f"  Timestamp: {batch['timestamp']}")
        logger.info(f"  Models: {batch['models']}")
        logger.info(f"  Response count: {batch['count']}\n")


if __name__ == "__main__":
    try:
        # Check that diverse_queries.py exists and can be imported
        query_count = len(get_all_queries())
        logger.info(f"Loaded {query_count} diverse queries")
        
        # Run a small test with 3 themes, 2 queries each
        test_themes = ["science_technology", "philosophy_ethics", "business_economics"]
        collection, batch_id, timestamp = run_test_queries(
            num_queries_per_theme=2,
            themes=test_themes
        )
        
        # Demonstrate different ways to retrieve responses
        demonstrate_retrieval(collection, batch_id)
        
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}", exc_info=True)
        sys.exit(1)