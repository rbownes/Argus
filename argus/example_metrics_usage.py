"""
Example usage of the LLM metrics collection framework.

This script demonstrates how to use the metrics framework to evaluate
LLM responses stored in ChromaDB.
"""

import os
import sys
from datetime import datetime, timedelta
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from llm_query import query_llm_models, get_batch_summary
    from embedding_store import embed_and_store_model_outputs, get_responses_by_batch_id
    from diverse_queries import get_queries_by_theme, get_themes
    from llm_metrics import (
        create_metric_pipeline, 
        get_batch_ids,
        MetricCategory,
        get_batch_responses
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)


def run_metrics_on_batch(batch_id: str, collection_name: str = "llm_responses", 
                       persist_directory: str = "./chroma_db") -> pd.DataFrame:
    """
    Run all available metrics on a specific batch.
    
    Args:
        batch_id: ID of the batch to evaluate
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory where ChromaDB is stored
        
    Returns:
        DataFrame with metric results
    """
    import chromadb
    
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_collection(collection_name)
    
    # Create metric pipeline
    registry, pipeline = create_metric_pipeline()
    
    # List available metrics
    available_metrics = registry.list_metrics()
    logger.info(f"Available metrics: {len(available_metrics)}")
    for metric in available_metrics:
        logger.info(f"  {metric['name']} ({metric['category']})")
    
    # Get responses for this batch to verify it exists
    responses = get_batch_responses(collection, batch_id)
    if not responses:
        logger.error(f"No responses found for batch {batch_id}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(responses)} responses in batch {batch_id}")
    
    # Get models in the batch
    models = set()
    for response in responses:
        model_name = response["metadata"].get("model_name")
        if model_name:
            models.add(model_name)
    
    logger.info(f"Models in batch: {models}")
    
    # Run all metrics for each model
    all_results = []
    
    # First run model-agnostic metrics
    logger.info("Running metrics across all models in batch")
    results = pipeline.run_all_metrics(batch_id, collection)
    logger.info(f"Completed {len(results)} metrics for all models")
    
    # Then run model-specific metrics
    for model in models:
        logger.info(f"Running metrics for model: {model}")
        model_results = pipeline.run_all_metrics(batch_id, collection, model_name=model)
        logger.info(f"Completed {len(model_results)} metrics for {model}")
    
    # Convert results to DataFrame
    df_results = pipeline.export_results_to_dataframe()
    
    logger.info("Metrics calculation complete")
    return df_results


def generate_sample_batch_for_testing(num_queries: int = 3) -> str:
    """
    Generate a sample batch of LLM queries for testing the metrics framework.
    
    Args:
        num_queries: Number of queries to generate
        
    Returns:
        Batch ID of the generated batch
    """
    # Define models to query
    models = [
        "gpt-3.5-turbo", 
        # "claude-3-opus-20240229"
    ]
    
    # Select sample prompts
    prompts = get_queries_by_theme("science_technology")[:num_queries]
    
    # Query the models
    logger.info(f"Querying {len(models)} models with {len(prompts)} prompts")
    response_data = query_llm_models(
        models=models,
        prompts=prompts,
        max_tokens=300,  # Keep responses small for testing
        temperature=0.7
    )
    
    # Get summary statistics
    summary = get_batch_summary(response_data)
    logger.info(f"Query batch summary: {summary}")
    
    # Embed and store the responses
    logger.info("Storing responses in ChromaDB")
    collection, _, batch_id, _ = embed_and_store_model_outputs(
        model_outputs=response_data,
        embedding_model_name="BAAI/bge-base-en-v1.5",
        persist_directory="./vector_db",
        collection_name="llm_responses_metrics_test",
        additional_metadata={"purpose": "metrics_testing"}
    )
    
    logger.info(f"Generated batch with ID: {batch_id}")
    return batch_id


def visualize_metrics_results(df: pd.DataFrame, output_dir: str = "./metric_results") -> None:
    """
    Visualize metrics results with basic charts.
    
    Args:
        df: DataFrame with metric results
        output_dir: Directory to save visualizations
    """
    if df.empty:
        logger.error("No metrics data to visualize")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data
    csv_path = os.path.join(output_dir, "metrics_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics data to {csv_path}")
    
    # Basic visualizations
    try:
        # 1. Model comparison (if multiple models)
        models = df['model_name'].unique()
        if len(models) > 1 and not all(m is None for m in models):
            plt.figure(figsize=(12, 8))
            
            # Filter to only successful metrics and group by model and metric
            model_comparison = df[df['success'] == True].pivot_table(
                index='metric_name', 
                columns='model_name', 
                values='value',
                aggfunc='mean'
            )
            
            ax = model_comparison.plot(kind='bar')
            plt.title('Metric Values by Model')
            plt.ylabel('Value')
            plt.xlabel('Metric')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, "model_comparison.png"))
            logger.info(f"Saved model comparison chart")
        
        # 2. Metric distribution
        for metric_name in df['metric_name'].unique():
            metric_data = df[df['metric_name'] == metric_name]
            
            if metric_data['success'].all() and len(metric_data) > 0:
                plt.figure(figsize=(10, 6))
                
                # If we have model information, use it for coloring
                if len(models) > 1 and not all(m is None for m in models):
                    for model in models:
                        model_data = metric_data[metric_data['model_name'] == model]
                        if not model_data.empty:
                            plt.hist(model_data['value'], alpha=0.5, label=str(model))
                    plt.legend()
                else:
                    plt.hist(metric_data['value'], bins=10)
                
                plt.title(f'Distribution of {metric_name} Values')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.savefig(os.path.join(output_dir, f"metric_{metric_name}_distribution.png"))
                logger.info(f"Saved distribution chart for {metric_name}")
        
        logger.info("Visualization complete")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


def run_pipeline_for_existing_batches(max_batches: int = 5, 
                                    collection_name: str = "llm_responses",
                                    persist_directory: str = "./chroma_db") -> None:
    """
    Run the metrics pipeline on existing batches in the database.
    
    Args:
        max_batches: Maximum number of batches to process
        collection_name: Name of the ChromaDB collection
        persist_directory: Directory where ChromaDB is stored
    """
    import chromadb
    
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_collection(collection_name)
    
    # Get available batches
    batch_ids = get_batch_ids(collection)
    logger.info(f"Found {len(batch_ids)} batches in the collection")
    
    # Process batches
    for i, batch_id in enumerate(batch_ids[:max_batches]):
        logger.info(f"Processing batch {i+1}/{min(max_batches, len(batch_ids))}: {batch_id}")
        
        # Run metrics
        results_df = run_metrics_on_batch(
            batch_id=batch_id,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # Visualize results
        output_dir = f"./metric_results/{batch_id}"
        visualize_metrics_results(results_df, output_dir)


# Create and demonstrate a custom metric
class WordCountRatioMetric:
    """
    Custom metric that calculates the ratio between unique words and total words.
    A higher value indicates more lexical diversity.
    """
    
    def __init__(self):
        self.name = "WordCountRatioMetric"
        self.category = MetricCategory.CUSTOM
        self.description = "Measures lexical diversity as the ratio of unique words to total words"
        self.dependencies = []
    
    def compute(self, params):
        from llm_metrics import MetricResult, get_batch_responses
        
        # Get responses
        responses = get_batch_responses(
            params.collection, params.batch_id, 
            params.model_name, params.max_samples
        )
        
        if not responses:
            return MetricResult(
                metric_name=self.name,
                batch_id=params.batch_id,
                value=0.0,
                sample_count=0,
                model_name=params.model_name,
                success=False,
                error="No responses found for batch"
            )
        
        # Process each response
        scores = []
        details = []
        
        for response in responses:
            text = response["text"].lower()
            
            # Remove punctuation (simple approach)
            for char in ",.:;!?()[]{}\"'":
                text = text.replace(char, " ")
            
            # Split into words
            all_words = [word for word in text.split() if word]
            unique_words = set(all_words)
            
            # Calculate ratio
            total_words = len(all_words)
            unique_count = len(unique_words)
            
            ratio = unique_count / total_words if total_words > 0 else 0
            
            scores.append(ratio)
            details.append({
                "id": response["id"],
                "unique_words": unique_count,
                "total_words": total_words,
                "ratio": ratio
            })
        
        # Calculate average
        avg_ratio = sum(scores) / len(scores) if scores else 0
        
        return MetricResult(
            metric_name=self.name,
            batch_id=params.batch_id,
            value=avg_ratio,
            sample_count=len(responses),
            model_name=params.model_name,
            details={
                "per_response_details": details,
                "score_distribution": scores
            }
        )
    
    def get_dependencies(self):
        return self.dependencies
    
    def get_info(self):
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "dependencies": []
        }


def demonstrate_custom_metric(batch_id: str):
    """
    Demonstrate how to register and use a custom metric.
    
    Args:
        batch_id: Batch ID to evaluate
    """
    import chromadb
    from llm_metrics import create_metric_pipeline, MetricRegistry
    
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path="./vector_db")
    collection = chroma_client.get_collection("llm_responses_metrics_test")
    
    # Create pipeline
    registry, pipeline = create_metric_pipeline()
    
    # Register custom metric
    registry.register(WordCountRatioMetric)
    
    # Run just the custom metric
    result = pipeline.run_metric("WordCountRatioMetric", batch_id, collection)
    
    logger.info(f"Custom metric result: {result.value}")
    logger.info(f"Details: {json.dumps(result.details, indent=2)}")


if __name__ == "__main__":
    try:
        # Generate a test batch
        batch_id = generate_sample_batch_for_testing(num_queries=3)
        
        # Run metrics on the batch
        results_df = run_metrics_on_batch(
            batch_id=batch_id,
            collection_name="llm_responses_metrics_test",
            persist_directory="./vector_db"
        )
        
        # Visualize results
        visualize_metrics_results(results_df)
        
        # Demonstrate custom metric
        demonstrate_custom_metric(batch_id)
        
        logger.info("Metrics demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Metrics demonstration failed: {e}", exc_info=True)
        sys.exit(1)


# from llm_metrics import create_metric_pipeline

# # Create the pipeline with default metrics
# registry, pipeline = create_metric_pipeline()

# # Run all metrics on a batch
# results = pipeline.run_all_metrics(batch_id, collection)

# # Or run specific metrics
# results = pipeline.run_metrics(["ResponseLengthMetric", "HallucinationMetricAdapter"], 
#                               batch_id, collection)

# class MyCustomMetric(Metric):
#     def __init__(self):
#         super().__init__()
#         self.category = MetricCategory.CUSTOM
    
#     def compute(self, params: MetricParams) -> MetricResult:
#         # Your custom computation logic here
        
# # Register the custom metric
# registry.register(MyCustomMetric)

# # Export to DataFrame
# df = pipeline.export_results_to_dataframe()

# # Save results to JSON
# pipeline.save_results_to_json("metrics_results.json")   