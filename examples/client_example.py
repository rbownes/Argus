"""
Example of using the LLM Evaluation client library.
"""
import asyncio
import os
import json
from datetime import datetime, timedelta

from llm_eval.core.models import ModelConfig, ModelProvider, ThemeCategory, EvaluationMetric
from llm_eval.client import LLMEvalClient


async def main():
    """Run the client example."""
    # Load API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not set. Using demo mode.")
    
    # Initialize client
    async with LLMEvalClient() as client:
        # Configure models to evaluate
        models = [
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_id="gpt-3.5-turbo",
                api_key=openai_api_key
            )
        ]
        
        # Select themes to evaluate
        themes = [
            ThemeCategory.SCIENCE_TECHNOLOGY,
            ThemeCategory.PHILOSOPHY_ETHICS
        ]
        
        # Create an evaluation run
        try:
            print("Creating evaluation run...")
            run_id = await client.create_evaluation(
                models=models,
                themes=themes,
                evaluator_ids=["rule_based_evaluator"],
                metrics=[
                    EvaluationMetric.RELEVANCE,
                    EvaluationMetric.COHERENCE,
                    EvaluationMetric.TOXICITY
                ],
                metadata={
                    "description": "Client example run",
                    "created_by": "client example script"
                }
            )
            
            print(f"Created evaluation run with ID: {run_id}")
            
            # Get the run status
            print("Getting run status...")
            status = await client.get_evaluation_status(run_id)
            print(f"Run status: {json.dumps(status, indent=2)}")
            
            # In a real application, you would poll until the run is complete
            
            # Example of model performance query
            print("\nQuerying model performance...")
            try:
                # Get performance for the last week
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=7)
                
                performance = await client.get_model_performance(
                    model_provider="openai",
                    model_id="gpt-3.5-turbo",
                    start_time=start_time,
                    end_time=end_time
                )
                
                print(f"Model performance: {json.dumps(performance, indent=2)}")
            except Exception as e:
                print(f"Model performance not yet available: {str(e)}")
            
            # Example of semantic search
            print("\nPerforming semantic search...")
            try:
                similar_responses = await client.search_semantically_similar(
                    query_text="Explain quantum mechanics in simple terms",
                    n_results=3,
                    filter_metadata={"model_id": "gpt-3.5-turbo"}
                )
                
                print("Semantically similar responses:")
                for i, response in enumerate(similar_responses):
                    print(f"{i+1}. {response['text'][:100]}...")
            except Exception as e:
                print(f"Semantic search not yet available: {str(e)}")
                
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
