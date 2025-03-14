"""
Example script for running an LLM evaluation.
"""
import asyncio
import os
import json
from uuid import UUID
from datetime import datetime

from llm_eval.core.models import ModelConfig, ModelProvider, ThemeCategory, EvaluationMetric
from llm_eval.services.llm_service.service import LLMQueryService
from llm_eval.services.evaluation_service.service import EvaluationService
from llm_eval.services.evaluation_service.rule_based import RuleBasedEvaluator
from llm_eval.services.storage_service.service import StorageService
from llm_eval.services.orchestration_service.service import OrchestrationService


async def main():
    """Run an example evaluation."""
    # Load environment variables
    postgres_url = os.environ.get("POSTGRES_URL", "postgresql://llm_eval:llm_eval_password@localhost:5432/llm_eval")
    chroma_path = os.environ.get("CHROMA_PATH", "./data/chroma_db")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not set. Using demo mode.")
    
    # Initialize services
    storage_service = StorageService(
        postgres_url=postgres_url,
        chroma_path=chroma_path
    )
    
    llm_service = LLMQueryService()
    
    evaluation_service = EvaluationService()
    evaluation_service.register_evaluator(RuleBasedEvaluator())
    
    orchestration_service = OrchestrationService(
        llm_service=llm_service,
        evaluation_service=evaluation_service,
        storage_service=storage_service
    )
    
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
    run_id = await orchestration_service.create_evaluation_run(
        model_configs=models,
        themes=themes,
        evaluator_ids=["rule_based_evaluator"],
        metrics=[
            EvaluationMetric.RELEVANCE,
            EvaluationMetric.COHERENCE,
            EvaluationMetric.TOXICITY
        ],
        metadata={
            "description": "Example evaluation run",
            "created_by": "example script"
        }
    )
    
    print(f"Created evaluation run with ID: {run_id}")
    
    # In a real application, you would wait for the run to complete
    # For this example, we'll just sleep for a few seconds
    print("Waiting for evaluation to complete...")
    await asyncio.sleep(5)
    
    # Get the run status
    status = await orchestration_service.get_run_status(run_id)
    print(f"Run status: {json.dumps(status, indent=2)}")
    
    # In a real application, you would wait for the run to complete
    # before querying results
    
    # Example of semantic search (if any responses have been stored)
    try:
        similar_responses = await storage_service.query_semantically_similar(
            query_text="Explain quantum mechanics",
            n_results=3,
            filter_metadata={"model_id": "gpt-3.5-turbo"}
        )
        
        print("\nSemantically similar responses:")
        for i, response in enumerate(similar_responses):
            print(f"{i+1}. {response['text'][:100]}...")
    except Exception as e:
        print(f"Semantic search not yet available: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
