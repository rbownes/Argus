"""
Test script to verify the benchmark and LLM-as-judge integration.
"""
import asyncio
import os
import json
from datetime import datetime

from llm_eval.core.models import EvaluationType
from llm_eval.services.llm_service import MockLLMService
from llm_eval.services.evaluation_service import EvaluationService, LLMJudgeEvaluator
from llm_eval.services.benchmark_service import BenchmarkService, BenchmarkConfig


async def main():
    """Run a simple test of the benchmark and LLM-as-judge integration."""
    print("Initializing services...")
    
    # Initialize services
    llm_service = MockLLMService()
    evaluation_service = EvaluationService()
    
    # Register LLM-as-Judge evaluator
    judge_evaluator = LLMJudgeEvaluator(
        llm_service=llm_service,
        judge_model="gpt-4-turbo",
        evaluation_type=EvaluationType.QUALITY
    )
    await evaluation_service.register_evaluator(judge_evaluator)
    
    # Initialize benchmark service
    benchmark_service = BenchmarkService(llm_service=llm_service)
    
    # Register benchmarks with data paths
    data_dir = os.path.join(os.getcwd(), "data")
    benchmark_service.register_benchmark(
        "mmlu", 
        data_dir=os.path.join(data_dir, "mmlu")
    )
    benchmark_service.register_benchmark(
        "truthfulqa", 
        data_file=os.path.join(data_dir, "truthfulqa/truthfulqa.json")
    )
    benchmark_service.register_benchmark(
        "gsm8k", 
        data_file=os.path.join(data_dir, "gsm8k/gsm8k.jsonl")
    )
    
    # List available benchmarks
    benchmarks = benchmark_service.get_available_benchmarks()
    print("Available benchmarks:")
    for benchmark_id, details in benchmarks.items():
        print(f"- {benchmark_id}: {details['name']}")
        print(f"  Description: {details['description']}")
        print(f"  Categories: {len(details['categories'])} categories")
    
    # Run a sample benchmark
    print("\nRunning MMLU benchmark on gpt-3.5-turbo...")
    result = await benchmark_service.run_benchmark(
        benchmark_id="mmlu",
        model_name="gpt-3.5-turbo",
        config={
            "sample_limit": 3,  # Small limit for quick testing
            "categories": ["philosophy"],
            "shuffle": True
        }
    )
    
    if result.is_err:
        print(f"Error: {result.error}")
    else:
        benchmark_result = result.unwrap()
        print(f"Sample count: {benchmark_result['metrics']['sample_count']}")
        print(f"Accuracy: {benchmark_result['metrics']['accuracy']:.2%}")
        
        # Print detailed results
        print("\nDetailed results:")
        for i, evaluation in enumerate(benchmark_result['evaluations'][:2]):  # Show first 2
            print(f"\nEvaluation {i+1}:")
            print(f"Question: {evaluation['question']}")
            print(f"Answer: {evaluation['expected_answer']}")
            print(f"Model predicted: {evaluation['predicted_answer']}")
            print(f"Correct: {evaluation['correct']}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
