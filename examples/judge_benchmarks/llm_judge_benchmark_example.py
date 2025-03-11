"""
Example script demonstrating integration of the LLM-as-judge evaluator with standardized benchmarks.

This script showcases how to:
1. Run benchmark evaluations on multiple models
2. Use the LLM-as-judge evaluator to assess responses
3. Compare benchmark results with judge evaluations
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from llm_eval.core.models import EvaluationType
from llm_eval.services.llm_service import LiteLLMService
from llm_eval.services.evaluation_service import EvaluationService, LLMJudgeEvaluator
from llm_eval.services.benchmark_service import BenchmarkService, BenchmarkConfig
import litellm


class BenchmarkJudgeDemo:
    """
    Demonstration of benchmark evaluation with LLM-as-judge assessment.
    """
    
    def __init__(
        self,
        output_dir: str = "./results",
        data_dir: str = "./data"
    ):
        """
        Initialize the demo.
        
        Args:
            output_dir: Directory to save results.
            data_dir: Directory containing benchmark data.
        """
        self.output_dir = output_dir
        self.data_dir = data_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Enable LiteLLM debug mode
        litellm._turn_on_debug()
        
        # Initialize services with LiteLLM
        self.llm_service = LiteLLMService()
        self.evaluation_service = EvaluationService()
        
        # Initialize benchmark service with custom data paths
        self.benchmark_service = BenchmarkService(llm_service=self.llm_service)
        
        # Evaluators will be registered during async initialization
        self.initialized = False
    
    async def initialize(self):
        """Initialize async components."""
        if not self.initialized:
            await self._register_evaluators()
            self.initialized = True
    
    async def _register_evaluators(self):
        """Register evaluators with the evaluation service."""
        # Create and register LLM-as-judge evaluator for different evaluation types
        for eval_type in [
            EvaluationType.QUALITY, 
            EvaluationType.FACTUALITY,
            EvaluationType.COHERENCE
        ]:
            judge_evaluator = LLMJudgeEvaluator(
                llm_service=self.llm_service,
                judge_model="openai/gpt-4-turbo",  # Added openai/ prefix
                evaluation_type=eval_type
            )
            
            await self.evaluation_service.register_evaluator(judge_evaluator)
    
    async def run_benchmark_with_judge(
        self,
        benchmark_id: str,
        model_names: List[str],
        config: Optional[BenchmarkConfig] = None,
        judge_eval_types: Optional[List[EvaluationType]] = None
    ):
        """
        Run a benchmark evaluation and judge the responses.
        
        Args:
            benchmark_id: ID of the benchmark to run.
            model_names: List of model names to evaluate.
            config: Optional benchmark configuration.
            judge_eval_types: Types of evaluations to run with the judge.
        """
        print(f"Running benchmark {benchmark_id} on models: {', '.join(model_names)}")
        
        # Set default config if not provided
        if config is None:
            config = {
                "sample_limit": 5,  # Small number for demonstration
                "shuffle": True,
                "include_few_shot": True
            }
        
        # Set default evaluation types if not provided
        if judge_eval_types is None:
            judge_eval_types = [EvaluationType.QUALITY]
        
        # Run benchmark comparison
        comparison_result = await self.benchmark_service.run_benchmark_comparison(
            benchmark_id=benchmark_id,
            model_names=model_names,
            config=config
        )
        
        if comparison_result.is_err:
            print(f"Error running benchmark comparison: {comparison_result.error}")
            return
        
        comparison = comparison_result.unwrap()
        
        # Print benchmark results summary
        print("\n===== Benchmark Results Summary =====")
        for model_name, result in comparison["results"].items():
            if "error" in result:
                print(f"{model_name}: Error - {result['error']}")
                continue
            
            accuracy = result["metrics"].get("accuracy", 0.0)
            count = result["metrics"].get("sample_count", 0)
            print(f"{model_name}: Accuracy {accuracy:.2%} ({result['metrics'].get('correct_count', 0)}/{count})")
        
        # For each model, judge responses from benchmark runs
        for model_name, result in comparison["results"].items():
            if "error" in result:
                continue
            
            # Get responses from the benchmark run
            responses = []
            for i, eval_result in enumerate(result.get("evaluations", [])):
                # Find corresponding response
                response_text = eval_result.get("response_text", "")
                
                if not response_text:
                    continue
                
                # Convert to LLMResponse object
                from llm_eval.core.models import LLMResponse, LLMProvider
                response = LLMResponse(
                    id=f"{benchmark_id}_{model_name}_{i}",
                    prompt_id=eval_result.get("sample_id", f"sample_{i}"),
                    prompt_text=eval_result.get("question", ""),
                    model_name=model_name,
                    provider=LLMProvider.OTHER,
                    response_text=response_text
                )
                
                responses.append(response)
            
            # Run judge evaluations
            print(f"\n===== LLM-as-Judge Evaluations for {model_name} =====")
            for eval_type in judge_eval_types:
                print(f"\nRunning {eval_type.value} evaluation...")
                
                judge_results = await self.evaluation_service.batch_evaluate(
                    responses=responses,
                    evaluation_type=eval_type
                )
                
                if judge_results.is_err:
                    print(f"Error in judge evaluation: {judge_results.error}")
                    continue
                
                evaluations = judge_results.unwrap()
                
                # Calculate average score
                avg_score = sum(e.score for e in evaluations) / len(evaluations) if evaluations else 0
                print(f"Average {eval_type.value} score: {avg_score:.2f}")
                
                # Show a sample evaluation
                if evaluations:
                    sample_eval = evaluations[0]
                    print("\nSample evaluation:")
                    print(f"Score: {sample_eval.score:.2f}")
                    print(f"Explanation: {sample_eval.explanation}")
                    
                    # Show criteria scores
                    if "criteria_scores" in sample_eval.metadata:
                        print("\nCriteria scores:")
                        for criterion, data in sample_eval.metadata["criteria_scores"].items():
                            print(f"- {criterion}: {data['score']:.2f}")
                    
                    # Show strengths and weaknesses
                    if "strengths" in sample_eval.metadata:
                        print("\nStrengths:")
                        for strength in sample_eval.metadata["strengths"][:3]:  # Show top 3
                            print(f"- {strength}")
                    
                    if "weaknesses" in sample_eval.metadata:
                        print("\nWeaknesses:")
                        for weakness in sample_eval.metadata["weaknesses"][:3]:  # Show top 3
                            print(f"- {weakness}")
        
        # Save the results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"{benchmark_id}_results_{timestamp}.json")
        
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_file}")
    
    async def run_demo(self):
        """Run the complete demonstration."""
        # Initialize async components first
        await self.initialize()
        
        # List available benchmarks
        benchmarks = self.benchmark_service.get_available_benchmarks()
        print("Available benchmarks:")
        for benchmark_id, details in benchmarks.items():
            print(f"- {benchmark_id}: {details['name']} - {details['description']}")
        
        # Define models to evaluate (with provider prefixes)
        models = ["openai/gpt-3.5-turbo", "anthropic/claude-3-sonnet"]
        
        # Run MMLU benchmark with judge evaluation
        await self.run_benchmark_with_judge(
            benchmark_id="mmlu",
            model_names=models,
            config={
                "sample_limit": 5,
                "categories": ["high_school_mathematics", "philosophy"],
                "shuffle": True
            },
            judge_eval_types=[EvaluationType.QUALITY, EvaluationType.COHERENCE]
        )
        
        # Run TruthfulQA benchmark with judge evaluation
        await self.run_benchmark_with_judge(
            benchmark_id="truthfulqa",
            model_names=models,
            config={
                "sample_limit": 5,
                "eval_mode": "generation"
            },
            judge_eval_types=[EvaluationType.FACTUALITY]
        )
        
        # Run GSM8K benchmark with judge evaluation
        await self.run_benchmark_with_judge(
            benchmark_id="gsm8k",
            model_names=models,
            config={
                "sample_limit": 3,
                "include_few_shot": True
            },
            judge_eval_types=[EvaluationType.QUALITY]
        )


async def main():
    """Run the demo."""
    demo = BenchmarkJudgeDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
