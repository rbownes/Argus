"""
Benchmark service for evaluating LLMs on standardized benchmarks.
"""
from typing import Dict, List, Any, Optional, Type, Union
import asyncio
import os
from datetime import datetime

from llm_eval.core.models import Prompt, LLMResponse
from llm_eval.core.utils import Result, generate_id
from llm_eval.services.benchmark_service.benchmark_interface import BenchmarkInterface
from llm_eval.services.benchmark_service.mmlu_benchmark import MMLUBenchmark, MMLUSample
from llm_eval.services.benchmark_service.truthfulqa_benchmark import TruthfulQABenchmark, TruthfulQASample
from llm_eval.services.benchmark_service.gsm8k_benchmark import GSM8KBenchmark, GSM8KSample
from llm_eval.services.llm_service.interface import LLMServiceInterface


class BenchmarkConfig(Dict[str, Any]):
    """Configuration for a benchmark run."""
    pass


class BenchmarkResult(Dict[str, Any]):
    """Results of a benchmark run."""
    pass


class BenchmarkRun(Dict[str, Any]):
    """Records a benchmark run with its configuration and results."""
    pass


class BenchmarkService:
    """
    Service for evaluating LLMs on standardized benchmarks.
    """
    
    def __init__(self, llm_service: LLMServiceInterface):
        """
        Initialize the benchmark service.
        
        Args:
            llm_service: LLM service for querying models.
        """
        self.llm_service = llm_service
        self.benchmarks = {}
        
        # Register built-in benchmarks
        self.register_benchmark("mmlu", MMLUBenchmark)
        self.register_benchmark("truthfulqa", TruthfulQABenchmark)
        self.register_benchmark("gsm8k", GSM8KBenchmark)
    
    def register_benchmark(
        self,
        benchmark_id: str,
        benchmark_class: Type[BenchmarkInterface],
        **kwargs
    ) -> None:
        """
        Register a benchmark with the service.
        
        Args:
            benchmark_id: Unique ID for the benchmark.
            benchmark_class: Benchmark class to instantiate.
            **kwargs: Additional arguments to pass to the benchmark constructor.
        """
        benchmark = benchmark_class(**kwargs)
        self.benchmarks[benchmark_id] = benchmark
    
    def get_available_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of available benchmarks.
        
        Returns:
            Dict mapping benchmark IDs to their details.
        """
        return {
            benchmark_id: {
                "name": benchmark.name,
                "description": benchmark.description,
                "version": benchmark.version,
                "categories": benchmark.categories
            }
            for benchmark_id, benchmark in self.benchmarks.items()
        }
    
    async def run_benchmark(
        self,
        benchmark_id: str,
        model_name: str,
        config: Optional[BenchmarkConfig] = None
    ) -> Result[BenchmarkRun]:
        """
        Run a benchmark on a specific model.
        
        Args:
            benchmark_id: ID of the benchmark to run.
            model_name: Name of the model to evaluate.
            config: Optional benchmark configuration.
            
        Returns:
            Result containing the benchmark run results.
        """
        try:
            # Validate benchmark ID
            if benchmark_id not in self.benchmarks:
                return Result.err(ValueError(f"Benchmark not found: {benchmark_id}"))
            
            # Get the benchmark
            benchmark = self.benchmarks[benchmark_id]
            
            # Apply default config if not provided
            if config is None:
                config = {}
            
            # Set default values for common config options
            sample_limit = config.get("sample_limit", 100)
            categories = config.get("categories", None)
            shuffle = config.get("shuffle", True)
            include_few_shot = config.get("include_few_shot", True)
            
            # Load benchmark samples
            samples_result = await benchmark.load_samples(
                categories=categories,
                limit=sample_limit,
                shuffle=shuffle
            )
            
            if samples_result.is_err:
                return Result.err(samples_result.error)
            
            samples = samples_result.unwrap()
            
            if not samples:
                return Result.err(ValueError(f"No samples loaded for benchmark: {benchmark_id}"))
            
            # Start timing the benchmark run
            start_time = datetime.now()
            
            # Process each sample
            all_prompts = []
            sample_map = {}
            
            # Create prompts for all samples
            for sample in samples:
                prompt_result = await benchmark.create_prompt(
                    sample=sample,
                    include_few_shot=include_few_shot
                )
                
                if prompt_result.is_err:
                    continue
                
                prompt = prompt_result.unwrap()
                all_prompts.append(prompt)
                sample_map[prompt.id] = sample
            
            # Query the model with all prompts
            responses_result = await self.llm_service.batch_query(
                model_names=[model_name],
                prompts=all_prompts,
                parameters=config.get("model_parameters", {})
            )
            
            if responses_result.is_err:
                return Result.err(responses_result.error)
            
            responses = responses_result.unwrap()
            
            # Evaluate each response
            evaluations = []
            for response in responses:
                sample = sample_map.get(response.prompt_id)
                if not sample:
                    continue
                
                eval_result = await benchmark.evaluate_response(
                    sample=sample,
                    response_text=response.response_text
                )
                
                if eval_result.is_err:
                    continue
                
                evaluation = eval_result.unwrap()
                evaluations.append(evaluation)
            
            # Calculate benchmark metrics
            metrics_result = await benchmark.calculate_benchmark_metrics(evaluations)
            
            if metrics_result.is_err:
                return Result.err(metrics_result.error)
            
            metrics = metrics_result.unwrap()
            
            # End timing
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create benchmark run result
            run_id = generate_id()
            run = {
                "id": run_id,
                "benchmark_id": benchmark_id,
                "model_name": model_name,
                "config": config,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "sample_count": len(samples),
                "evaluated_count": len(evaluations),
                "metrics": metrics,
                "evaluations": evaluations
            }
            
            return Result.ok(run)
        
        except Exception as e:
            return Result.err(e)
    
    async def run_benchmark_comparison(
        self,
        benchmark_id: str,
        model_names: List[str],
        config: Optional[BenchmarkConfig] = None
    ) -> Result[Dict[str, Any]]:
        """
        Run a benchmark comparison across multiple models.
        
        Args:
            benchmark_id: ID of the benchmark to run.
            model_names: Names of the models to evaluate.
            config: Optional benchmark configuration.
            
        Returns:
            Result containing the benchmark comparison results.
        """
        try:
            # Run the benchmark for each model
            results = {}
            
            for model_name in model_names:
                run_result = await self.run_benchmark(
                    benchmark_id=benchmark_id,
                    model_name=model_name,
                    config=config
                )
                
                if run_result.is_err:
                    results[model_name] = {"error": str(run_result.error)}
                else:
                    results[model_name] = run_result.unwrap()
            
            # Create comparison summary
            comparison_id = generate_id()
            comparison = {
                "id": comparison_id,
                "benchmark_id": benchmark_id,
                "model_names": model_names,
                "config": config,
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "summary": self._create_comparison_summary(results)
            }
            
            return Result.ok(comparison)
        
        except Exception as e:
            return Result.err(e)
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of benchmark comparison results."""
        summary = {
            "models": [],
            "overall_scores": {},
            "ranking": []
        }
        
        for model_name, result in results.items():
            if "error" in result:
                continue
            
            summary["models"].append(model_name)
            if "metrics" in result and "overall_score" in result["metrics"]:
                summary["overall_scores"][model_name] = result["metrics"]["overall_score"]
        
        # Rank models by overall score
        ranked_models = sorted(
            summary["overall_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        summary["ranking"] = [{"model": model, "score": score} for model, score in ranked_models]
        
        return summary
