"""
Implementation of the Evaluation Service.
"""
import asyncio
from typing import Dict, List, Any, Optional

from llm_eval.core.models import LLMResponse, EvaluationResult, EvaluationType
from llm_eval.core.utils import Result
from .interface import EvaluationServiceInterface, EvaluatorInterface


class EvaluationService(EvaluationServiceInterface):
    """
    Service for evaluating LLM responses using registered evaluators.
    """
    
    def __init__(self):
        """Initialize the evaluation service with an empty evaluator registry."""
        self._evaluators: Dict[EvaluationType, EvaluatorInterface] = {}
    
    async def register_evaluator(self, evaluator: EvaluatorInterface) -> None:
        """Register an evaluator with the service."""
        self._evaluators[evaluator.evaluation_type] = evaluator
    
    async def get_evaluator(
        self, 
        evaluation_type: EvaluationType
    ) -> Result[EvaluatorInterface]:
        """Get an evaluator by type."""
        try:
            if evaluation_type not in self._evaluators:
                return Result.err(
                    KeyError(f"No evaluator registered for type {evaluation_type}")
                )
            return Result.ok(self._evaluators[evaluation_type])
        except Exception as e:
            return Result.err(e)
    
    async def list_evaluators(self) -> Result[List[EvaluatorInterface]]:
        """List all registered evaluators."""
        try:
            return Result.ok(list(self._evaluators.values()))
        except Exception as e:
            return Result.err(e)
    
    async def evaluate_response(
        self,
        response: LLMResponse,
        evaluation_type: EvaluationType,
        **kwargs
    ) -> Result[EvaluationResult]:
        """Evaluate a single LLM response with a specific evaluator."""
        try:
            # Get the requested evaluator
            evaluator_result = await self.get_evaluator(evaluation_type)
            if evaluator_result.is_err:
                return Result.err(evaluator_result.error)
            
            evaluator = evaluator_result.unwrap()
            
            # Perform the evaluation
            return await evaluator.evaluate(response, **kwargs)
        except Exception as e:
            return Result.err(e)
    
    async def batch_evaluate(
        self,
        responses: List[LLMResponse],
        evaluation_types: Optional[List[EvaluationType]] = None,
        **kwargs
    ) -> Result[List[EvaluationResult]]:
        """Perform multiple evaluations on multiple responses."""
        try:
            # Determine which evaluators to use
            if evaluation_types is None:
                # Use all registered evaluators
                evaluators = list(self._evaluators.values())
            else:
                # Use only the specified evaluators
                evaluators = []
                for eval_type in evaluation_types:
                    evaluator_result = await self.get_evaluator(eval_type)
                    if evaluator_result.is_ok:
                        evaluators.append(evaluator_result.unwrap())
            
            if not evaluators:
                return Result.err(ValueError("No evaluators available for the requested evaluation types"))
            
            # Create tasks for all response-evaluator pairs
            tasks = []
            for response in responses:
                for evaluator in evaluators:
                    tasks.append(evaluator.evaluate(response, **kwargs))
            
            # Run all evaluations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            evaluation_results = []
            for result in results:
                if isinstance(result, Result) and result.is_ok:
                    evaluation_results.append(result.unwrap())
                elif isinstance(result, Exception):
                    # Log the error but continue processing
                    print(f"Error during evaluation: {result}")
            
            return Result.ok(evaluation_results)
        except Exception as e:
            return Result.err(e)
