"""
Evaluation service implementation.
"""
import asyncio
from typing import List, Dict, Any, Optional, Type
from uuid import UUID

from llm_eval.core.models import ModelResponse, EvaluationMetric, EvaluationResult
from llm_eval.services.evaluation_service.base import BaseEvaluator
from llm_eval.services.evaluation_service.rule_based import RuleBasedEvaluator


class EvaluationService:
    """Service for evaluating LLM responses."""
    
    def __init__(self):
        """Initialize with available evaluators."""
        self.evaluators: Dict[str, BaseEvaluator] = {}
        
        # Register default evaluators
        self.register_evaluator(RuleBasedEvaluator())
    
    def register_evaluator(self, evaluator: BaseEvaluator) -> None:
        """Register a new evaluator."""
        self.evaluators[evaluator.evaluator_id] = evaluator
    
    async def evaluate_response(
        self,
        response: ModelResponse,
        evaluator_id: str,
        metrics: Optional[List[EvaluationMetric]] = None,
        run_id: Optional[UUID] = None,
    ) -> EvaluationResult:
        """
        Evaluate a model response using a specific evaluator.
        
        Args:
            response: The model response to evaluate
            evaluator_id: ID of the evaluator to use
            metrics: Specific metrics to evaluate
            run_id: UUID of the evaluation run
            
        Returns:
            EvaluationResult containing scores
        """
        if evaluator_id not in self.evaluators:
            raise ValueError(f"Evaluator {evaluator_id} not found")
            
        evaluator = self.evaluators[evaluator_id]
        return await evaluator.evaluate(response, metrics, run_id)
    
    async def evaluate_responses(
        self,
        responses: List[ModelResponse],
        evaluator_ids: List[str],
        metrics: Optional[List[EvaluationMetric]] = None,
        run_id: Optional[UUID] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple responses using multiple evaluators.
        
        Args:
            responses: List of model responses to evaluate
            evaluator_ids: List of evaluator IDs to use
            metrics: Specific metrics to evaluate
            run_id: UUID of the evaluation run
            
        Returns:
            List of EvaluationResults
        """
        tasks = []
        for response in responses:
            for evaluator_id in evaluator_ids:
                tasks.append(
                    self.evaluate_response(
                        response=response,
                        evaluator_id=evaluator_id,
                        metrics=metrics,
                        run_id=run_id
                    )
                )
        
        return await asyncio.gather(*tasks)
