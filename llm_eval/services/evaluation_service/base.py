"""
Base evaluator interface and implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from uuid import UUID

from llm_eval.core.models import ModelResponse, EvaluationMetric, MetricScore, EvaluationResult


class BaseEvaluator(ABC):
    """Base class for implementing LLM evaluators."""
    
    @property
    @abstractmethod
    def evaluator_id(self) -> str:
        """Unique identifier for this evaluator."""
        pass
    
    @property
    @abstractmethod
    def supported_metrics(self) -> List[EvaluationMetric]:
        """List of metrics this evaluator can calculate."""
        pass
    
    @abstractmethod
    async def evaluate(
        self, 
        response: ModelResponse, 
        metrics: Optional[List[EvaluationMetric]] = None,
        run_id: Optional[UUID] = None,
    ) -> EvaluationResult:
        """
        Evaluate a model response.
        
        Args:
            response: The model response to evaluate
            metrics: Specific metrics to evaluate (defaults to all supported metrics)
            run_id: UUID of the evaluation run
            
        Returns:
            EvaluationResult containing scores for the requested metrics
        """
        pass
