"""
Interface for the Evaluation Service and evaluators.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

from llm_eval.core.models import LLMResponse, EvaluationResult, EvaluationType
from llm_eval.core.utils import Result


class EvaluatorInterface(ABC):
    """Interface for an individual evaluator."""
    
    @property
    @abstractmethod
    def evaluation_type(self) -> EvaluationType:
        """Get the type of evaluation this evaluator performs."""
        pass
    
    @abstractmethod
    async def evaluate(
        self, 
        response: LLMResponse,
        **kwargs
    ) -> Result[EvaluationResult]:
        """
        Evaluate an LLM response.
        
        Args:
            response: The LLM response to evaluate.
            **kwargs: Additional arguments for the evaluation.
            
        Returns:
            Result containing the evaluation result.
        """
        pass


class EvaluationServiceInterface(ABC):
    """Interface for the evaluation service."""
    
    @abstractmethod
    async def register_evaluator(self, evaluator: EvaluatorInterface) -> None:
        """
        Register an evaluator with the service.
        
        Args:
            evaluator: The evaluator to register.
        """
        pass
    
    @abstractmethod
    async def get_evaluator(
        self, 
        evaluation_type: EvaluationType
    ) -> Result[EvaluatorInterface]:
        """
        Get an evaluator by type.
        
        Args:
            evaluation_type: The type of evaluator to retrieve.
            
        Returns:
            Result containing the evaluator if found.
        """
        pass
    
    @abstractmethod
    async def list_evaluators(self) -> Result[List[EvaluatorInterface]]:
        """
        List all registered evaluators.
        
        Returns:
            Result containing a list of all evaluators.
        """
        pass
    
    @abstractmethod
    async def evaluate_response(
        self,
        response: LLMResponse,
        evaluation_type: EvaluationType,
        **kwargs
    ) -> Result[EvaluationResult]:
        """
        Evaluate a single LLM response with a specific evaluator.
        
        Args:
            response: The LLM response to evaluate.
            evaluation_type: The type of evaluation to perform.
            **kwargs: Additional arguments for the evaluation.
            
        Returns:
            Result containing the evaluation result.
        """
        pass
    
    @abstractmethod
    async def batch_evaluate(
        self,
        responses: List[LLMResponse],
        evaluation_types: Optional[List[EvaluationType]] = None,
        **kwargs
    ) -> Result[List[EvaluationResult]]:
        """
        Perform multiple evaluations on multiple responses.
        
        Args:
            responses: The LLM responses to evaluate.
            evaluation_types: The types of evaluations to perform.
                              If None, uses all registered evaluators.
            **kwargs: Additional arguments for the evaluations.
            
        Returns:
            Result containing a list of evaluation results.
        """
        pass
