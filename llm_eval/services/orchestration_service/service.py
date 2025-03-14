"""
Orchestration service implementation.
"""
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4

from llm_eval.core.models import (
    ModelConfig, ThemeCategory, QueryPrompt,
    EvaluationRun, ModelResponse, EvaluationResult, EvaluationMetric
)
from llm_eval.services.llm_service.service import LLMQueryService
from llm_eval.services.evaluation_service.service import EvaluationService
from llm_eval.services.storage_service.service import StorageService
from diverse_queries import get_queries_by_theme


class OrchestrationService:
    """Service for orchestrating LLM evaluation runs."""
    
    def __init__(
        self,
        llm_service: LLMQueryService,
        evaluation_service: EvaluationService,
        storage_service: StorageService
    ):
        """Initialize with required services."""
        self.llm_service = llm_service
        self.evaluation_service = evaluation_service
        self.storage_service = storage_service
    
    async def create_evaluation_run(
        self,
        model_configs: List[ModelConfig],
        themes: List[ThemeCategory],
        evaluator_ids: List[str],
        metrics: Optional[List[EvaluationMetric]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        Create and execute a complete evaluation run.
        
        Args:
            model_configs: List of model configurations to evaluate
            themes: List of themes to evaluate
            evaluator_ids: List of evaluator IDs to use
            metrics: Optional list of metrics to evaluate
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate
            metadata: Optional metadata for the run
            
        Returns:
            UUID of the created run
        """
        # Create a new run
        run = EvaluationRun(
            models=model_configs,
            themes=themes,
            metadata=metadata or {}
        )
        
        # Store the run
        await self.storage_service.store_run(run)
        
        # Launch the run asynchronously
        asyncio.create_task(
            self._execute_run(
                run=run,
                evaluator_ids=evaluator_ids,
                metrics=metrics,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )
        
        return run.id
    
    async def _execute_run(
        self,
        run: EvaluationRun,
        evaluator_ids: List[str],
        metrics: Optional[List[EvaluationMetric]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> None:
        """
        Execute an evaluation run.
        
        This internal method handles the actual execution of the run.
        """
        try:
            # Update run status
            run.status = "running"
            await self.storage_service.store_run(run)
            
            # Collect prompts from all selected themes
            all_prompts = []
            for theme in run.themes:
                theme_queries = get_queries_by_theme(theme.value)
                for query in theme_queries:
                    all_prompts.append(QueryPrompt(
                        text=query,
                        theme=theme,
                        metadata={"theme": theme.value}
                    ))
            
            # Query all models with all prompts
            responses = await self.llm_service.query_multiple_models(
                model_configs=run.models,
                prompts=all_prompts,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Store all responses
            for response in responses:
                await self.storage_service.store_response(response)
            
            # Evaluate all responses
            evaluation_results = await self.evaluation_service.evaluate_responses(
                responses=responses,
                evaluator_ids=evaluator_ids,
                metrics=metrics,
                run_id=run.id
            )
            
            # Store all evaluation results
            for result in evaluation_results:
                await self.storage_service.store_evaluation_result(result)
            
            # Update run status and end time
            run.status = "completed"
            run.end_time = datetime.utcnow()
            await self.storage_service.store_run(run)
            
        except Exception as e:
            # Update run status to failed
            run.status = "failed"
            run.end_time = datetime.utcnow()
            run.metadata["error"] = str(e)
            await self.storage_service.store_run(run)
            
            # Re-raise the exception for logging
            raise
    
    async def get_run_status(self, run_id: UUID) -> Dict[str, Any]:
        """
        Get the status of an evaluation run.
        
        Args:
            run_id: UUID of the run
            
        Returns:
            Status and progress information
        """
        # This would need to query the database for the run status
        # For now, return a placeholder
        return {
            "run_id": str(run_id),
            "status": "pending",
            "progress": "0%"
        }
    
    async def get_run_results(self, run_id: UUID) -> Dict[str, Any]:
        """
        Get the results of a completed evaluation run.
        
        Args:
            run_id: UUID of the run
            
        Returns:
            Comprehensive results with metrics by model and theme
        """
        # Get all evaluation results for this run
        results = await self.storage_service.get_evaluation_results_by_run(run_id)
        
        # Process and summarize the results
        # This would need additional logic to structure the results
        return {
            "run_id": str(run_id),
            "results": results
        }
