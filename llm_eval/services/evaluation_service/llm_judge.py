"""
LLM-as-judge evaluator implementation.
"""
import json
from typing import List, Dict, Any, Optional
from uuid import UUID

from llm_eval.core.models import ModelConfig, ModelProvider, QueryPrompt, ModelResponse, EvaluationMetric, MetricScore, EvaluationResult
from llm_eval.services.evaluation_service.base import BaseEvaluator
from llm_eval.services.llm_service.service import LLMQueryService


class LLMAsJudgeEvaluator(BaseEvaluator):
    """Evaluator that uses another LLM to judge responses."""
    
    def __init__(self, judge_model_config: ModelConfig):
        """Initialize with a model config for the judge LLM."""
        self.judge_model_config = judge_model_config
        self.llm_service = LLMQueryService()
    
    @property
    def evaluator_id(self) -> str:
        return f"llm_judge_{self.judge_model_config.model_id}"
    
    @property
    def supported_metrics(self) -> List[EvaluationMetric]:
        return [
            EvaluationMetric.RELEVANCE,
            EvaluationMetric.FACTUAL_ACCURACY,
            EvaluationMetric.COHERENCE,
            EvaluationMetric.CREATIVITY,
            EvaluationMetric.REASONING,
            EvaluationMetric.INSTRUCTION_FOLLOWING,
        ]
    
    def _create_judge_prompt(
        self, 
        response: ModelResponse, 
        metrics: List[EvaluationMetric]
    ) -> QueryPrompt:
        """Create a prompt for the judge LLM."""
        metrics_str = ", ".join([m.value for m in metrics])
        prompt_text = f"""
        You are an expert evaluator of language model outputs. Your task is to evaluate the following response 
        to a given prompt based on these criteria: {metrics_str}.
        
        For each criterion, provide a score from 0.0 to 1.0 and a brief explanation of your reasoning.
        
        Original prompt:
        {response.model_config.prompt_text}
        
        Response to evaluate:
        {response.content}
        
        Please format your evaluation as a JSON object with the following structure:
        {{
            "evaluations": [
                {{
                    "metric": "metric_name",
                    "score": 0.0-1.0,
                    "explanation": "Brief explanation"
                }}
            ]
        }}
        """
        
        return QueryPrompt(
            text=prompt_text,
            metadata={
                "response_id": str(response.id),
                "metrics": [m.value for m in metrics]
            }
        )
    
    def _parse_judge_response(
        self,
        judge_response: ModelResponse, 
        metrics: List[EvaluationMetric]
    ) -> List[MetricScore]:
        """Parse the judge's response to extract scores."""
        try:
            # Try to parse the JSON response
            response_json = json.loads(judge_response.content)
            
            scores = []
            for evaluation in response_json.get("evaluations", []):
                metric_name = evaluation.get("metric")
                metric = None
                
                # Find the matching EvaluationMetric enum
                for m in metrics:
                    if m.value == metric_name:
                        metric = m
                        break
                
                if metric:
                    scores.append(MetricScore(
                        metric=metric,
                        score=float(evaluation.get("score", 0.5)),
                        explanation=evaluation.get("explanation", "")
                    ))
            
            # If we couldn't parse any scores, create default ones
            if not scores:
                scores = [
                    MetricScore(
                        metric=metric,
                        score=0.5,
                        explanation="Unable to extract score from judge response"
                    )
                    for metric in metrics
                ]
            
            return scores
            
        except Exception as e:
            # If parsing fails, return default scores
            return [
                MetricScore(
                    metric=metric,
                    score=0.5,
                    explanation=f"Error parsing judge response: {str(e)}"
                )
                for metric in metrics
            ]
    
    async def evaluate(
        self, 
        response: ModelResponse, 
        metrics: Optional[List[EvaluationMetric]] = None,
        run_id: Optional[UUID] = None,
    ) -> EvaluationResult:
        """
        Evaluate a model response using an LLM as a judge.
        """
        if metrics is None:
            metrics = self.supported_metrics
        
        # Create prompt for the judge
        judge_prompt = self._create_judge_prompt(response, metrics)
        
        # Query the judge LLM
        judge_response = await self.llm_service.query_model(
            model_config=self.judge_model_config,
            prompt=judge_prompt,
            temperature=0.2,  # Low temperature for more deterministic responses
            max_tokens=1024
        )
        
        # Parse the judge's response
        scores = self._parse_judge_response(judge_response, metrics)
        
        # Create and return evaluation result
        return EvaluationResult(
            response_id=response.id,
            run_id=run_id or UUID(int=0),
            evaluator_id=self.evaluator_id,
            scores=scores,
            metadata={
                "judge_prompt": judge_prompt.text,
                "judge_response": judge_response.content
            }
        )
