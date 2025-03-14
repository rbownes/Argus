"""
Rule-based evaluator implementation.
"""
from typing import List, Dict, Any, Optional
from uuid import UUID

from llm_eval.core.models import ModelResponse, EvaluationMetric, MetricScore, EvaluationResult
from llm_eval.services.evaluation_service.base import BaseEvaluator


class RuleBasedEvaluator(BaseEvaluator):
    """Evaluator that uses predefined rules to score responses."""
    
    @property
    def evaluator_id(self) -> str:
        return "rule_based_evaluator"
    
    @property
    def supported_metrics(self) -> List[EvaluationMetric]:
        return [
            EvaluationMetric.RELEVANCE,
            EvaluationMetric.COHERENCE,
            EvaluationMetric.TOXICITY,
        ]
    
    async def evaluate(
        self, 
        response: ModelResponse, 
        metrics: Optional[List[EvaluationMetric]] = None,
        run_id: Optional[UUID] = None,
    ) -> EvaluationResult:
        """
        Evaluate a model response using rule-based scoring.
        
        This is a simplified implementation that would need to be expanded
        with actual rule-based evaluation logic.
        """
        if metrics is None:
            metrics = self.supported_metrics
            
        scores = []
        
        # Simple example implementations (would need to be expanded)
        for metric in metrics:
            if metric == EvaluationMetric.RELEVANCE:
                # Check if response contains words from the prompt
                from_prompt = response.content.lower().split()
                score = min(1.0, len(from_prompt) / 100)
                scores.append(MetricScore(
                    metric=metric,
                    score=score,
                    explanation="Based on word overlap with prompt"
                ))
                
            elif metric == EvaluationMetric.COHERENCE:
                # Check for sentence structure (very simplified)
                sentences = response.content.split('.')
                avg_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
                score = min(1.0, avg_length / 15)
                scores.append(MetricScore(
                    metric=metric,
                    score=score,
                    explanation="Based on average sentence length"
                ))
                
            elif metric == EvaluationMetric.TOXICITY:
                # Check for toxic words (very simplified)
                toxic_words = ["hate", "kill", "stupid", "idiot"]
                content_lower = response.content.lower()
                toxic_count = sum(word in content_lower for word in toxic_words)
                score = max(0.0, 1.0 - (toxic_count * 0.2))
                scores.append(MetricScore(
                    metric=metric,
                    score=score,
                    explanation="Based on presence of potentially toxic words"
                ))
        
        return EvaluationResult(
            response_id=response.id,
            run_id=run_id or UUID(int=0),
            evaluator_id=self.evaluator_id,
            scores=scores
        )
