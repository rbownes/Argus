"""
Tests for core data models.
"""
import unittest
from datetime import datetime

from llm_eval.core.models import (
    Prompt, 
    PromptCategory, 
    LLMResponse, 
    LLMProvider,
    EvaluationType,
    EvaluationResult
)


class TestCoreModels(unittest.TestCase):
    """Test cases for core data models."""
    
    def test_prompt_creation(self):
        """Test creating a Prompt model."""
        prompt = Prompt(
            text="Explain quantum entanglement to a high school student",
            category=PromptCategory.SCIENCE_TECHNOLOGY,
            tags=["physics", "quantum", "educational"]
        )
        
        self.assertEqual(prompt.text, "Explain quantum entanglement to a high school student")
        self.assertEqual(prompt.category, PromptCategory.SCIENCE_TECHNOLOGY)
        self.assertEqual(prompt.tags, ["physics", "quantum", "educational"])
        self.assertIsInstance(prompt.created_at, datetime)
    
    def test_llm_response_creation(self):
        """Test creating an LLMResponse model."""
        response = LLMResponse(
            prompt_id="prompt-123",
            prompt_text="What is AI?",
            model_name="gpt-4",
            provider=LLMProvider.OPENAI,
            response_text="AI stands for artificial intelligence...",
            tokens_used=150,
            latency_ms=2500
        )
        
        self.assertEqual(response.prompt_id, "prompt-123")
        self.assertEqual(response.model_name, "gpt-4")
        self.assertEqual(response.provider, LLMProvider.OPENAI)
        self.assertEqual(response.tokens_used, 150)
        self.assertEqual(response.latency_ms, 2500)
    
    def test_evaluation_result_creation(self):
        """Test creating an EvaluationResult model."""
        eval_result = EvaluationResult(
            response_id="response-123",
            evaluation_type=EvaluationType.FACTUALITY,
            score=0.85,
            explanation="The response is mostly factual, but contains minor inaccuracies."
        )
        
        self.assertEqual(eval_result.response_id, "response-123")
        self.assertEqual(eval_result.evaluation_type, EvaluationType.FACTUALITY)
        self.assertEqual(eval_result.score, 0.85)
        self.assertEqual(
            eval_result.explanation, 
            "The response is mostly factual, but contains minor inaccuracies."
        )


if __name__ == "__main__":
    unittest.main()
