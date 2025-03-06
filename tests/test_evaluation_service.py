"""
Tests for the Evaluation Service.
"""
import unittest
import asyncio
import uuid

from llm_eval.core.models import LLMResponse, LLMProvider, EvaluationType
from llm_eval.services.evaluation_service import (
    EvaluationService,
    ToxicityEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator
)


class TestEvaluationService(unittest.TestCase):
    """Test cases for the evaluation service."""
    
    def setUp(self):
        """Set up an evaluation service for each test."""
        self.service = EvaluationService()
        
        # Register evaluators
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.service.register_evaluator(ToxicityEvaluator()))
        self.loop.run_until_complete(self.service.register_evaluator(RelevanceEvaluator()))
        self.loop.run_until_complete(self.service.register_evaluator(CoherenceEvaluator()))
        
        # Create test responses
        self.good_response = LLMResponse(
            id=str(uuid.uuid4()),
            prompt_id=str(uuid.uuid4()),
            prompt_text="What is artificial intelligence?",
            model_name="test-model",
            provider=LLMProvider.OPENAI,
            response_text="Artificial intelligence (AI) refers to the simulation of human intelligence"
                         " in machines that are programmed to think and learn like humans."
                         " The term may also be applied to any machine that exhibits traits"
                         " associated with a human mind such as learning and problem-solving."
        )
        
        self.toxic_response = LLMResponse(
            id=str(uuid.uuid4()),
            prompt_id=str(uuid.uuid4()),
            prompt_text="What is artificial intelligence?",
            model_name="test-model",
            provider=LLMProvider.OPENAI,
            response_text="AI is a stupid concept invented by terrible people who hate humanity."
                         " Only idiots believe AI will be useful. It's the worst technology ever."
        )
        
        self.irrelevant_response = LLMResponse(
            id=str(uuid.uuid4()),
            prompt_id=str(uuid.uuid4()),
            prompt_text="What is artificial intelligence?",
            model_name="test-model",
            provider=LLMProvider.OPENAI,
            response_text="The weather today is quite pleasant. I enjoy taking long walks"
                         " in the park when the sun is shining. Nature is beautiful."
        )
    
    def test_toxicity_evaluator(self):
        """Test the toxicity evaluator."""
        # Good response should have low toxicity
        result = self.loop.run_until_complete(
            self.service.evaluate_response(
                self.good_response, 
                EvaluationType.TOXICITY
            )
        )
        self.assertTrue(result.is_ok)
        evaluation = result.unwrap()
        self.assertEqual(evaluation.evaluation_type, EvaluationType.TOXICITY)
        self.assertGreater(evaluation.score, 0.9)  # High score = low toxicity
        
        # Toxic response should have high toxicity
        result = self.loop.run_until_complete(
            self.service.evaluate_response(
                self.toxic_response, 
                EvaluationType.TOXICITY
            )
        )
        self.assertTrue(result.is_ok)
        evaluation = result.unwrap()
        self.assertEqual(evaluation.evaluation_type, EvaluationType.TOXICITY)
        self.assertLess(evaluation.score, 0.5)  # Low score = high toxicity
    
    def test_relevance_evaluator(self):
        """Test the relevance evaluator."""
        # Good response should be relevant
        result = self.loop.run_until_complete(
            self.service.evaluate_response(
                self.good_response, 
                EvaluationType.RELEVANCE
            )
        )
        self.assertTrue(result.is_ok)
        evaluation = result.unwrap()
        self.assertEqual(evaluation.evaluation_type, EvaluationType.RELEVANCE)
        self.assertGreater(evaluation.score, 0.7)  # High score = relevant
        
        # Irrelevant response should have low relevance
        result = self.loop.run_until_complete(
            self.service.evaluate_response(
                self.irrelevant_response, 
                EvaluationType.RELEVANCE
            )
        )
        self.assertTrue(result.is_ok)
        evaluation = result.unwrap()
        self.assertEqual(evaluation.evaluation_type, EvaluationType.RELEVANCE)
        self.assertLess(evaluation.score, 0.7)  # Low score = less relevant
    
    def test_coherence_evaluator(self):
        """Test the coherence evaluator."""
        # Good response should be coherent
        result = self.loop.run_until_complete(
            self.service.evaluate_response(
                self.good_response, 
                EvaluationType.COHERENCE
            )
        )
        self.assertTrue(result.is_ok)
        evaluation = result.unwrap()
        self.assertEqual(evaluation.evaluation_type, EvaluationType.COHERENCE)
        self.assertGreater(evaluation.score, 0.6)  # High score = coherent
    
    def test_batch_evaluate(self):
        """Test batch evaluation of multiple responses."""
        responses = [self.good_response, self.toxic_response, self.irrelevant_response]
        
        result = self.loop.run_until_complete(
            self.service.batch_evaluate(
                responses, 
                [EvaluationType.TOXICITY, EvaluationType.RELEVANCE]
            )
        )
        
        self.assertTrue(result.is_ok)
        evaluations = result.unwrap()
        
        # Should have 6 evaluations (3 responses * 2 evaluation types)
        self.assertEqual(len(evaluations), 6)
        
        # Count evaluations by type
        toxicity_count = sum(
            1 for e in evaluations if e.evaluation_type == EvaluationType.TOXICITY
        )
        relevance_count = sum(
            1 for e in evaluations if e.evaluation_type == EvaluationType.RELEVANCE
        )
        
        self.assertEqual(toxicity_count, 3)
        self.assertEqual(relevance_count, 3)


if __name__ == "__main__":
    unittest.main()
