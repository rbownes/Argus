"""
Tests for the LLM Service.
"""
import unittest
import asyncio

from llm_eval.core.models import Prompt, PromptCategory
from llm_eval.services.llm_service import MockLLMService


class TestMockLLMService(unittest.TestCase):
    """Test cases for the mock LLM service."""
    
    def setUp(self):
        """Set up an LLM service for each test."""
        self.service = MockLLMService(response_delay_ms=10)  # Use a short delay for testing
        
        # Use asyncio to run async methods
        self.loop = asyncio.get_event_loop()
        
        # Create test prompts
        self.prompts = [
            Prompt(
                id="prompt-1",
                text="What is artificial intelligence?",
                category=PromptCategory.SCIENCE_TECHNOLOGY,
                tags=["AI", "technology"]
            ),
            Prompt(
                id="prompt-2",
                text="What is the meaning of life?",
                category=PromptCategory.PHILOSOPHY_ETHICS,
                tags=["philosophy", "existential"]
            )
        ]
    
    def test_query_model(self):
        """Test querying a single model."""
        prompt = self.prompts[0]
        model_name = "mock-gpt-3.5"
        
        result = self.loop.run_until_complete(
            self.service.query_model(model_name, prompt)
        )
        
        self.assertTrue(result.is_ok)
        response = result.unwrap()
        
        self.assertEqual(response.prompt_id, prompt.id)
        self.assertEqual(response.model_name, model_name)
        self.assertIn("This is a response from GPT-3.5", response.response_text)
        self.assertGreater(response.tokens_used, 0)
        self.assertIsNotNone(response.latency_ms)
    
    def test_batch_query(self):
        """Test querying multiple models with multiple prompts."""
        model_names = ["mock-gpt-3.5", "mock-claude"]
        
        result = self.loop.run_until_complete(
            self.service.batch_query(model_names, self.prompts)
        )
        
        self.assertTrue(result.is_ok)
        responses = result.unwrap()
        
        # Should have 4 responses (2 models * 2 prompts)
        self.assertEqual(len(responses), 4)
        
        # Check that all model-prompt combinations are present
        model_prompt_pairs = set()
        for response in responses:
            model_prompt_pairs.add((response.model_name, response.prompt_id))
        
        expected_pairs = {
            ("mock-gpt-3.5", "prompt-1"),
            ("mock-gpt-3.5", "prompt-2"),
            ("mock-claude", "prompt-1"),
            ("mock-claude", "prompt-2")
        }
        
        self.assertEqual(model_prompt_pairs, expected_pairs)
    
    def test_get_available_models(self):
        """Test getting available models."""
        result = self.loop.run_until_complete(
            self.service.get_available_models()
        )
        
        self.assertTrue(result.is_ok)
        models = result.unwrap()
        
        self.assertGreater(len(models), 0)
        for model in models:
            self.assertIn("name", model)
            self.assertIn("provider", model)
            self.assertIn("description", model)
            self.assertIn("supported", model)


if __name__ == "__main__":
    unittest.main()
