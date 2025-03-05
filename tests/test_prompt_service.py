"""
Tests for the Prompt Service.
"""
import unittest
import os
import tempfile
import json
import asyncio

from llm_eval.core.models import Prompt, PromptCategory
from llm_eval.services.prompt_service import InMemoryPromptService


class TestInMemoryPromptService(unittest.TestCase):
    """Test cases for the in-memory prompt service."""
    
    def setUp(self):
        """Set up a prompt service for each test."""
        self.service = InMemoryPromptService()
        
        # Use asyncio to run async methods
        self.loop = asyncio.get_event_loop()
    
    def test_create_and_get_prompt(self):
        """Test creating and retrieving a prompt."""
        prompt = Prompt(
            text="What is the meaning of life?",
            category=PromptCategory.PHILOSOPHY_ETHICS,
            tags=["philosophy", "existential"]
        )
        
        # Create prompt
        result = self.loop.run_until_complete(self.service.create_prompt(prompt))
        self.assertTrue(result.is_ok)
        created_prompt = result.unwrap()
        self.assertIsNotNone(created_prompt.id)
        
        # Get prompt
        result = self.loop.run_until_complete(self.service.get_prompt(created_prompt.id))
        self.assertTrue(result.is_ok)
        retrieved_prompt = result.unwrap()
        self.assertEqual(retrieved_prompt.text, prompt.text)
        self.assertEqual(retrieved_prompt.category, prompt.category)
        self.assertEqual(retrieved_prompt.tags, prompt.tags)
    
    def test_list_prompts(self):
        """Test listing prompts with filters."""
        # Create test prompts
        prompts = [
            Prompt(
                text="What is quantum physics?",
                category=PromptCategory.SCIENCE_TECHNOLOGY,
                tags=["physics", "quantum"]
            ),
            Prompt(
                text="How do vaccines work?",
                category=PromptCategory.HEALTH_MEDICINE,
                tags=["medical", "vaccines"]
            ),
            Prompt(
                text="What is moral relativism?",
                category=PromptCategory.PHILOSOPHY_ETHICS,
                tags=["philosophy", "ethics"]
            )
        ]
        
        for prompt in prompts:
            self.loop.run_until_complete(self.service.create_prompt(prompt))
        
        # List all prompts
        result = self.loop.run_until_complete(self.service.list_prompts())
        self.assertTrue(result.is_ok)
        all_prompts = result.unwrap()
        self.assertEqual(len(all_prompts), 3)
        
        # Filter by category
        result = self.loop.run_until_complete(
            self.service.list_prompts(category=PromptCategory.SCIENCE_TECHNOLOGY)
        )
        self.assertTrue(result.is_ok)
        science_prompts = result.unwrap()
        self.assertEqual(len(science_prompts), 1)
        self.assertEqual(science_prompts[0].text, "What is quantum physics?")
        
        # Filter by tags
        result = self.loop.run_until_complete(
            self.service.list_prompts(tags=["philosophy"])
        )
        self.assertTrue(result.is_ok)
        philosophy_prompts = result.unwrap()
        self.assertEqual(len(philosophy_prompts), 1)
        self.assertEqual(philosophy_prompts[0].text, "What is moral relativism?")
    
    def test_update_prompt(self):
        """Test updating a prompt."""
        # Create a prompt
        prompt = Prompt(
            text="What is climate change?",
            category=PromptCategory.ENVIRONMENT_SUSTAINABILITY,
            tags=["climate", "environment"]
        )
        
        result = self.loop.run_until_complete(self.service.create_prompt(prompt))
        self.assertTrue(result.is_ok)
        created_prompt = result.unwrap()
        
        # Update the prompt
        updated_prompt = Prompt(
            text="What causes climate change?",
            category=PromptCategory.ENVIRONMENT_SUSTAINABILITY,
            tags=["climate", "environment", "causes"]
        )
        
        result = self.loop.run_until_complete(
            self.service.update_prompt(created_prompt.id, updated_prompt)
        )
        self.assertTrue(result.is_ok)
        
        # Verify update
        result = self.loop.run_until_complete(self.service.get_prompt(created_prompt.id))
        self.assertTrue(result.is_ok)
        retrieved_prompt = result.unwrap()
        self.assertEqual(retrieved_prompt.text, "What causes climate change?")
        self.assertEqual(len(retrieved_prompt.tags), 3)
        self.assertIn("causes", retrieved_prompt.tags)
    
    def test_delete_prompt(self):
        """Test deleting a prompt."""
        # Create a prompt
        prompt = Prompt(
            text="Test prompt for deletion",
            category=PromptCategory.OTHER
        )
        
        result = self.loop.run_until_complete(self.service.create_prompt(prompt))
        self.assertTrue(result.is_ok)
        created_prompt = result.unwrap()
        
        # Delete the prompt
        result = self.loop.run_until_complete(
            self.service.delete_prompt(created_prompt.id)
        )
        self.assertTrue(result.is_ok)
        self.assertTrue(result.unwrap())
        
        # Verify deletion
        result = self.loop.run_until_complete(self.service.get_prompt(created_prompt.id))
        self.assertTrue(result.is_err)
        self.assertIsInstance(result.error, KeyError)
    
    def test_search_prompts(self):
        """Test searching prompts."""
        # Create test prompts
        prompts = [
            Prompt(
                text="What is artificial intelligence?",
                category=PromptCategory.SCIENCE_TECHNOLOGY,
                tags=["AI", "technology"]
            ),
            Prompt(
                text="How do neural networks learn?",
                category=PromptCategory.SCIENCE_TECHNOLOGY,
                tags=["AI", "neural networks"]
            ),
            Prompt(
                text="What is the Turing test?",
                category=PromptCategory.SCIENCE_TECHNOLOGY,
                tags=["AI", "Turing"]
            )
        ]
        
        for prompt in prompts:
            self.loop.run_until_complete(self.service.create_prompt(prompt))
        
        # Search by text
        result = self.loop.run_until_complete(self.service.search_prompts("neural"))
        self.assertTrue(result.is_ok)
        neural_prompts = result.unwrap()
        self.assertEqual(len(neural_prompts), 1)
        self.assertEqual(neural_prompts[0].text, "How do neural networks learn?")
        
        # Search by tag
        result = self.loop.run_until_complete(self.service.search_prompts("turing"))
        self.assertTrue(result.is_ok)
        turing_prompts = result.unwrap()
        self.assertEqual(len(turing_prompts), 1)
        self.assertEqual(turing_prompts[0].text, "What is the Turing test?")
    
    def test_import_prompts_from_json(self):
        """Test importing prompts from a JSON file."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "prompts": [
                    {
                        "text": "What is machine learning?",
                        "category": "science_technology",
                        "tags": ["AI", "machine learning"]
                    },
                    {
                        "text": "How does reinforcement learning work?",
                        "category": "science_technology",
                        "tags": ["AI", "reinforcement learning"]
                    }
                ]
            }, f)
            temp_file = f.name
        
        try:
            # Import prompts
            result = self.loop.run_until_complete(
                self.service.import_prompts_from_file(temp_file)
            )
            self.assertTrue(result.is_ok)
            imported_prompts = result.unwrap()
            self.assertEqual(len(imported_prompts), 2)
            
            # Verify imported prompts
            texts = [p.text for p in imported_prompts]
            self.assertIn("What is machine learning?", texts)
            self.assertIn("How does reinforcement learning work?", texts)
        finally:
            # Clean up
            os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()
