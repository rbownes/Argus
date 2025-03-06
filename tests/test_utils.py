"""
Tests for utility functions.
"""
import unittest
import time
import uuid

from llm_eval.core.utils import generate_id, measure_latency, Result


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_generate_id(self):
        """Test the generate_id function."""
        id1 = generate_id()
        id2 = generate_id()
        
        # IDs should be valid UUIDs
        uuid.UUID(id1)
        uuid.UUID(id2)
        
        # IDs should be unique
        self.assertNotEqual(id1, id2)
    
    def test_measure_latency(self):
        """Test the measure_latency decorator."""
        @measure_latency
        def slow_function():
            time.sleep(0.1)
            return "result"
        
        result, latency = slow_function()
        
        self.assertEqual(result, "result")
        self.assertGreaterEqual(latency, 100)  # at least 100ms
        self.assertLess(latency, 200)  # but not too much more
    
    def test_result_ok(self):
        """Test the Result class with a successful result."""
        result = Result.ok(42)
        
        self.assertTrue(result.is_ok)
        self.assertFalse(result.is_err)
        self.assertEqual(result.unwrap(), 42)
        self.assertEqual(result.unwrap_or(0), 42)
    
    def test_result_err(self):
        """Test the Result class with an error result."""
        error = ValueError("test error")
        result = Result.err(error)
        
        self.assertFalse(result.is_ok)
        self.assertTrue(result.is_err)
        self.assertEqual(result.unwrap_or(0), 0)
        
        with self.assertRaises(ValueError):
            result.unwrap()


if __name__ == "__main__":
    unittest.main()
