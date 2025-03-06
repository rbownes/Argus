# LLM Evaluation Framework Tests

This directory contains tests for the LLM Evaluation Framework.

## Test Structure

- `test_core_models.py`: Tests for the core data models
- `test_utils.py`: Tests for utility functions
- `test_prompt_service.py`: Tests for the prompt service
- `test_llm_service.py`: Tests for the LLM query service
- `test_evaluation_service.py`: Tests for the evaluation service

## Running Tests

### Using the Run Tests Script

For convenience, you can use the `run_tests.sh` script in the project root:

```bash
# Make the script executable first
python make_executable.py

# Run all tests
./run_tests.sh

# Run with coverage
./run_tests.sh -c

# Run specific module
./run_tests.sh -m test_core_models

# Run in Docker
./run_tests.sh -d

# See all options
./run_tests.sh -h
```

### Manual Test Execution

You can also run tests manually using the Python unittest module:

```bash
# Run all tests
python -m unittest discover -s tests

# Run specific test file
python -m unittest tests.test_core_models

# Run with coverage
coverage run -m unittest discover -s tests
coverage report -m
```

## Writing New Tests

When adding new features, please follow these guidelines for writing tests:

1. Place test files in the `tests` directory with names starting with `test_`
2. Use descriptive test method names that clearly state what is being tested
3. Follow the existing pattern of setting up test fixtures
4. Test both success and error cases
5. For services, test each method of the service interface

Example test structure:

```python
class TestNewFeature(unittest.TestCase):
    
    def setUp(self):
        # Setup code
        pass
    
    def test_successful_case(self):
        # Test successful operation
        pass
    
    def test_error_case(self):
        # Test error handling
        pass
```
