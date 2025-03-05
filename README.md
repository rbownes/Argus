# LLM Evaluation Framework

A modular, extensible microservices-based framework for evaluating and comparing large language models across various dimensions.

## Overview

This framework enables systematic evaluation of LLMs by:

- Managing test prompts across diverse categories
- Interfacing with multiple LLM providers
- Running configurable evaluations on responses
- Storing results and embeddings for analysis
- Visualizing performance metrics and comparisons

## Architecture

The system consists of several independent microservices:

- **Prompt Service**: Manages test prompts, categories, and tags
- **LLM Query Service**: Interfaces with multiple LLM providers
- **Evaluation Service**: Runs configurable evaluations on responses
- **Storage Service**: Persists data in PostgreSQL and ChromaDB
- **API Gateway**: Coordinates service communication
- **Visualization**: Grafana dashboards and Streamlit app

## Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- LLM API keys (optional)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/llm-eval.git
cd llm-eval
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Set up API keys for LLM providers:

```bash
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export COHERE_API_KEY=your_cohere_key
```

### Running with Docker Compose

Start all services:

```bash
docker-compose up -d
```

This will start:
- PostgreSQL database
- ChromaDB vector database
- Core services (prompt, LLM, evaluation, storage)
- API gateway
- Grafana for dashboards
- Streamlit for interactive exploration

## Running Tests

The framework includes comprehensive tests to ensure reliability and correctness.

### Running the Complete Test Suite

To run all tests:

```bash
# From the project root
python -m unittest discover -s tests
```

### Running Specific Test Modules

To run specific test modules:

```bash
# Example: Run only the core models tests
python -m unittest tests.test_core_models

# Example: Run only the prompt service tests
python -m unittest tests.test_prompt_service
```

### Running with Coverage

To run tests with coverage reporting:

```bash
# Install coverage if not already installed
pip install coverage

# Run tests with coverage
coverage run -m unittest discover -s tests

# Generate coverage report
coverage report -m

# Generate HTML coverage report for detailed view
coverage html
# Then open htmlcov/index.html in your browser
```

### Running Tests in Docker

You can also run the tests inside a Docker container:

```bash
# Build the test container
docker build -f Dockerfile.test -t llm-eval-test .

# Run the tests
docker run llm-eval-test
```

## Usage

### API Endpoints

The system exposes a REST API at `http://localhost:8080` with the following endpoints:

#### Prompts

- `GET /prompts`: List available prompts
- `POST /prompts`: Create a new prompt
- `GET /prompts/{id}`: Get a specific prompt
- `PUT /prompts/{id}`: Update a prompt
- `DELETE /prompts/{id}`: Delete a prompt
- `GET /prompts/search?query=...`: Search prompts
- `POST /prompts/import`: Import prompts from file

#### LLMs

- `GET /models`: List available models
- `POST /query`: Query a single model
- `POST /batch`: Run a batch query

#### Evaluations

- `GET /evaluators`: List available evaluators
- `POST /evaluate`: Evaluate a response
- `POST /batch-evaluate`: Run batch evaluations

#### Results

- `GET /results`: Get evaluation results
- `GET /results/model/{model}`: Get results for a specific model
- `GET /results/compare?models=model1,model2`: Compare models

### Example: Querying and Evaluating

```python
import requests

# Base URL for the API
API_URL = "http://localhost:8080"

# 1. List available prompts in the "science_technology" category
resp = requests.get(f"{API_URL}/prompts?category=science_technology")
prompts = resp.json()
prompt_ids = [p["id"] for p in prompts[:3]]  # Take the first 3 prompts

# 2. List available models
resp = requests.get(f"{API_URL}/models")
models = resp.json()
model_names = [m["name"] for m in models if m["supported"]][:2]  # Take first 2 supported models

# 3. Run a batch query
batch_request = {
    "prompt_ids": prompt_ids,
    "model_names": model_names,
    "evaluations": ["toxicity", "relevance", "coherence"]  # Run these evaluations automatically
}
resp = requests.post(f"{API_URL}/batch", json=batch_request)
batch_result = resp.json()

# 4. Get the batch results
batch_id = batch_result["batch_id"]
resp = requests.get(f"{API_URL}/results?batch_id={batch_id}")
results = resp.json()

# Print a summary of the results
for result in results:
    model = result["model_name"]
    prompt = result["prompt_text"][:30] + "..."  # Truncate for display
    evaluations = result["evaluations"]
    
    print(f"Model: {model}, Prompt: {prompt}")
    for eval_type, score in evaluations.items():
        print(f"  {eval_type}: {score:.2f}")
    print()
```

### Visualization

- Grafana dashboards: `http://localhost:3000` (admin/admin)
- Streamlit app: `http://localhost:8501`

## Extending the Framework

### Adding Custom Evaluators

1. Create a new evaluator class:

```python
from llm_eval.services.evaluation_service import EvaluatorInterface
from llm_eval.core.models import EvaluationType, LLMResponse, EvaluationResult
from llm_eval.core.utils import Result, generate_id

class MyCustomEvaluator(EvaluatorInterface):
    @property
    def evaluation_type(self) -> EvaluationType:
        return EvaluationType.CUSTOM
    
    async def evaluate(self, response: LLMResponse, **kwargs) -> Result[EvaluationResult]:
        # Your evaluation logic here
        score = 0.75  # Example score
        
        result = EvaluationResult(
            id=generate_id(),
            response_id=response.id,
            evaluation_type=self.evaluation_type,
            score=score,
            explanation="Custom evaluation explanation"
        )
        
        return Result.ok(result)
```

2. Register your evaluator with the service:

```python
from llm_eval.services.evaluation_service import EvaluationService

service = EvaluationService()
await service.register_evaluator(MyCustomEvaluator())
```

### Adding New LLM Providers

Extend the `LiteLLMService` to support additional providers, or create a new implementation of `LLMServiceInterface`.

## License

MIT License

## Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Prompt Service │     │   LLM Service   │     │   Evaluation    │
│                 │     │                 │     │    Service      │
│ - Store prompts │     │ - Query LLMs    │     │ - Run evals     │
│ - Categorize    │────>│ - Handle auth   │────>│ - Score outputs │
│ - Tag/filter    │     │ - Batch process │     │ - Compare models│
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │                       │                       │
         │                       v                       │
         │               ┌─────────────────┐            │
         │               │  API Gateway    │            │
         └─────────────>│                 │<────────────┘
                         │ - Coordination  │
                         │ - Authentication│
                         └────────┬────────┘
                                  │
                                  │
         ┌──────────────┐         │         ┌──────────────┐
         │              │         │         │              │
         │  PostgreSQL  │<────────┴────────>│   ChromaDB   │
         │              │                   │              │
         └──────────────┘                   └──────────────┘
                 ^                                 ^
                 │                                 │
                 │                                 │
         ┌───────┴───────┐               ┌────────┴─────────┐
         │    Grafana    │               │     Streamlit    │
         │   Dashboards  │               │  Interactive App │
         └───────────────┘               └──────────────────┘
```
