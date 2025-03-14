# LLM Evaluation System

A comprehensive system for evaluating, tracking, and visualizing LLM model performance over time.

## Overview

This system enables ML Engineers and researchers to:

- Evaluate any LiteLLM-compatible model with a single API call
- Run tests across diverse thematic categories
- Store responses in a vector database with rich metadata
- Track and compare model performance over time
- Visualize performance trends through API endpoints
- Extend the system with custom evaluation metrics
- Scale to handle parallel evaluations across multiple models

## Architecture

The system follows a modular, microservices-oriented architecture with the following components:

```
┌─────────────────────────┐
│                         │
│    FastAPI API Layer    │
│                         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│                         │
│  Orchestration Service  │
│                         │
└───────┬─────┬───────────┘
        │     │
        │     │
┌───────▼─────▼───────────┐
│                         │
│  LLM Query Service      │◄─────────┐
│                         │          │
└───────────┬─────────────┘          │
            │                        │
            ▼                        │
┌─────────────────────────┐          │
│                         │          │
│  Evaluation Service     │          │
│                         │          │
└───────────┬─────────────┘          │
            │                        │
            ▼                        │
┌─────────────────────────┐          │
│                         │          │
│    Storage Service      │          │
│                         │          │
└───────┬───────┬─────────┘          │
        │       │                    │
┌───────▼─┐ ┌───▼───────┐    ┌───────▼───────┐
│         │ │           │    │               │
│ChromaDB │ │PostgreSQL │    │  LLM Models   │
│         │ │           │    │  (via LiteLLM)│
└─────────┘ └───────────┘    └───────────────┘
```

1. **API Layer**: FastAPI application providing endpoints for evaluation and results
2. **Orchestration Service**: Manages evaluation workflow and execution
3. **LLM Query Service**: Handles interaction with LLM models via LiteLLM
4. **Evaluation Service**: Evaluates responses using various metrics
5. **Storage Service**: Stores data in PostgreSQL and ChromaDB
6. **Visualization Layer**: Provides API endpoints for visualization

## Installation

### Requirements

- Python 3.9+
- PostgreSQL database
- Required packages (see `requirements.txt`)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/llm-evaluation-system.git
   cd llm-evaluation-system
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the PostgreSQL database:
   ```bash
   psql -U postgres -f schema.sql
   ```

4. Configure environment variables:
   ```bash
   export POSTGRES_URL="postgresql://user:password@localhost:5432/llm_eval"
   export CHROMA_PATH="./chroma_db"
   ```

5. Run the API server:
   ```bash
   uvicorn llm_eval.services.api_service.app:app --reload
   ```

## Usage

### API Endpoints

The system provides the following API endpoints:

- **POST /api/v1/evaluations**: Create a new evaluation run
- **GET /api/v1/evaluations/{run_id}**: Get the status of an evaluation run
- **GET /api/v1/evaluations/{run_id}/results**: Get the results of an evaluation run
- **POST /api/v1/performance**: Get performance metrics for a specific model
- **POST /api/v1/semantic_search**: Search for semantically similar responses
- **GET /api/v1/models/{model_provider}/{model_id}/responses**: Get responses from a specific model

### Example: Running an Evaluation

```python
import requests
import json

# Create an evaluation run
response = requests.post(
    "http://localhost:8000/api/v1/evaluations",
    json={
        "models": [
            {
                "provider": "openai",
                "model_id": "gpt-3.5-turbo",
                "api_key": "your-openai-api-key"
            }
        ],
        "themes": ["science_technology", "philosophy_ethics"],
        "evaluator_ids": ["rule_based_evaluator"],
        "metrics": ["relevance", "coherence", "toxicity"]
    }
)

run_id = response.json()["run_id"]
print(f"Created evaluation run with ID: {run_id}")

# Check run status
status_response = requests.get(f"http://localhost:8000/api/v1/evaluations/{run_id}")
print(f"Run status: {status_response.json()['status']}")

# Query model performance
performance_response = requests.post(
    "http://localhost:8000/api/v1/performance",
    json={
        "model_provider": "openai",
        "model_id": "gpt-3.5-turbo"
    }
)
print(f"Model performance: {json.dumps(performance_response.json(), indent=2)}")
```

## Extending the System

### Adding Custom Evaluators

Create a new evaluator by implementing the `BaseEvaluator` interface:

```python
from llm_eval.services.evaluation_service.base import BaseEvaluator
from llm_eval.core.models import ModelResponse, EvaluationMetric, MetricScore, EvaluationResult

class MyCustomEvaluator(BaseEvaluator):
    @property
    def evaluator_id(self) -> str:
        return "my_custom_evaluator"
    
    @property
    def supported_metrics(self) -> List[EvaluationMetric]:
        return [EvaluationMetric.CREATIVITY, EvaluationMetric.REASONING]
    
    async def evaluate(
        self, 
        response: ModelResponse, 
        metrics: Optional[List[EvaluationMetric]] = None,
        run_id: Optional[UUID] = None,
    ) -> EvaluationResult:
        # Implement your custom evaluation logic
        scores = []
        
        # Add scores for each metric
        # ...
        
        return EvaluationResult(
            response_id=response.id,
            run_id=run_id or UUID(int=0),
            evaluator_id=self.evaluator_id,
            scores=scores
        )
```

Then register your evaluator with the evaluation service:

```python
from llm_eval.services.evaluation_service.service import EvaluationService
from my_custom_evaluator import MyCustomEvaluator

evaluation_service = EvaluationService()
evaluation_service.register_evaluator(MyCustomEvaluator())
```

## Key Features

- **Diverse Thematic Testing**: Evaluate models across 10 different thematic categories
- **Multi-Dimensional Metrics**: Assess performance across relevance, factual accuracy, coherence, toxicity, and more
- **Semantic Search**: Find semantically similar responses across models and time
- **Temporal Tracking**: Monitor model performance changes over time
- **Extensible Framework**: Easily add new evaluators and metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details.
