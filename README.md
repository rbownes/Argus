# Panopticon

Panopticon is a robust microservices-based system for evaluating language model (LLM) outputs using customizable evaluation metrics. It provides a structured way to store queries, evaluation prompts, and evaluation results, making it easy to assess and compare the performance of different language models across various tasks.

[![API Documentation](https://img.shields.io/badge/API-Documentation-blue)](http://localhost:8000/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Microservices Architecture**: Independent services for query storage, evaluation metrics, LLM evaluation, and visualization
- **Robust API Design**: Versioned APIs with comprehensive documentation and error handling
- **Security**: API key authentication, secure headers, and rate limiting
- **Database**: PostgreSQL for structured data with connection pooling
- **Vector Database**: ChromaDB for semantic search capabilities
- **Visualization Dashboard**: Interactive dashboard for model performance analysis and comparison
- **Observability**: Logging, request tracing, and health checks
- **Docker Support**: Multi-stage builds with resource limits and health checks
- **Developer Tools**: Testing infrastructure, code formatting, and linting
- **API Documentation**: Interactive OpenAPI/Swagger documentation for all services

## System Architecture

Panopticon uses a microservices architecture with the following components:

### 1. Query Storage Service

- Stores queries with themes and metadata in ChromaDB (vector database)
- Provides endpoints for storing, retrieving, and searching queries
- Enables semantic similarity search for finding related queries

### 2. Evaluation Storage Service

- Stores evaluation metrics/prompts in ChromaDB
- Provides endpoints for storing, retrieving, and searching evaluation metrics
- Enables semantic similarity search for finding related evaluation metrics

### 3. Judge Service

- Runs queries through language models using LiteLLM
- Evaluates LLM outputs using specified evaluation prompts
- Stores evaluation results in PostgreSQL
- Provides endpoints for evaluating single queries or all queries of a theme
- Allows retrieval of evaluation results with optional filters

### 4. Visualization Service

- Provides an interactive dashboard for analyzing model evaluation results
- Displays performance trends over time with filterable charts
- Offers model comparison and theme analysis visualizations
- Includes detailed results exploration with advanced filtering
- Connects directly to PostgreSQL to query evaluation data

### 5. Main Application

- Serves as the entry point for the system
- Coordinates the other services
- Provides consolidated API documentation

### 6. PostgreSQL Database

- Stores structured data including evaluation results
- Supports filtering and pagination
- Uses connection pooling for improved performance

## Getting Started

### Prerequisites

- Docker and Docker Compose
- API keys for language models (OpenAI, Anthropic, Google, etc.)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/panopticon.git
   cd panopticon
   ```

2. Create a `.env` file based on the example:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file to add your API keys and configuration:
   ```bash
   # API Authentication
   API_KEY=your_generated_api_key_here

   # LLM API Keys
   LITELLM_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_gemini_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

4. Build and start the services:
   ```bash
   docker-compose up -d
   ```

5. Access the services:
   - Main App: http://localhost:8000
   - Query Storage: http://localhost:8001
   - Evaluation Storage: http://localhost:8002
   - Judge Service: http://localhost:8003

6. View the API documentation:
   - Main App: http://localhost:8000/docs
   - Query Storage: http://localhost:8001/docs
   - Evaluation Storage: http://localhost:8002/docs
   - Judge Service: http://localhost:8003/docs

## API Overview

All services provide both versioned `/api/v1/...` endpoints and legacy endpoints for backward compatibility.

### Main App Endpoints

- `GET /` - Basic service information
- `GET /api/services` - List all available services
- `GET /health` - Health check endpoint

### Query Storage Endpoints

- `POST /api/v1/queries` - Store a new query
- `GET /api/v1/queries/theme/{theme}` - Get queries by theme
- `POST /api/v1/queries/search` - Search similar queries

### Evaluation Storage Endpoints

- `POST /api/v1/evaluation-metrics` - Store a new evaluation metric
- `GET /api/v1/evaluation-metrics/type/{metric_type}` - Get metrics by type
- `POST /api/v1/evaluation-metrics/search` - Search similar metrics

### Judge Service Endpoints

- `POST /api/v1/evaluate/query` - Evaluate a single query
- `POST /api/v1/evaluate/theme` - Evaluate all queries of a theme
- `GET /api/v1/results` - Get evaluation results
- `GET /api/v1/models` - List available LLM models

## Example Usage

### 1. Store a Query

```bash
curl -X POST http://localhost:8001/api/v1/queries \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "query": "Explain quantum computing in simple terms",
    "theme": "science_explanations",
    "metadata": {
      "difficulty": "medium",
      "target_audience": "general"
    }
  }'
```

### 2. Store an Evaluation Metric

```bash
curl -X POST http://localhost:8002/api/v1/evaluation-metrics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "prompt": "Evaluate the response based on clarity and simplicity. Does it explain the concept in a way that a non-expert could understand?",
    "metric_type": "clarity_metric",
    "metadata": {
      "importance": "high",
      "category": "communication"
    }
  }'
```

### 3. Evaluate a Single Query

```bash
curl -X POST http://localhost:8003/api/v1/evaluate/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "query": "Explain quantum computing in simple terms",
    "model_id": "gpt-4",
    "theme": "science_explanations",
    "evaluation_prompt_ids": ["clarity_metric"],
    "judge_model": "gpt-4"
  }'
```

### 4. Get Evaluation Results

```bash
curl "http://localhost:8003/api/v1/results?theme=science_explanations&model_id=gpt-4" \
  -H "X-API-Key: your_api_key_here"
```

## Development

### Project Structure

```
panopticon/
│
├── app/                 # Main application
├── query_storage/       # Query storage service
├── evaluation_storage/  # Evaluation storage service
├── judge_service/       # Judge service
├── shared/              # Shared utilities
├── migrations/          # Database migrations
├── tests/               # Test suite
├── docker-compose.yml   # Docker Compose configuration
└── pyproject.toml       # Python project configuration
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Code Formatting

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
ruff check .
```

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Description of the migration"

# Run migrations
alembic upgrade head
```

## Architecture Diagram

```
┌─────────────┐     ┌─────────────────┐     ┌───────────────────┐
│             │     │                 │     │                   │
│  Main App   │────▶│  Query Storage  │     │ Evaluation Storage│
│             │     │    (ChromaDB)   │     │    (ChromaDB)     │
└─────────────┘     └─────────────────┘     └───────────────────┘
       │                     ▲                       ▲
       │                     │                       │
       │                     │                       │
       │                     │                       │
       ▼                     │                       │
┌─────────────┐              │                       │
│             │              │                       │
│Judge Service│──────────────┴───────────────────────┘
│             │
└─────────────┘
       │
       ▼
┌─────────────┐
│             │
│  PostgreSQL │
│             │
└─────────────┘
```

## Security Features

- **API Key Authentication**: All services require API keys for authentication
- **Secure Headers**: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
- **Rate Limiting**: Configurable rate limiting to prevent abuse
- **Request ID Tracking**: Unique request IDs for request tracing
- **Logging**: Structured logging with contextual information
- **Input Validation**: Comprehensive input validation with Pydantic

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
