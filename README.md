# Panopticon

[![API Documentation](https://img.shields.io/badge/API-Documentation-blue)](http://localhost:8000/docs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Panopticon is a comprehensive system for evaluating and benchmarking language model (LLM) performance across diverse tasks and criteria. Built on a microservices architecture, it enables organizations to systematically test, evaluate, and compare multiple language models using customizable evaluation metrics.

## Core API Endpoints

Panopticon's API endpoints enable comprehensive LLM evaluation workflows. Here are the key endpoints:

### Judge Service Endpoints

The Judge Service is responsible for evaluating LLM outputs against specific criteria:

#### Model Evaluation

- **POST `/api/v1/evaluate/query`**: Evaluate a single query using a specific model
  ```json
  {
    "query": "Explain quantum computing to a 5-year-old",
    "model_id": "gpt-4-turbo",
    "theme": "explanations",
    "evaluation_prompt_ids": ["clarity", "accuracy"],
    "judge_model": "claude-3-opus-20240229",
    "model_provider": "openai"
  }
  ```

- **POST `/api/v1/evaluate/theme`**: Evaluate all queries of a specific theme
  ```json
  {
    "theme": "reasoning_tasks",
    "model_id": "gemini-pro",
    "evaluation_prompt_ids": ["correctness", "clarity"],
    "judge_model": "gpt-4",
    "model_provider": "gemini"
  }
  ```

#### Model and Result Management

- **GET `/api/v1/models`**: List all available models for evaluation
- **GET `/api/v1/results`**: Retrieve evaluation results with filtering

### Query Storage Service Endpoints

The Query Storage Service manages evaluation test cases:

- **POST `/api/v1/queries`**: Add a new query for evaluation
  ```json
  {
    "query": "What are three ways to address climate change?",
    "theme": "environmental_problems",
    "metadata": {
      "difficulty": "medium",
      "required_knowledge": "science, policy"
    }
  }
  ```

- **GET `/api/v1/queries/theme/{theme}`**: Get all queries for a specific theme
- **POST `/api/v1/queries/search`**: Search for similar queries

### Evaluation Storage Service Endpoints

The Evaluation Storage Service manages evaluation metrics:

- **POST `/api/v1/evaluation-metrics`**: Create a new evaluation metric
  ```json
  {
    "prompt": "Rate the clarity of this explanation on a scale of 1-10",
    "metric_type": "clarity",
    "metadata": {
      "description": "Evaluates how clear and understandable the explanation is"
    }
  }
  ```

- **GET `/api/v1/evaluation-metrics/{metric_id}`**: Get a specific evaluation metric
- **GET `/api/v1/evaluation-metrics/type/{metric_type}`**: Get all metrics of a specific type

### Visualization Service Endpoints

The Visualization Service provides analytics and visualization features:

- **GET `/api/v1/dashboard/summary`**: Get overview of evaluation results
- **GET `/api/v1/dashboard/models`**: Get comparative model performance data
- **GET `/api/v1/dashboard/themes`**: Get performance breakdown by themes

## End-to-End Workflow Examples

### Example 1: Evaluating Models from Different Providers

This example shows how to evaluate models from OpenAI, Anthropic, and Google on the same set of tasks.

#### Step 1: Create Test Queries

First, create a set of reasoning tasks to evaluate:

```bash
# Add a mathematical reasoning query
curl -X POST http://localhost:8001/api/v1/queries \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "theme": "reasoning_tasks",
    "metadata": {
      "type": "mathematical",
      "difficulty": "medium"
    }
  }'

# Add a logical reasoning query
curl -X POST http://localhost:8001/api/v1/queries \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "If all A are B, and all B are C, what can we conclude about A and C?",
    "theme": "reasoning_tasks",
    "metadata": {
      "type": "logical",
      "difficulty": "medium"
    }
  }'
```

#### Step 2: Create Evaluation Metrics

Define metrics to evaluate model responses:

```bash
# Correctness metric
curl -X POST http://localhost:8002/api/v1/evaluation-metrics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "Evaluate the correctness of this answer. Is the reasoning sound and is the final answer correct? Rate on a scale of 1-10.",
    "metric_type": "correctness",
    "metadata": {
      "domain": "reasoning"
    }
  }'

# Clarity metric
curl -X POST http://localhost:8002/api/v1/evaluation-metrics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "Evaluate how clearly the response explains its reasoning. Is the explanation easy to follow? Rate on a scale of 1-10.",
    "metric_type": "clarity",
    "metadata": {
      "domain": "explanation"
    }
  }'
```

#### Step 3: Evaluate Each Provider's Model

##### OpenAI GPT-4 Evaluation

```bash
curl -X POST http://localhost:8003/api/v1/evaluate/theme \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "theme": "reasoning_tasks",
    "model_id": "gpt-4-turbo",
    "model_provider": "openai",
    "evaluation_prompt_ids": ["correctness", "clarity"],
    "judge_model": "claude-3-opus-20240229"
  }'
```

##### Anthropic Claude Evaluation

```bash
curl -X POST http://localhost:8003/api/v1/evaluate/theme \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "theme": "reasoning_tasks",
    "model_id": "claude-3-opus-20240229",
    "model_provider": "anthropic",
    "evaluation_prompt_ids": ["correctness", "clarity"],
    "judge_model": "gpt-4-turbo"
  }'
```

##### Google Gemini Evaluation

```bash
curl -X POST http://localhost:8003/api/v1/evaluate/theme \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "theme": "reasoning_tasks",
    "model_id": "gemini-pro",
    "model_provider": "gemini",
    "evaluation_prompt_ids": ["correctness", "clarity"],
    "judge_model": "gpt-4-turbo"
  }'
```

#### Step 4: View and Compare Results

Access the visualization dashboard at http://localhost:8004 to view and compare performance across providers:

1. Navigate to the "Model Comparison" view
2. Select all three models in the filter options
3. Compare performance metrics across providers
4. Drill down into detailed results for specific queries

### Example 2: Using OpenAI Models for Content Generation and Evaluation

This example demonstrates using OpenAI models for both content generation and evaluation.

#### Step 1: Create Evaluation Tasks

```bash
# Add a creative writing task
curl -X POST http://localhost:8001/api/v1/queries \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "Write a short story about a robot discovering its emotions for the first time.",
    "theme": "creative_writing",
    "metadata": {
      "type": "narrative",
      "length": "short"
    }
  }'
```

#### Step 2: Create Evaluation Metrics

```bash
# Creativity metric
curl -X POST http://localhost:8002/api/v1/evaluation-metrics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "Evaluate the creativity and originality of this story. Does it present novel ideas and perspectives? Rate on a scale of 1-10.",
    "metric_type": "creativity",
    "metadata": {
      "domain": "creative_writing"
    }
  }'

# Emotional impact metric
curl -X POST http://localhost:8002/api/v1/evaluation-metrics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "Evaluate the emotional impact of this story. Does it evoke strong emotions or feelings? Rate on a scale of 1-10.",
    "metric_type": "emotional_impact",
    "metadata": {
      "domain": "creative_writing"
    }
  }'
```

#### Step 3: Generate and Evaluate Content

Generate content with GPT-3.5 and evaluate with GPT-4:

```bash
# Evaluate a single query using GPT-3.5 with GPT-4 as judge
curl -X POST http://localhost:8003/api/v1/evaluate/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "Write a short story about a robot discovering its emotions for the first time.",
    "model_id": "gpt-3.5-turbo",
    "theme": "creative_writing",
    "evaluation_prompt_ids": ["creativity", "emotional_impact"],
    "judge_model": "gpt-4-turbo",
    "model_provider": "openai"
  }'
```

#### Step 4: Retrieve and Analyze Results

```bash
# Get evaluation results for the creative writing theme
curl -X GET "http://localhost:8003/api/v1/results?theme=creative_writing&model_id=gpt-3.5-turbo" \
  -H "X-API-Key: your_api_key"
```

### Example 3: Evaluating Anthropic Claude Models

This example shows how to evaluate different versions of Anthropic's Claude models.

#### Step 1: Create Test Queries for Instruction Following

```bash
# Add an instruction following task
curl -X POST http://localhost:8001/api/v1/queries \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "Please analyze this poem line by line: [insert poem]. Format your answer as a table with three columns: Line Number, Text, and Analysis.",
    "theme": "instruction_following",
    "metadata": {
      "complexity": "high",
      "formatting_required": true
    }
  }'
```

#### Step 2: Create Evaluation Metrics

```bash
# Instruction following metric
curl -X POST http://localhost:8002/api/v1/evaluation-metrics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "Evaluate how well the response follows the given instructions. Does it complete all required tasks and format the output as requested? Rate on a scale of 1-10.",
    "metric_type": "instruction_following",
    "metadata": {
      "domain": "task_completion"
    }
  }'
```

#### Step 3: Evaluate Different Claude Models

```bash
# Evaluate Claude 3 Opus
curl -X POST http://localhost:8003/api/v1/evaluate/theme \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "theme": "instruction_following",
    "model_id": "claude-3-opus-20240229",
    "model_provider": "anthropic",
    "evaluation_prompt_ids": ["instruction_following"],
    "judge_model": "gpt-4-turbo"
  }'

# Evaluate Claude 3 Sonnet
curl -X POST http://localhost:8003/api/v1/evaluate/theme \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "theme": "instruction_following",
    "model_id": "claude-3-sonnet-20240229",
    "model_provider": "anthropic",
    "evaluation_prompt_ids": ["instruction_following"],
    "judge_model": "gpt-4-turbo"
  }'
```

#### Step 4: Compare Claude Model Versions

Compare the performance of different Claude models on the "Model Comparison" dashboard to identify performance differences between Claude model versions.

### Example 4: Evaluating Google Gemini Models

This example demonstrates evaluating Google's Gemini models on scientific knowledge tasks.

#### Step 1: Create Scientific Knowledge Queries

```bash
# Add a science question
curl -X POST http://localhost:8001/api/v1/queries \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "Explain quantum entanglement and its implications for quantum computing.",
    "theme": "scientific_knowledge",
    "metadata": {
      "field": "physics",
      "difficulty": "advanced"
    }
  }'
```

#### Step 2: Create Evaluation Metrics

```bash
# Scientific accuracy metric
curl -X POST http://localhost:8002/api/v1/evaluation-metrics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "Evaluate the scientific accuracy of this response. Are there any factual errors or misrepresentations of scientific concepts? Rate on a scale of 1-10.",
    "metric_type": "scientific_accuracy",
    "metadata": {
      "domain": "science"
    }
  }'
```

#### Step 3: Evaluate Gemini Model

```bash
# Evaluate Gemini Pro
curl -X POST http://localhost:8003/api/v1/evaluate/theme \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "theme": "scientific_knowledge",
    "model_id": "gemini-pro",
    "model_provider": "gemini",
    "evaluation_prompt_ids": ["scientific_accuracy", "clarity"],
    "judge_model": "claude-3-opus-20240229"
  }'
```

#### Step 4: Analyze Results

Visualize Gemini's performance on scientific knowledge tasks using the dashboard, focusing on accuracy scores and comparing them to other models if available.

## Cross-Provider Comparison Workflow

This workflow demonstrates how to set up a comprehensive comparison between models from different providers.

### Step 1: Set Up Testing Dataset

Create a diverse set of queries across multiple themes:

```bash
# Reasoning tasks
curl -X POST http://localhost:8001/api/v1/queries \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "theme": "reasoning",
    "metadata": {"type": "mathematical"}
  }'

# Creative tasks
curl -X POST http://localhost:8001/api/v1/queries \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "Write a poem about artificial intelligence in the style of Robert Frost.",
    "theme": "creative",
    "metadata": {"type": "poetry"}
  }'

# Knowledge tasks
curl -X POST http://localhost:8001/api/v1/queries \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "query": "Explain how mRNA vaccines work and how they differ from traditional vaccines.",
    "theme": "knowledge",
    "metadata": {"field": "biology"}
  }'
```

### Step 2: Define Standard Metrics

```bash
# Accuracy metric
curl -X POST http://localhost:8002/api/v1/evaluation-metrics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "Evaluate the factual accuracy of this response. Rate on a scale of 1-10.",
    "metric_type": "accuracy"
  }'

# Creativity metric
curl -X POST http://localhost:8002/api/v1/evaluation-metrics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "Evaluate the creativity and originality of this response. Rate on a scale of 1-10.",
    "metric_type": "creativity"
  }'

# Clarity metric
curl -X POST http://localhost:8002/api/v1/evaluation-metrics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "Evaluate how clearly explained this response is. Rate on a scale of 1-10.",
    "metric_type": "clarity"
  }'
```

### Step 3: Run Comprehensive Evaluation

Evaluate each model provider on each theme:

```bash
# Models to evaluate
MODELS=(
  "gpt-4-turbo:openai"
  "claude-3-opus-20240229:anthropic"
  "gemini-pro:gemini"
)

# Themes to evaluate
THEMES=("reasoning" "creative" "knowledge")

# Loop through models and themes
for model_info in "${MODELS[@]}"; do
  IFS=':' read -r model_id provider <<< "$model_info"
  
  for theme in "${THEMES[@]}"; do
    curl -X POST http://localhost:8003/api/v1/evaluate/theme \
      -H "Content-Type: application/json" \
      -H "X-API-Key: your_api_key" \
      -d '{
        "theme": "'$theme'",
        "model_id": "'$model_id'",
        "model_provider": "'$provider'",
        "evaluation_prompt_ids": ["accuracy", "clarity", "creativity"],
        "judge_model": "gpt-4-turbo"
      }'
  done
done
```

### Step 4: Visualize Comprehensive Results

1. Open the visualization dashboard at http://localhost:8004
2. Navigate to the "Model Comparison" view
3. Compare all models across all themes and metrics
4. Use radar charts to identify each provider's strengths and weaknesses
5. Export comparison data for further analysis

## Configuration and Setup

### Environment Variables

The system uses the following environment variables for configuration:

```
# API Keys for LLM Providers
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key

# Model Parameters
MODEL_DEFAULT_TEMPERATURE=0.7

# Authentication
API_KEY=your_internal_api_key

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=panopticon
```

### Starting the System

Use Docker Compose to start all services:

```bash
docker-compose up -d
```

This will start all required services on their respective ports:
- Main App: http://localhost:8000
- Query Storage: http://localhost:8001
- Evaluation Storage: http://localhost:8002
- Judge Service: http://localhost:8003
- Visualization Service: http://localhost:8004

### API Documentation

Access API documentation for each service at:
- Main App: http://localhost:8000/docs
- Query Storage: http://localhost:8001/docs
- Evaluation Storage: http://localhost:8002/docs
- Judge Service: http://localhost:8003/docs
- Visualization Service: http://localhost:8004/docs
