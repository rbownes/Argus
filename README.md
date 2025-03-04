# LLM Query and Embedding System

This repository contains a modular system for querying multiple LLM models, embedding their responses, and retrieving them based on model, time, or semantic similarity.

## Overview

The system consists of three main components:

1. **LLM Query Module (`llm_query.py`)**: Query multiple LLM models with text prompts and organize their responses
2. **Embedding Store Module (`embedding_store.py`)**: Embed LLM responses using Hugging Face models and store them in a ChromaDB vector database
3. **Diverse Queries (`diverse_queries.py`)**: A collection of 100 diverse queries across 10 themes for testing LLM capabilities

## Installation

### Requirements

- Python 3.7+
- Required packages:
  - `litellm`: For querying different LLM models through a unified API
  - `sentence-transformers`: For text embedding (e.g., BAAI/bge-base-en-v1.5)
  - `chromadb`: For vector storage and retrieval
  - `pydantic`: For data validation

Install the requirements:

```bash
pip install litellm sentence-transformers chromadb pydantic
```

## Usage

### Basic Example

```python
from llm_query import query_llm_models
from embedding_store import embed_and_store_model_outputs, query_vector_database
from diverse_queries import get_queries_by_theme

# 1. Get some diverse prompts
prompts = get_queries_by_theme("science_technology")[:3]  # First 3 science prompts

# 2. Query models
models = ["gpt-3.5-turbo", "claude-3-opus-20240229"]
model_responses = query_llm_models(models, prompts)

# 3. Embed and store responses
collection, embedding_model, batch_id, timestamp = embed_and_store_model_outputs(
    model_outputs=model_responses,
    embedding_model_name="BAAI/bge-base-en-v1.5",
    persist_directory="./my_vector_db"
)

# 4. Retrieve semantically similar responses later
results = query_vector_database(
    collection=collection,
    query_text="Explain artificial intelligence concepts",
    n_results=3
)

# Print the results
for i, result in enumerate(results):
    print(f"Result {i+1} from {result['metadata'].get('model_name')}:")
    print(f"Text: {result['text'][:200]}...")  # First 200 chars
```

### Advanced Retrieval Options

The system offers multiple ways to retrieve responses:

```python
from datetime import datetime, timedelta
from embedding_store import (
    get_responses_by_model,
    get_responses_by_time_range,
    get_responses_by_batch_id
)

# Retrieve by model
gpt_responses = get_responses_by_model(
    collection=collection,
    model_name="gpt-3.5-turbo",
    query_text="ethics of AI"  # Optional semantic search
)

# Retrieve by time range
yesterday = datetime.now() - timedelta(days=1)
recent_responses = get_responses_by_time_range(
    collection=collection,
    start_time=yesterday,
    model_name="claude-3-opus-20240229"  # Optional model filter
)

# Retrieve by batch ID
batch_responses = get_responses_by_batch_id(
    collection=collection,
    batch_id=batch_id
)
```

## Module Details

### LLM Query Module

The `llm_query.py` module provides:

- Functions to query multiple LLM models with a list of prompts
- Structured response objects with metadata
- Batch tracking with unique batch IDs
- Detailed logging and error handling

### Embedding Store Module

The `embedding_store.py` module provides:

- Functions to embed text using Hugging Face models
- Storage of embeddings in ChromaDB with rich metadata
- Retrieval based on semantic similarity, model, time, or batch ID
- Utility functions for managing the vector database

### Diverse Queries Module

The `diverse_queries.py` module provides:

- 100 high-quality, diverse queries across 10 themes
- Each theme contains 10 semantically diverse queries
- Functions to retrieve queries by theme or get all queries

## Example Script

See `example_usage.py` for a complete demonstration of the system's capabilities.

## Best Practices

1. **Use Batch IDs**: Always keep track of batch IDs for easier retrieval and organization
2. **Add Metadata**: Include relevant metadata when storing responses for better filtering
3. **Proper Error Handling**: Handle potential errors during model queries and database operations
4. **Monitor Performance**: Log query times and response lengths to optimize your workflow
5. **Iterative Testing**: Test with small query batches before scaling to larger workloads

## Limitations

- The system relies on external services for model queries (through litellm)
- Large embedding batches may require significant memory
- ChromaDB performance may degrade with very large collections

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Rough Architecture

┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  ChromaDB     │────▶│  Metrics        │────▶│  Results        │
│  Data Source  │     │  Pipeline       │     │  Storage        │
└───────────────┘     └─────────────────┘     └─────────────────┘
                             │
                             ▼
                      ┌─────────────────┐
                      │  Metric         │
                      │  Registry       │
                      └─────────────────┘
                             │
       ┌───────────┬─────────┴─────────┬───────────┐
       ▼           ▼                   ▼           ▼
┌─────────────┐┌─────────────┐  ┌─────────────┐┌─────────────┐
│ Built-in    ││ DeepEval    │  │ Custom      ││ Custom      │
│ Metrics     ││ Metrics     │  │ Metric 1    ││ Metric N    │
└─────────────┘└─────────────┘  └─────────────┘└─────────────┘

┌─────────────────┐     ┌───────────────────┐     ┌───────────────────────┐
│  LLM Outputs    │────▶│  Storage Layer    │────▶│  Visualization Layer  │
└─────────────────┘     └───────────────────┘     └───────────────────────┘
                               │                              │
                         ┌─────┴─────┐               ┌────────┴────────┐
                         │           │               │                 │
                    ┌────▼─────┐┌────▼─────┐    ┌────▼─────┐     ┌────▼─────┐
                    │  SQLite  ││ InfluxDB │    │  Grafana │     │ Streamlit│
                    └──────────┘└──────────┘    └──────────┘     └──────────┘
                    Structured    Time-series    Dashboards &     Interactive
                       Data          Data         Monitoring       Analysis