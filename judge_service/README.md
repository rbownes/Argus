# Judge Service

This service runs queries through LLM models and evaluates outputs using specified evaluation prompts.

## Storage Options

The service supports two backend storage options:

1. **Hybrid (default)**: 
   - Uses ChromaDB for LLM outputs
   - Uses PostgreSQL for evaluation results

2. **Full PostgreSQL with pgvector**:
   - Uses PostgreSQL with pgvector for both LLM outputs and evaluation results
   - Provides better performance for large datasets and vector similarity search

## Configuration

Storage behavior is controlled via environment variables:

```
# Storage type selection
JUDGE_STORAGE_TYPE=chroma  # or "postgres" for full PostgreSQL

# ChromaDB configuration (when JUDGE_STORAGE_TYPE=chroma)
JUDGE_PERSIST_DIRECTORY=/app/judge_service_data

# PostgreSQL configuration (used by both options)
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/panopticon
```

## Database Schema (PostgreSQL)

### Evaluation Results Table (used by both storage options)

```sql
CREATE TABLE evaluation_results (
    id VARCHAR PRIMARY KEY,
    query_id VARCHAR,
    query_text TEXT,
    output_text TEXT,
    model_id VARCHAR,
    theme VARCHAR,
    evaluation_prompt_id VARCHAR,
    evaluation_prompt TEXT,
    score FLOAT,
    judge_model VARCHAR,
    timestamp TIMESTAMP,
    result_metadata JSONB
);
```

### LLM Outputs Table (when JUDGE_STORAGE_TYPE=postgres)

```sql
CREATE TABLE llm_outputs (
    id UUID PRIMARY KEY,
    output_text TEXT NOT NULL,
    embedding VECTOR(384),       -- 384-dimensional vector from all-MiniLM-L6-v2
    model_id TEXT NOT NULL,
    theme TEXT NOT NULL,
    query TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX llm_outputs_embedding_idx
ON llm_outputs
USING ivfflat (embedding vector_cosine_ops);
```

## Migration from ChromaDB to PostgreSQL

A migration script is provided to transfer LLM outputs data from ChromaDB to PostgreSQL:

```bash
python migrate_to_postgres.py \
  --chroma-dir /app/judge_service_data \
  --postgres-url postgresql://postgres:postgres@postgres:5432/panopticon \
  --batch-size 100
```

## API Endpoints

The service exposes the following REST API endpoints:

### Evaluate a single query
```
POST /api/v1/evaluate/query
{
  "query": "What is the capital of France?",
  "model_id": "gpt-4",
  "theme": "geography",
  "evaluation_prompt_ids": ["8f7e6d5c-4b3a-2c1d-0e9f-8a7b6c5d4e3f"],
  "judge_model": "gpt-4"
}
```

### Evaluate all queries of a theme
```
POST /api/v1/evaluate/theme
{
  "theme": "geography",
  "model_id": "gpt-4",
  "evaluation_prompt_ids": ["8f7e6d5c-4b3a-2c1d-0e9f-8a7b6c5d4e3f"],
  "judge_model": "gpt-4",
  "limit": 10
}
```

### Get evaluation results with filtering
```
GET /api/v1/results?theme=geography&model_id=gpt-4&min_score=7&page=1&limit=10
```

### List available models
```
GET /api/v1/models
```

## Features

- Vector similarity search to find similar outputs
- Automatic model registration from previously unseen model IDs
- Support for multiple evaluation prompts per query
- Detailed metadata storage for analysis

## Performance Considerations

- PostgreSQL with pgvector provides better performance for larger datasets compared to ChromaDB
- Both implementations use the same embedding model (all-MiniLM-L6-v2) with 384 dimensions
- Evaluation results always use PostgreSQL for consistent query performance
