# Evaluation Storage Service

This service stores and retrieves evaluation metrics and prompts for the Panopticon system.

## Storage Options

The service supports two backend storage options:

1. **ChromaDB** (default): Uses ChromaDB vector database
2. **PostgreSQL with pgvector**: Uses PostgreSQL with pgvector extension for vector similarity search

## Configuration

Storage behavior is controlled via environment variables:

```
# Storage type selection
EVALUATION_STORAGE_TYPE=chroma  # or "postgres" for PostgreSQL

# ChromaDB configuration (when EVALUATION_STORAGE_TYPE=chroma)
EVALUATION_PERSIST_DIRECTORY=/app/evaluation_storage_data

# PostgreSQL configuration (when EVALUATION_STORAGE_TYPE=postgres)
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/panopticon
```

## Database Schema (PostgreSQL)

When using PostgreSQL, the following schema is used:

```sql
CREATE TABLE evaluation_metrics (
    id UUID PRIMARY KEY,
    prompt TEXT NOT NULL,
    embedding VECTOR(384),       -- 384-dimensional vector from all-MiniLM-L6-v2
    metric_type TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX evaluation_metrics_embedding_idx
ON evaluation_metrics
USING ivfflat (embedding vector_cosine_ops);
```

## Migration from ChromaDB to PostgreSQL

A migration script is provided to transfer data from ChromaDB to PostgreSQL:

```bash
python migrate_to_postgres.py \
  --chroma-dir /app/evaluation_storage_data \
  --postgres-url postgresql://postgres:postgres@postgres:5432/panopticon \
  --batch-size 100
```

## API Endpoints

The service exposes the following REST API endpoints:

### Store a new evaluation metric
```
POST /api/v1/evaluation-metrics
{
  "prompt": "How well does the response address the query?",
  "metric_type": "relevance",
  "metadata": {
    "category": "comprehension",
    "scale": "1-10"
  }
}
```

### Get evaluation metrics by type
```
GET /api/v1/evaluation-metrics/type/{metric_type}?page=1&limit=10
```

### Get a specific evaluation metric by ID
```
GET /api/v1/evaluation-metrics/{metric_id}
```

### Search for semantically similar metrics
```
POST /api/v1/evaluation-metrics/search?prompt=How%20well&limit=5
```

## Performance Considerations

- PostgreSQL with pgvector provides better performance for larger datasets compared to ChromaDB
- The index uses IVFFLAT for efficient vector similarity search
- Both implementations use the same embedding model (all-MiniLM-L6-v2) with 384 dimensions
