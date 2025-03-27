# PGVector Migration Project

This project implements PostgreSQL with pgvector extension as a replacement for ChromaDB across all embedding storage functions in the Panopticon system.

## Overview

The goal was to migrate all vector storage from ChromaDB to PostgreSQL with pgvector for improved performance, scalability, and reliability. This migration covers:

1. **Query Storage Service** (already implemented)
2. **Evaluation Storage Service** (newly implemented)
3. **Judge Service** (newly implemented)

## Architecture

Each service now has:

1. A PostgreSQL implementation that parallels the ChromaDB implementation
2. A factory pattern to select between implementations
3. Environment variables to control which backend is used
4. Migration scripts to transfer data from ChromaDB to PostgreSQL

## Configuration

Storage behavior is controlled via environment variables in `.env` or Docker Compose configuration:

```env
# Storage type selection
QUERY_STORAGE_TYPE=postgres      # or "chroma" for ChromaDB
EVALUATION_STORAGE_TYPE=postgres # or "chroma" for ChromaDB
JUDGE_STORAGE_TYPE=postgres      # or "chroma" for ChromaDB

# PostgreSQL connection (used by all services)
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/panopticon

# ChromaDB directories (if using ChromaDB)
CHROMA_PERSIST_DIRECTORY=/app/query_storage_data
EVALUATION_PERSIST_DIRECTORY=/app/evaluation_storage_data
JUDGE_PERSIST_DIRECTORY=/app/judge_service_data
```

## Services Affected

### Query Storage Service

Already implemented with PostgreSQL + pgvector.

### Evaluation Storage Service

- New implementation: `pg_evaluation_storage.py`
- Factory: `evaluation_storage/storage_factory.py`
- Migration script: `evaluation_storage/migrate_to_postgres.py`

### Judge Service

- New implementation: `pg_judge_storage.py`
- Factory: `judge_service/storage_factory.py`
- Migration script: `judge_service/migrate_to_postgres.py`

## Migration Process

To migrate all services to PostgreSQL:

1. **Ensure PostgreSQL with pgvector extension is installed**

2. **Update environment variables** in your .env file or docker-compose.yml:
   ```
   QUERY_STORAGE_TYPE=postgres
   EVALUATION_STORAGE_TYPE=postgres
   JUDGE_STORAGE_TYPE=postgres
   DATABASE_URL=postgresql://postgres:postgres@postgres:5432/panopticon
   ```

3. **Run the migration scripts** for each service:
   ```bash
   # Query Storage
   cd query_storage
   python migrate_to_postgres.py
   
   # Evaluation Storage
   cd evaluation_storage
   python migrate_to_postgres.py
   
   # Judge Service
   cd judge_service
   python migrate_to_postgres.py
   ```

4. **Restart the services** to apply the configuration changes:
   ```bash
   docker-compose restart
   ```

## Database Schema

Each service uses its own table with a similar schema pattern:

1. **Query Storage**: `queries` table
2. **Evaluation Storage**: `evaluation_metrics` table 
3. **Judge Service**: `llm_outputs` table for LLM outputs and `evaluation_results` table for results

All vector tables follow this general pattern:
```sql
CREATE TABLE table_name (
    id UUID PRIMARY KEY,
    text_field TEXT NOT NULL,
    embedding VECTOR(384),  -- 384-dimensional vector
    category_field TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_name
ON table_name
USING ivfflat (embedding vector_cosine_ops);
```

## Performance Considerations

- PostgreSQL with pgvector provides better performance for larger datasets compared to ChromaDB
- The index uses IVFFLAT for efficient vector similarity search
- All implementations use the same embedding model (all-MiniLM-L6-v2) with 384 dimensions
- Queries that involve both text filtering and vector similarity will perform faster in PostgreSQL

## Future Enhancements

Potential future enhancements include:

- Adding more sophisticated PostgreSQL indexes for common query patterns
- Implementing batch processing for better performance
- Automated monitoring of vector search performance
- Implementation of periodic VACUUM operations for database maintenance
