# Query Storage Service

This service provides storage and retrieval of query data with semantic search capabilities.

## Storage Implementations

The service supports two storage implementations:

1. **ChromaDB** (default): Vector database for semantic search capabilities
2. **PostgreSQL with pgvector**: Uses PostgreSQL with the pgvector extension for more reliable persistence

## Storage Configuration

You can configure which storage implementation to use via environment variables:

```
QUERY_STORAGE_TYPE=postgres  # Use 'postgres' for pgvector, 'chroma' (default) for ChromaDB
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/panopticon  # Required for PostgreSQL
```

## Fixing ChromaDB Persistence Issues

### Problem

ChromaDB was configured to use a persisted storage directory (`/app/query_storage_data`), but was not actually writing files to disk. This was because:

1. The `is_persistent` flag was not explicitly set to `True`
2. The `client.persist()` method was not being called after adding data

### Solution

The `query_storage.py` file has been updated to:

1. Add the `is_persistent=True` setting to the ChromaDB client configuration
2. Explicitly call `self.client.persist()` after adding data to ensure it's written to disk

## Migrating to pgvector

If ChromaDB persistence issues continue, follow these steps to migrate to PostgreSQL with pgvector:

### Prerequisites

1. PostgreSQL with the pgvector extension installed
2. The necessary tables created in the database (handled automatically by `pg_query_storage.py`)

### Migration Steps

1. **Update the environment variables** in your `.env` file or Docker Compose configuration:

   ```
   # Add these environment variables to the query-storage service
   QUERY_STORAGE_TYPE=postgres
   DATABASE_URL=postgresql://postgres:postgres@postgres:5432/panopticon
   ```

2. **Rebuild and restart the query-storage service**:

   ```bash
   docker-compose build query-storage
   docker-compose up -d query-storage
   ```

3. **Run the migration script** to transfer existing data from ChromaDB to PostgreSQL:

   ```bash
   # Execute inside the container
   docker exec -it panopticon-query-storage-1 python -m query_storage.migrate_to_postgres \
     --chroma-dir=/app/query_storage_data \
     --db-url=postgresql://postgres:postgres@postgres:5432/panopticon
   ```

### Verifying Migration

After migration, you can verify that queries are now being stored in PostgreSQL by:

1. Making a test query:

   ```bash
   curl -X POST http://localhost:8001/api/v1/queries \
     -H "Content-Type: application/json" \
     -H "X-API-Key: dev_api_key_for_testing" \
     -d '{
       "query": "Test query after migration",
       "theme": "test_theme",
       "metadata": {
         "test": true
       }
     }'
   ```

2. Checking the PostgreSQL database:

   ```bash
   docker exec -it panopticon-postgres-1 psql -U postgres -d panopticon -c "SELECT * FROM queries LIMIT 5;"
   ```

## Advantages of pgvector

- **Reliable persistence**: Data is stored directly in the PostgreSQL database
- **Simpler architecture**: No separate vector database to manage
- **Mature technology**: PostgreSQL offers robust data management features
- **Consistent backups**: Database backups include your vector data
- **Transactional integrity**: ACID compliance for reliable data operations
