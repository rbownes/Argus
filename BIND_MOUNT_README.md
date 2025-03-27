# Docker Bind Mount Persistence Solution

This document explains how to use bind mounts to maintain database persistence for the Panopticon system when containers are restarted or shut down.

## Problem Solved

The previous Docker volume-based approach was not correctly persisting data between container restarts. We've now implemented a direct bind mount solution that:

1. Makes storage locations more transparent and accessible
2. Simplifies debugging and maintenance
3. Ensures proper data persistence across container lifecycle events

## Implementation Components

1. **Modified docker-compose.yml**:
   - Uses host directory bind mounts instead of named Docker volumes
   - Maps local `./data/*` directories directly into containers
   
2. **Updated storage service paths**:
   - All storage services now use absolute paths matching mount points
   - Ensures consistent data storage location references

3. **setup-bind-mounts.sh**:
   - Creates the necessary directory structure on the host
   - Sets appropriate permissions for container access

## Setup Instructions

1. **Create Host Directory Structure**

   Run the setup-bind-mounts script to create all required directories:

   ```bash
   ./setup-bind-mounts.sh
   ```

2. **Start Services**

   Start your Docker containers:

   ```bash
   docker-compose up -d
   ```

## Storage Structure

The implementation creates the following directory structure in your project root:

```
./data/
  ├── query_storage/         # ChromaDB data for query storage
  ├── evaluation_storage/    # ChromaDB data for evaluation storage
  ├── judge_service/         # ChromaDB data for judge service
  ├── postgres/              # PostgreSQL database files
  └── model_registry/        # Model registry data
```

## Advantages Over Previous Solution

- **Direct Access**: You can inspect data files directly on your host system
- **Simplified Backups**: Copy the data directory for instant backups
- **Better Transparency**: Easy to see where and how data is stored
- **Easier Debugging**: Direct access to database files for troubleshooting
- **No Docker Volume Management**: Avoids issues with Docker volume lifecycle

## Usage Guidelines

1. **Always shut down gracefully**:
   ```bash
   docker-compose stop -t 60
   ```
   This gives databases (especially PostgreSQL) time to clean up and flush data

2. **Backup regularly**:
   ```bash
   # Simple backup of all data
   cp -r ./data ./data-backup-$(date +%Y%m%d)
   ```

3. **Check permissions** if you encounter access issues:
   ```bash
   sudo chmod -R 777 ./data
   ```
   (Note: In production, use more restrictive permissions)

## Troubleshooting

If data persistence issues persist:

1. Check that your container user has write permissions to the mounted directories
2. Verify that the paths in code match the mount points in docker-compose.yml
3. Ensure you're not accidentally running with `docker-compose down --volumes`
4. Check logs for any file access errors
