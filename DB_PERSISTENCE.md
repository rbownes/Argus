# Database Persistence Solution for Panopticon

This document explains how to maintain database persistence for the Panopticon system when Docker containers are restarted or shut down.

## Problem

By default, Docker's volumes may not persist data when containers are removed, especially when the `docker-compose down` command is used with the `--volumes` flag. This results in data loss for:

- PostgreSQL database
- ChromaDB storage in the Judge Service
- Query Storage data
- Evaluation Storage data
- Model Registry data

## Solution

The solution implements external named volumes that persist regardless of container lifecycle, as well as proper container shutdown procedures to ensure database consistency.

## Components

1. **Modified docker-compose.yml**: 
   - Updated to use external named volumes that persist independently of the Docker Compose lifecycle
   
2. **persistence-setup.sh**:
   - Script that creates the necessary external volumes
   - Provides instructions for proper container shutdown/startup

## Setup Instructions

1. **Create External Volumes**

   Run the persistence-setup script to create all required external volumes:

   ```bash
   ./persistence-setup.sh
   ```

2. **Start Services**

   Start your Docker containers normally:

   ```bash
   docker-compose up -d
   ```

## Usage Guidelines

To ensure data persistence, follow these guidelines:

### ✅ DO:

- Use `docker-compose down` without the `--volumes` flag to stop services
- Use `docker-compose stop -t 60` to give databases time for clean shutdown
- Use `docker-compose up -d` to start services with persistent data

### ❌ DON'T:

- Use `docker-compose down --volumes` as this will remove all volume data
- Force-kill container processes without allowing proper shutdown

## Volumes Created

The solution creates these external volumes:

- `panopticon_postgres_data`: PostgreSQL database files
- `panopticon_judge_data`: Judge service storage (ChromaDB)
- `panopticon_query_data`: Query storage data
- `panopticon_evaluation_data`: Evaluation storage data
- `panopticon_model_registry_data`: Model registry data

## Troubleshooting

**Issue**: Data still not persisting after implementation
- Ensure you're not using `--volumes` flag with `docker-compose down`
- Check volume permissions with `docker volume inspect panopticon_postgres_data`
- Verify database connection strings point to correct persistent paths

**Issue**: Slow container startup
- This is normal - databases loading from persistent storage may take longer on initial startup

## Advanced Configuration

For production environments, consider:

1. **Backup automation**:
   ```bash
   docker run --rm -v panopticon_postgres_data:/data -v /backup:/backup ubuntu tar -czvf /backup/postgres-backup-$(date +%Y%m%d).tar.gz /data
   ```

2. **Volume driver options** for specialized storage requirements:
   ```yaml
   volumes:
     postgres_data:
       external: true
       name: panopticon_postgres_data
       driver_opts:
         type: nfs
         o: addr=192.168.1.1,rw
         device: ":/path/to/nfs/share"
   ```

## Further Recommendations

For mission-critical deployments:
- Consider managed database services instead of containerized databases
- Implement regular database dumps to offline storage
- Use database replication for high availability
