#!/bin/bash
# setup-bind-mounts.sh - Create host directories for Docker bind mounts
#
# This script sets up the directory structure needed for persistent storage
# with Docker bind mounts.

set -e  # Exit immediately if any command exits with non-zero status

# Color definitions for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up bind mount directories for Panopticon...${NC}"

# Create directories if they don't exist
echo -e "${YELLOW}Creating host directories if they don't exist...${NC}"
directories=(
    "./data/query_storage"
    "./data/evaluation_storage" 
    "./data/judge_service"
    "./data/postgres"
    "./data/model_registry"
)

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓ Directory $dir already exists${NC}"
    else
        echo -e "Creating directory $dir..."
        mkdir -p "$dir"
        echo -e "${GREEN}✓ Directory $dir created${NC}"
    fi
done

# Set proper permissions for PostgreSQL
echo -e "${YELLOW}Setting PostgreSQL directory permissions...${NC}"
chmod 777 ./data/postgres  # This wide permission is for demonstration; adjust as needed

echo -e "\n${BLUE}All directories created successfully!${NC}"
echo -e "\n${YELLOW}IMPORTANT:${NC} Usage guidelines:"
echo -e "  1. Start your containers with: ${GREEN}docker-compose up -d${NC}"
echo -e "  2. Data will persist in the ./data directory"
echo -e "  3. For database issues, you can directly inspect files in ./data directories"
echo -e "  4. Back up your data by copying the ./data directory"
