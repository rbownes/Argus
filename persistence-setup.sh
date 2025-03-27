#!/bin/bash
# persistence-setup.sh - Setup external volumes for Docker containers persistence
#
# This script creates external Docker volumes to ensure data persistence
# when containers are restarted or when running docker-compose down.

set -e  # Exit immediately if any command exits with non-zero status

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up persistent external volumes for Panopticon...${NC}"

# Create external volumes if they don't exist
echo -e "${YELLOW}Creating external volumes if they don't exist...${NC}"
volumes=(
    "panopticon_query_data"
    "panopticon_evaluation_data" 
    "panopticon_judge_data"
    "panopticon_postgres_data"
    "panopticon_model_registry_data"
)

for volume in "${volumes[@]}"; do
    if docker volume inspect "$volume" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Volume $volume already exists${NC}"
    else
        echo -e "Creating volume $volume..."
        docker volume create "$volume"
        echo -e "${GREEN}✓ Volume $volume created${NC}"
    fi
done

echo -e "\n${BLUE}All volumes created successfully!${NC}"
echo -e "\n${YELLOW}IMPORTANT:${NC} To maintain data persistence, follow these guidelines:"
echo -e "  1. Use ${GREEN}docker-compose down${NC} without the ${RED}--volumes${NC} flag to stop containers"
echo -e "  2. When starting containers, use ${GREEN}docker-compose up -d${NC}"
echo -e "  3. To ensure proper database shutdown, use ${GREEN}docker-compose stop -t 60${NC}"
echo -e "     This provides a 60-second grace period for clean database shutdown"
echo -e "\nYou can now start your containers with: ${GREEN}docker-compose up -d${NC}"
