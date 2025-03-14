#!/bin/bash
# Setup script for LLM Evaluation System

set -e

echo "Setting up LLM Evaluation System..."

# Create necessary directories
mkdir -p data/chroma_db

# Check if Python virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if PostgreSQL is running
if command -v pg_isready > /dev/null; then
    if pg_isready -q; then
        echo "PostgreSQL is running"
        
        # Check if database exists
        if psql -lqt | cut -d \| -f 1 | grep -qw llm_eval; then
            echo "Database 'llm_eval' already exists"
        else
            echo "Creating database 'llm_eval'..."
            createdb llm_eval || echo "Failed to create database. You may need to create it manually."
            
            echo "Initializing database schema..."
            psql -d llm_eval -f schema.sql
        fi
    else
        echo "PostgreSQL is not running. Please start PostgreSQL and run this script again."
        echo "Alternatively, you can use Docker Compose to set up the entire system."
    fi
else
    echo "PostgreSQL command line tools not found."
    echo "You can use Docker Compose to set up the entire system."
fi

# Set up environment variables
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
POSTGRES_URL=postgresql://llm_eval:llm_eval_password@localhost:5432/llm_eval
CHROMA_PATH=./data/chroma_db
# OPENAI_API_KEY=your-openai-api-key
# ANTHROPIC_API_KEY=your-anthropic-api-key
EOL
    echo "Please edit .env file with your API keys"
fi

echo "Setup completed!"
echo ""
echo "To start the API server:"
echo "source .venv/bin/activate"
echo "uvicorn llm_eval.services.api_service.app:app --reload"
echo ""
echo "To run using Docker Compose:"
echo "docker-compose up -d"
echo ""
echo "To run the example evaluation script:"
echo "source .venv/bin/activate"
echo "python examples/run_evaluation.py"
