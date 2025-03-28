"""
Main application for the Panopticon system.
"""
import os
from typing import Dict
import logging

from shared.utils import create_api_app, ApiResponse, ResponseStatus, setup_logging

# Call logging setup early
setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"))
from shared.config import ServiceConfig
from shared.middleware import add_middleware

# Create FastAPI app
app = create_api_app(
    title="Panopticon",
    description="Evaluation system for language models",
    version="0.1.0"
)

# Add middleware
config = ServiceConfig.from_env("panopticon-app")
add_middleware(app, api_key=os.environ.get("API_KEY"))

@app.get("/")
async def root():
    """Root endpoint that returns basic service information."""
    return ApiResponse(
        status=ResponseStatus.SUCCESS,
        message="Panopticon API is running",
        data={
            "name": "Panopticon LLM Evaluation System",
            "version": "0.1.0",
            "services": [
                {"name": "Item Storage - Queries", "url": "/api/queries"},
                {"name": "Item Storage - Metrics", "url": "/api/metrics"},
                {"name": "Judge Service", "url": "/api/judge"},
                {"name": "Model Registry", "url": "/api/models"},
                {"name": "Visualization Service", "url": "/api/dashboard"}
            ]
        }
    )

@app.get("/api/services")
async def list_services():
    """List all available services."""
    services = [
        {
            "name": "Item Storage - Queries",
            "description": "Store and retrieve query data",
            "base_url": os.environ.get("QUERY_STORAGE_URL", "http://item-storage-queries:8000"),
            "endpoints": [
                {"path": "/api/v1/items", "method": "POST", "description": "Store a query"},
                {"path": "/api/v1/items/type/{item_type}", "method": "GET", "description": "Get queries by theme"},
                {"path": "/api/v1/items/search", "method": "POST", "description": "Search similar queries"},
                {"path": "/api/v1/items/{item_id}", "method": "GET", "description": "Get query by ID"}
            ]
        },
        {
            "name": "Item Storage - Metrics",
            "description": "Store and retrieve evaluation metrics",
            "base_url": os.environ.get("EVALUATION_STORAGE_URL", "http://item-storage-metrics:8000"),
            "endpoints": [
                {"path": "/api/v1/items", "method": "POST", "description": "Store an evaluation metric"},
                {"path": "/api/v1/items/type/{item_type}", "method": "GET", "description": "Get metrics by type"},
                {"path": "/api/v1/items/search", "method": "POST", "description": "Search similar metrics"},
                {"path": "/api/v1/items/{item_id}", "method": "GET", "description": "Get metric by ID"}
            ]
        },
        {
            "name": "Judge Service",
            "description": "Evaluate language model outputs",
            "base_url": os.environ.get("JUDGE_SERVICE_URL", "http://judge-service:8000"),
            "endpoints": [
                {"path": "/api/v1/evaluate/query", "method": "POST", "description": "Evaluate a single query"},
                {"path": "/api/v1/evaluate/theme", "method": "POST", "description": "Evaluate all queries of a theme"},
                {"path": "/api/v1/results", "method": "GET", "description": "Get evaluation results"},
                {"path": "/api/v1/models", "method": "GET", "description": "List available LLM models"}
            ]
        },
        {
            "name": "Model Registry",
            "description": "Registry for LLM models and providers",
            "base_url": os.environ.get("MODEL_REGISTRY_URL", "http://model-registry:8000"),
            "endpoints": [
                {"path": "/api/v1/models", "method": "GET", "description": "List available models"},
                {"path": "/api/v1/providers", "method": "GET", "description": "List available providers"},
                {"path": "/api/v1/completion", "method": "POST", "description": "Generate completions using models"}
            ]
        },
        {
            "name": "Visualization Service",
            "description": "Dashboard and visualization for model evaluations",
            "base_url": os.environ.get("VISUALIZATION_URL", "http://visualization:8000"),
            "endpoints": [
                {"path": "/api/v1/dashboard/summary", "method": "GET", "description": "Get dashboard summary"},
                {"path": "/api/v1/dashboard/results", "method": "GET", "description": "Get detailed results"},
                {"path": "/api/v1/dashboard/models", "method": "GET", "description": "Get model comparison data"}
            ]
        }
    ]
    
    return ApiResponse(
        status=ResponseStatus.SUCCESS,
        data=services
    )

if __name__ == "__main__":
    """Run the application using uvicorn when executed directly."""
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
