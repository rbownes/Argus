"""
Main application for the Panopticon system.
"""
import os
from typing import Dict

from shared.utils import create_api_app, ApiResponse, ResponseStatus
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
                {"name": "Query Storage", "url": "/api/queries"},
                {"name": "Evaluation Storage", "url": "/api/evaluations"},
                {"name": "Judge Service", "url": "/api/judge"}
            ]
        }
    )

@app.get("/api/services")
async def list_services():
    """List all available services."""
    services = [
        {
            "name": "Query Storage",
            "description": "Store and retrieve queries",
            "base_url": os.environ.get("QUERY_STORAGE_URL", "http://query-storage:8000"),
            "endpoints": [
                {"path": "/queries", "method": "POST", "description": "Store a query"},
                {"path": "/queries/theme/{theme}", "method": "GET", "description": "Get queries by theme"},
                {"path": "/queries/search", "method": "POST", "description": "Search similar queries"}
            ]
        },
        {
            "name": "Evaluation Storage",
            "description": "Store and retrieve evaluation metrics",
            "base_url": os.environ.get("EVALUATION_STORAGE_URL", "http://evaluation-storage:8000"),
            "endpoints": [
                {"path": "/evaluation-metrics", "method": "POST", "description": "Store an evaluation metric"},
                {"path": "/evaluation-metrics/type/{metric_type}", "method": "GET", "description": "Get metrics by type"},
                {"path": "/evaluation-metrics/search", "method": "POST", "description": "Search similar metrics"}
            ]
        },
        {
            "name": "Judge Service",
            "description": "Evaluate language model outputs",
            "base_url": os.environ.get("JUDGE_SERVICE_URL", "http://judge-service:8000"),
            "endpoints": [
                {"path": "/evaluate/query", "method": "POST", "description": "Evaluate a single query"},
                {"path": "/evaluate/theme", "method": "POST", "description": "Evaluate all queries of a theme"},
                {"path": "/results", "method": "GET", "description": "Get evaluation results"}
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
