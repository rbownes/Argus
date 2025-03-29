"""
Main application for the Panopticon system.
"""
import os
from typing import Dict, Any, Optional
import logging
import httpx
from fastapi import Request, Response, HTTPException
from fastapi.responses import StreamingResponse
import asyncio
from starlette.routing import Route
from starlette.responses import JSONResponse, Response as StarletteResponse

from shared.utils import create_api_app, ApiResponse, ResponseStatus, setup_logging

# Call logging setup early
setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"))
from shared.config import ServiceConfig
from shared.middleware import add_middleware

# Create FastAPI app with explicit redirect configuration
import fastapi
from fastapi import FastAPI
from starlette.routing import Route, Router

# Instead of using create_api_app, we'll create the app directly with disable_redirects
app = FastAPI(
    title="Panopticon",
    description="Evaluation system for language models",
    version="0.1.0", 
    redirect_slashes=False,
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
)

# CRUCIAL: Force override of the internal router's redirect behavior
# This is needed because FastAPI's higher-level API still tries to redirect
from starlette.routing import Match
app.router.redirect_slashes = False

# Create a direct route handler for the query endpoint
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.routing import APIRoute, Mount
import re

# Custom route class to ensure no trailing slash redirects
class NoRedirectRoute(APIRoute):
    def matches(self, scope):
        if scope["type"] != "http":
            return False
        path = scope["path"]
        
        # Handle both paths with and without trailing slash
        path_no_slash = path.rstrip('/')
        path_with_slash = f"{path_no_slash}/"
        
        # Log path for debugging
        logging.info(f"Matching route: {path} against {self.path}")
        
        # Create a pattern that matches both versions
        both_patterns = f"^({re.escape(path_no_slash)}|{re.escape(path_with_slash)})$"
        match = re.match(both_patterns, self.path)
        
        if match:
            return True
            
        return super().matches(scope)

# Ensure middleware order - this one must come first
@app.middleware("http")
async def no_redirects_middleware(request, call_next):
    """Middleware to prevent redirects by intercepting them before they happen"""
    # Run this before any other handling to fix path issues
    path = request.url.path
    
    # Log original path
    logging.info(f"Original request path: {path}")
    
    # For POST requests to /api/queries with or without trailing slash
    if request.method == "POST" and (path == "/api/queries" or path == "/api/queries/"):
        logging.warning(f"DIRECT INTERCEPT of queries POST: {request.url.path}")
        
        # Get request body
        body = await request.body()
        
        # Directly proxy to item storage
        url = f"{QUERY_STORAGE_URL}/api/v1/items"
        logging.info(f"Direct manual proxy to: {url}")
        
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        
        # Create a dedicated client for this request
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    url=url,
                    headers=headers, 
                    content=body
                )
                
                logging.info(f"DIRECT RESPONSE: status={response.status_code}")
                
                # Return the response directly
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get("content-type")
                )
            except Exception as e:
                logging.error(f"Error in middleware handler: {str(e)}", exc_info=True)
                return JSONResponse(
                    content={"detail": f"Error forwarding request: {str(e)}"},
                    status_code=500
                )

    # Handle any redirects that might still happen
    response = await call_next(request)
    if response.status_code in (307, 308):
        logging.warning(f"BLOCKING REDIRECT: {request.url} -> {response.headers.get('location')}")
        
        # Debug information
        if request.method == "POST" and "/api/queries" in request.url.path:
            logging.error(f"Still got redirected on queries endpoint: {request.url.path}")
        
        # Return an error instead
        return JSONResponse(
            content={"detail": "Redirects disabled, fix your implementation"},
            status_code=400
        )
    
    return response

# Add middleware
config = ServiceConfig.from_env("panopticon-app")
add_middleware(app, api_key=os.environ.get("API_KEY"))

# Service URLs
QUERY_STORAGE_URL = os.environ.get("QUERY_STORAGE_URL", "http://item-storage-queries:8000")
METRICS_STORAGE_URL = os.environ.get("EVALUATION_STORAGE_URL", "http://item-storage-metrics:8000")
JUDGE_SERVICE_URL = os.environ.get("JUDGE_SERVICE_URL", "http://judge-service:8000")
MODEL_REGISTRY_URL = os.environ.get("MODEL_REGISTRY_URL", "http://model-registry:8000")
VISUALIZATION_URL = os.environ.get("VISUALIZATION_URL", "http://visualization:8000")

# Create HTTP client to forward requests
http_client = httpx.AsyncClient(timeout=60.0)

# Create a direct route handler function for the queries endpoint
# This bypasses FastAPI's routing system for just this endpoint
async def handle_query_post(request):
    """Low-level handler for query POST requests that bypasses regular FastAPI routing"""
    logging.info(f"Direct handler called for path: {request.url.path}")
    
    # Get request body
    body = await request.body()
    
    # Target the correct endpoint - items not queries
    url = f"{QUERY_STORAGE_URL}/api/v1/items"
    logging.info(f"Making direct POST request to: {url}")
    
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    
    try:
        # Make explicit request directly to items endpoint with expanded logging
        logging.info(f"Sending POST to {url} with headers: {headers}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=url,
                headers=headers, 
                content=body
            )
        
        logging.info(f"Response received - status: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        
        # Create a raw response
        return StarletteResponse(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type")
        )
    except Exception as e:
        logging.error(f"Error in direct handler: {str(e)}")
        return JSONResponse({"detail": str(e)}, status_code=500)

# ======================================================
# IMPORTANT: Starlette-level direct routes for /api/queries
# This bypasses FastAPI's router completely to avoid redirects
# ======================================================

# First, clear any existing POST routes for /api/queries
for route in list(app.routes):
    if (
        isinstance(route, fastapi.routing.APIRoute) and 
        route.path in ["/api/queries", "/api/queries/"] and
        "POST" in route.methods
    ):
        app.routes.remove(route)
        logging.info(f"Removed route: {route.path} {route.methods}")

# Create direct Starlette routes instead of FastAPI routes
async def direct_query_post_handler(request):
    """
    Direct Starlette handler without FastAPI wrapping - handles /api/queries POST
    """
    logging.info(f"DIRECT STARLETTE HANDLER: {request.url.path}")
    
    # Get request body
    body = await request.body()
    
    # Forward directly to the item storage service using the docker-compose port mapping
    # This is more reliable than using internal Docker network names
    url = "http://localhost:8001/api/v1/items"
    
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    
    try:
        # Forward the request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url=url,
                headers=headers,
                content=body
            )
        
        logging.info(f"Direct response: status={response.status_code}")
        
        # Return raw response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type")
        )
    except Exception as e:
        logging.error(f"Error in direct handler: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"detail": f"Error forwarding request: {str(e)}"},
            status_code=500
        )

# Add Starlette routes directly to the ASGI app
app.router.routes.extend([
    Route("/api/queries", direct_query_post_handler, methods=["POST"]),
    Route("/api/queries/", direct_query_post_handler, methods=["POST"]),
])

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
            "base_url": QUERY_STORAGE_URL,
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
            "base_url": METRICS_STORAGE_URL,
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
            "base_url": JUDGE_SERVICE_URL,
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
            "base_url": MODEL_REGISTRY_URL,
            "endpoints": [
                {"path": "/api/v1/models", "method": "GET", "description": "List available models"},
                {"path": "/api/v1/providers", "method": "GET", "description": "List available providers"},
                {"path": "/api/v1/completion", "method": "POST", "description": "Generate completions using models"}
            ]
        },
        {
            "name": "Visualization Service",
            "description": "Dashboard and visualization for model evaluations",
            "base_url": VISUALIZATION_URL,
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

# Proxy functions to forward requests to microservices
async def proxy_request(request: Request, target_url: str, endpoint_override: str = None) -> Response:
    """
    Forward a request to a target service and return the response.

    Args:
        request: Original request
        target_url: URL to forward the request to
        endpoint_override: Optional specific endpoint to use instead of derived path

    Returns:
        Response from the target service
    """
    if endpoint_override:
        # Use override directly if provided
        service_path = endpoint_override
        logging.info(f"Using endpoint override: '{service_path}'")
    else:
        # Calculate path relative to the service base (/api/<service_name>/)
        path_parts = request.url.path.split("/")
        if len(path_parts) > 3: # Check if there's anything after /api/<service_name>/
            # Correctly skip '', 'api', '<service_name>'
            service_path = "/".join(path_parts[3:])
            logging.info(f"Calculated service path: '{service_path}' from {request.url.path}")
        else:
            # Handle root case, e.g., /api/judge -> proxies to /api/v1/
            # Check if target service handles requests to /api/v1/
            # judge_service does not have a root handler for /api/v1/, so this will likely 404 at the target.
            service_path = ""
            logging.info(f"Handling root request for {request.url.path}, setting service_path to empty.")

    # Build the final target URL
    # Ensure no double slashes if service_path is empty
    if service_path:
        url = f"{target_url}/api/v1/{service_path}"
    else:
        url = f"{target_url}/api/v1" # Target the base v1 path

    # Add query parameters from original request to the target URL
    if request.query_params:
        url += "?" + str(request.query_params)

    logging.info(f"Proxying {request.method} {request.url.path} to -> {url}") # Log the constructed URL

    # Get request method
    method = request.method

    # Get headers, excluding host
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    try:
        if method == "GET":
            response = await http_client.get(url, headers=headers) # Pass params in URL
        elif method == "POST":
            body = await request.body()
            response = await http_client.post(url, headers=headers, content=body) # Pass params in URL
        elif method == "PUT":
            body = await request.body()
            response = await http_client.put(url, headers=headers, content=body) # Pass params in URL
        elif method == "DELETE":
            response = await http_client.delete(url, headers=headers) # Pass params in URL
        else:
            raise HTTPException(status_code=405, detail=f"Method {method} not allowed for proxy")

        # Return response
        logging.info(f"Proxy response status: {response.status_code} from {url}")
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type")
        )
    except httpx.RequestError as e:
        logging.error(f"Error forwarding request to {url}: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Error communicating with service: {str(e)}")
    except httpx.HTTPStatusError as e: # Catch specific HTTP errors from the target
        logging.error(f"Target service returned error {e.response.status_code} from {url}: {e.response.text}")
        # Return the actual error from the target service
        return Response(
            content=e.response.content,
            status_code=e.response.status_code,
            headers=dict(e.response.headers),
            media_type=e.response.headers.get("content-type")
        )

@app.get("/api/queries")
async def proxy_query_storage_get(request: Request):
    """Proxy GET requests to Query Storage service."""
    logging.info(f"Proxying GET request to items endpoint with URL: {QUERY_STORAGE_URL}/api/v1/items")
    return await proxy_request(request, QUERY_STORAGE_URL, "items")

@app.api_route("/api/queries/search", methods=["POST"])
async def proxy_query_storage_search(request: Request):
    """Proxy requests to Query Storage search endpoint."""
    return await proxy_request(request, QUERY_STORAGE_URL, "items/search")

@app.api_route("/api/queries/type/{item_type}", methods=["GET"])
async def proxy_query_storage_by_type(request: Request, item_type: str):
    """Proxy requests to Query Storage by type."""
    return await proxy_request(request, QUERY_STORAGE_URL, f"items/type/{item_type}")

@app.api_route("/api/queries/{item_id}", methods=["GET", "PUT", "DELETE"])
async def proxy_query_storage_by_id(request: Request, item_id: str):
    """Proxy requests to Query Storage by ID."""
    return await proxy_request(request, QUERY_STORAGE_URL, f"items/{item_id}")

# API Routes for Metrics Storage
@app.api_route("/api/metrics", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_metrics_storage_root(request: Request):
    """Proxy requests to Metrics Storage service root."""
    # For root metrics endpoint, map to 'items' endpoint in the storage service
    return await proxy_request(request, METRICS_STORAGE_URL, "items")

@app.api_route("/api/metrics/search", methods=["POST"])
async def proxy_metrics_storage_search(request: Request):
    """Proxy requests to Metrics Storage search endpoint."""
    return await proxy_request(request, METRICS_STORAGE_URL, "items/search")

@app.api_route("/api/metrics/type/{item_type}", methods=["GET"])
async def proxy_metrics_storage_by_type(request: Request, item_type: str):
    """Proxy requests to Metrics Storage by type."""
    return await proxy_request(request, METRICS_STORAGE_URL, f"items/type/{item_type}")

@app.api_route("/api/metrics/{item_id}", methods=["GET", "PUT", "DELETE"])
async def proxy_metrics_storage_by_id(request: Request, item_id: str):
    """Proxy requests to Metrics Storage by ID."""
    return await proxy_request(request, METRICS_STORAGE_URL, f"items/{item_id}")

# API Routes for Judge Service
@app.api_route("/api/judge", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_judge_service_root(request: Request):
    """Proxy requests to Judge Service root."""
    return await proxy_request(request, JUDGE_SERVICE_URL)

@app.api_route("/api/judge/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_judge_service(request: Request, path: str):
    """Proxy requests to Judge Service with path."""
    return await proxy_request(request, JUDGE_SERVICE_URL)

# API Routes for Model Registry
@app.api_route("/api/models", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_model_registry_root(request: Request):
    """Proxy requests to Model Registry service root."""
    return await proxy_request(request, MODEL_REGISTRY_URL)

@app.api_route("/api/models/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_model_registry(request: Request, path: str):
    """Proxy requests to Model Registry service with path."""
    return await proxy_request(request, MODEL_REGISTRY_URL)

# API Routes for Visualization Service
@app.api_route("/api/dashboard", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_visualization_root(request: Request):
    """Proxy requests to Visualization Service root."""
    return await proxy_request(request, VISUALIZATION_URL)

@app.api_route("/api/dashboard/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_visualization(request: Request, path: str):
    """Proxy requests to Visualization Service with path."""
    return await proxy_request(request, VISUALIZATION_URL)

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "main-app"}

if __name__ == "__main__":
    """Run the application using uvicorn when executed directly."""
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
