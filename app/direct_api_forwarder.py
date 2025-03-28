#!/usr/bin/env python3
"""
Direct API Forwarder - Simple standalone script to forward requests without FastAPI redirects
"""
import asyncio
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

# Configure this to match the item-storage-queries service as seen from the host
ITEM_STORAGE_URL = "http://localhost:8001"  # Port mapped in docker-compose.yml

async def forward_query_post(request):
    """Forward POST requests to the query storage service directly"""
    print(f"Received request at: {request.url.path}")
    
    try:
        # Get the request body
        body = await request.body()
        
        # Copy the headers
        headers = dict(request.headers)
        if "host" in headers:
            del headers["host"]
        
        print(f"Forwarding to: {ITEM_STORAGE_URL}/api/v1/items")
        print(f"Headers: {headers}")
        print(f"Body size: {len(body)}")
        
        # Forward the request directly to the item storage service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ITEM_STORAGE_URL}/api/v1/items",
                headers=headers,
                content=body,
                timeout=30.0
            )
        
        print(f"Response status: {response.status_code}")
        
        # Return the response from the item storage service
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type")
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse({"detail": f"Error forwarding request: {str(e)}"}, status_code=500)

async def health(request):
    """Simple health check endpoint"""
    return JSONResponse({"status": "healthy", "service": "direct-forwarder"})

# Create a simple Starlette app that only handles the API endpoints we need
routes = [
    Route("/api/queries", forward_query_post, methods=["POST"]),
    Route("/api/queries/", forward_query_post, methods=["POST"]),
    Route("/health", health, methods=["GET"]),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
