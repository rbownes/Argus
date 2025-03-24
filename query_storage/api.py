"""
API for the Query Storage service.
"""
from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import os

from shared.utils import (
    create_api_app, 
    ApiResponse, 
    ResponseStatus, 
    ApiError, 
    PaginationParams,
    paginate_results
)
from shared.config import ServiceConfig
from shared.middleware import add_middleware
from .query_storage import QueryStorage

# Create FastAPI app
app = create_api_app(
    title="Query Storage",
    description="Store and retrieve query data for the Panopticon system",
    version="0.1.0"
)

# Initialize configuration
config = ServiceConfig.from_env("query-storage")
add_middleware(app, api_key=os.environ.get("API_KEY"))

# Initialize storage
storage = QueryStorage()

# API Models
class QueryRequest(BaseModel):
    """Request model for storing a query."""
    query: str = Field(..., description="The query text")
    theme: str = Field(..., description="Theme or category of the query")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata for the query")

class QueryResponse(BaseModel):
    """Response model for query data."""
    id: str = Field(..., description="Unique identifier for the query")
    query: str = Field(..., description="The query text")
    metadata: Dict = Field(..., description="Metadata including theme and other attributes")

# API Routes
@app.post("/api/v1/queries", response_model=ApiResponse, tags=["Queries"])
async def store_query(request: QueryRequest):
    """
    Store a new query with its theme and metadata.
    
    Returns the created query with its ID.
    """
    try:
        query_id = storage.store_query(
            query=request.query,
            theme=request.theme,
            metadata=request.metadata
        )
        
        # Fetch the stored query to return
        results = storage.get_queries_by_theme(request.theme, limit=1)
        if not results:
            raise ApiError(status_code=500, message="Failed to store query")
            
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            message="Query stored successfully",
            data=results[0]
        )
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

@app.get("/api/v1/queries/theme/{theme}", response_model=ApiResponse, tags=["Queries"])
async def get_queries_by_theme(
    theme: str, 
    pagination: PaginationParams = Depends()
):
    """
    Retrieve queries by theme with pagination.
    
    Returns a list of queries matching the specified theme.
    """
    try:
        results = storage.get_queries_by_theme(
            theme=theme, 
            limit=pagination.limit, 
            skip=pagination.get_skip()
        )
        
        # Get total count for pagination
        total_count = storage.count_queries_by_theme(theme)
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=paginate_results(results, pagination, total_count)
        )
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

@app.post("/api/v1/queries/search", response_model=ApiResponse, tags=["Queries"])
async def search_similar_queries(
    query: str = Query(..., description="Query text to search for"),
    limit: int = Query(5, ge=1, le=100, description="Maximum number of results to return")
):
    """
    Search for semantically similar queries.
    
    Returns a list of queries that are semantically similar to the input query.
    """
    try:
        results = storage.search_similar_queries(query, limit)
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=results
        )
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

# For backward compatibility, keep the original routes but direct to the new implementations
@app.post("/queries", response_model=QueryResponse, tags=["Legacy"])
async def store_query_legacy(request: QueryRequest):
    """Legacy endpoint for storing a query."""
    response = await store_query(request)
    return response.data

@app.get("/queries/theme/{theme}", response_model=List[QueryResponse], tags=["Legacy"])
async def get_queries_by_theme_legacy(
    theme: str, 
    limit: int = Query(10, ge=1, le=100)
):
    """Legacy endpoint for retrieving queries by theme."""
    response = await get_queries_by_theme(
        theme=theme, 
        pagination=PaginationParams(page=1, limit=limit)
    )
    return response.data["items"]

@app.post("/queries/search", response_model=List[QueryResponse], tags=["Legacy"])
async def search_similar_queries_legacy(
    query: str = Query(...),
    limit: int = Query(5, ge=1, le=100)
):
    """Legacy endpoint for searching similar queries."""
    response = await search_similar_queries(query=query, limit=limit)
    return response.data
