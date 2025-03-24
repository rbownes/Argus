"""
API for the Evaluation Storage service.
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
from .evaluation_storage import EvaluationStorage

# Create FastAPI app
app = create_api_app(
    title="Evaluation Storage",
    description="Store and retrieve evaluation metrics for the Panopticon system",
    version="0.1.0"
)

# Initialize configuration
config = ServiceConfig.from_env("evaluation-storage")
add_middleware(app, api_key=os.environ.get("API_KEY"))

# Initialize storage
storage = EvaluationStorage()

# API Models
class EvaluationMetricRequest(BaseModel):
    """Request model for storing an evaluation metric."""
    prompt: str = Field(..., description="The evaluation prompt text")
    metric_type: str = Field(..., description="Type or category of the evaluation metric")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata for the metric")

class EvaluationMetricResponse(BaseModel):
    """Response model for evaluation metric data."""
    id: str = Field(..., description="Unique identifier for the evaluation metric")
    prompt: str = Field(..., description="The evaluation prompt text")
    metadata: Dict = Field(..., description="Metadata including metric type and other attributes")

# API Routes
@app.post("/api/v1/evaluation-metrics", response_model=ApiResponse, tags=["Evaluation Metrics"])
async def store_evaluation_metric(request: EvaluationMetricRequest):
    """
    Store a new evaluation metric with its type and metadata.
    
    Returns the created metric with its ID.
    """
    try:
        metric_id = storage.store_evaluation_metric(
            prompt=request.prompt,
            metric_type=request.metric_type,
            metadata=request.metadata
        )
        
        # Fetch the stored metric to return
        results = storage.get_metrics_by_type(request.metric_type, limit=1)
        if not results:
            raise ApiError(status_code=500, message="Failed to store evaluation metric")
            
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            message="Evaluation metric stored successfully",
            data=results[0]
        )
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

@app.get("/api/v1/evaluation-metrics/type/{metric_type}", response_model=ApiResponse, tags=["Evaluation Metrics"])
async def get_metrics_by_type(
    metric_type: str, 
    pagination: PaginationParams = Depends()
):
    """
    Retrieve evaluation metrics by type with pagination.
    
    Returns a list of metrics matching the specified type.
    """
    try:
        results = storage.get_metrics_by_type(
            metric_type=metric_type, 
            limit=pagination.limit, 
            skip=pagination.get_skip()
        )
        
        # Get total count for pagination
        total_count = storage.count_metrics_by_type(metric_type)
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=paginate_results(results, pagination, total_count)
        )
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

@app.post("/api/v1/evaluation-metrics/search", response_model=ApiResponse, tags=["Evaluation Metrics"])
async def search_similar_metrics(
    prompt: str = Query(..., description="Prompt text to search for"),
    limit: int = Query(5, ge=1, le=100, description="Maximum number of results to return")
):
    """
    Search for semantically similar evaluation metrics.
    
    Returns a list of metrics that are semantically similar to the input prompt.
    """
    try:
        results = storage.search_similar_metrics(prompt, limit)
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=results
        )
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

# For backward compatibility, keep the original routes but direct to the new implementations
@app.post("/evaluation-metrics", response_model=EvaluationMetricResponse, tags=["Legacy"])
async def store_evaluation_metric_legacy(request: EvaluationMetricRequest):
    """Legacy endpoint for storing an evaluation metric."""
    response = await store_evaluation_metric(request)
    return response.data

@app.get("/evaluation-metrics/type/{metric_type}", response_model=List[EvaluationMetricResponse], tags=["Legacy"])
async def get_metrics_by_type_legacy(
    metric_type: str, 
    limit: int = Query(10, ge=1, le=100)
):
    """Legacy endpoint for retrieving evaluation metrics by type."""
    response = await get_metrics_by_type(
        metric_type=metric_type, 
        pagination=PaginationParams(page=1, limit=limit)
    )
    return response.data["items"]

@app.post("/evaluation-metrics/search", response_model=List[EvaluationMetricResponse], tags=["Legacy"])
async def search_similar_metrics_legacy(
    prompt: str = Query(...),
    limit: int = Query(5, ge=1, le=100)
):
    """Legacy endpoint for searching similar metrics."""
    response = await search_similar_metrics(prompt=prompt, limit=limit)
    return response.data
