"""
API for item storage service.
"""
from fastapi import Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import os
from shared.utils import (create_api_app, ApiResponse, ResponseStatus, ApiError,
                          PaginationParams, paginate_results)
from shared.middleware import add_middleware
from .storage_factory import get_item_storage  # Factory returns the correct instance

# Configure service based on environment variables
ITEM_TYPE_NAME = os.environ.get("ITEM_TYPE_NAME", "item")  # e.g., 'query' or 'metric'
ITEM_TYPE_CATEGORY_NAME = os.environ.get("ITEM_TYPE_CATEGORY_NAME", "type")  # e.g., 'theme' or 'metric_type'

app = create_api_app(
    title=f"{ITEM_TYPE_NAME.capitalize()} Storage",
    description=f"Store and retrieve {ITEM_TYPE_NAME} data",
    version="0.1.0"
)
add_middleware(app, api_key=os.environ.get("API_KEY"))
storage = get_item_storage()

class ItemRequest(BaseModel):
    item: str = Field(..., description=f"The {ITEM_TYPE_NAME} text")
    type: str = Field(..., description=f"{ITEM_TYPE_CATEGORY_NAME.capitalize()} or category of the {ITEM_TYPE_NAME}")
    metadata: Optional[Dict] = Field(default=None)

class ItemResponse(BaseModel):
    id: str
    item: str
    metadata: Dict

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.post(f"/api/v1/items", response_model=ApiResponse, tags=[f"{ITEM_TYPE_NAME.capitalize()}s"])
async def store_item(request: ItemRequest):
    """
    Store a new item with its type and optional metadata.
    """
    try:
        item_id = storage.store_item(item_text=request.item, item_type=request.type, metadata=request.metadata)
        # Fetch the stored item to return
        result = storage.get_item_by_id(item_id)
        if not result:
            raise ApiError(status_code=500, message=f"Failed to retrieve stored {ITEM_TYPE_NAME}")
        return ApiResponse(status=ResponseStatus.SUCCESS, data=result)
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

@app.get(f"/api/v1/items/type/{{item_type}}", response_model=ApiResponse, tags=[f"{ITEM_TYPE_NAME.capitalize()}s"])
async def get_items_by_type(item_type: str, pagination: PaginationParams = Depends()):
    """
    Retrieve items by type with pagination.
    """
    try:
        results = storage.get_items_by_type(item_type=item_type, limit=pagination.limit, skip=pagination.get_skip())
        total_count = storage.count_items_by_type(item_type)
        return ApiResponse(status=ResponseStatus.SUCCESS, data=paginate_results(results, pagination, total_count))
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

@app.post(f"/api/v1/items/search", response_model=ApiResponse, tags=[f"{ITEM_TYPE_NAME.capitalize()}s"])
async def search_similar_items(request: SearchRequest):
    """
    Search for similar items using semantic similarity.
    """
    try:
        results = storage.search_similar_items(query_text=request.query, limit=request.limit)
        return ApiResponse(status=ResponseStatus.SUCCESS, data=results)
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

@app.get(f"/api/v1/items/{{item_id}}", response_model=ApiResponse, tags=[f"{ITEM_TYPE_NAME.capitalize()}s"])
async def get_item_by_id(item_id: str):
    """
    Retrieve an item by its ID.
    """
    try:
        result = storage.get_item_by_id(item_id)
        if not result:
            raise ApiError(status_code=404, message=f"{ITEM_TYPE_NAME.capitalize()} not found")
        return ApiResponse(status=ResponseStatus.SUCCESS, data=result)
    except ApiError as e:
        raise e
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

# Add a health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# You can add legacy endpoints here if needed, pointing to new functions
