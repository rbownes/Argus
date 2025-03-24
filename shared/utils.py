"""
Shared utilities for all microservices in the Panopticon system.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from pydantic import BaseModel, Field

# Standard API response model
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

class ApiResponse(BaseModel):
    """Standard API response format for all services."""
    status: ResponseStatus
    data: Optional[Any] = None
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# Error handling utilities
class ApiError(Exception):
    """Base exception for API errors with status code and message."""
    def __init__(
        self, 
        status_code: int = 500, 
        message: str = "An unexpected error occurred", 
        details: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(self.message)

# Logging configuration
def setup_logging(service_name: str) -> logging.Logger:
    """Set up structured logging for a service."""
    logger = logging.getLogger(service_name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Configuration utilities
def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Get environment variable with validation.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        required: If True, raise error when missing
        
    Returns:
        Environment variable value or default
    """
    value = os.environ.get(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value

# API extension utilities
def create_api_app(title: str, description: str, version: str = "0.1.0") -> FastAPI:
    """Create a FastAPI application with standard configuration."""
    app = FastAPI(title=title, description=description, version=version)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For development - restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add exception handler for ApiError
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return JSONResponse(
            status_code=exc.status_code,
            content=ApiResponse(
                status=ResponseStatus.ERROR,
                message=exc.message,
                data=exc.details
            ).dict()
        )
    
    # Add generic exception handler
    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=ApiResponse(
                status=ResponseStatus.ERROR,
                message="An unexpected error occurred"
            ).dict()
        )
    
    # Add health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check():
        return ApiResponse(status=ResponseStatus.SUCCESS, message="Service is healthy")
    
    return app

# Pagination utilities
class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""
    page: int = Field(1, ge=1, description="Page number, starting from 1")
    limit: int = Field(10, ge=1, le=100, description="Number of items per page")
    
    def get_skip(self) -> int:
        """Calculate skip value for database queries."""
        return (self.page - 1) * self.limit

def paginate_results(
    items: List[Any], 
    params: PaginationParams, 
    total_count: int
) -> Dict[str, Any]:
    """Format paginated results with metadata."""
    return {
        "items": items,
        "pagination": {
            "page": params.page,
            "limit": params.limit,
            "total": total_count,
            "pages": (total_count + params.limit - 1) // params.limit
        }
    }
