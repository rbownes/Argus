"""
Middleware components for all microservices in the Panopticon system.
"""
import time
from typing import Callable, Dict, Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import logging
from .utils import ApiResponse, ResponseStatus, ApiError

# Request ID middleware
async def request_id_middleware(request: Request, call_next: Callable) -> Response:
    """Add unique request ID to each request for tracing."""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Logging middleware
async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log request and response details."""
    logger = logging.getLogger("api")
    
    # Extract request details
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    
    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code} - {process_time:.4f}s",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "status_code": response.status_code,
                "process_time": process_time
            }
        )
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as exc:
        process_time = time.time() - start_time
        logger.exception(
            f"Error during request processing: {str(exc)}",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "process_time": process_time,
                "error": str(exc)
            }
        )
        raise

# Rate limiting middleware (simple in-memory implementation)
class RateLimiter:
    """Simple in-memory rate limiter."""
    def __init__(self, rate_limit: int = 100, window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            rate_limit: Maximum requests per window
            window: Time window in seconds
        """
        self.rate_limit = rate_limit
        self.window = window
        self.clients: Dict[str, Dict[str, int]] = {}

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting to request."""
        # Get client identifier (IP address in this simple implementation)
        client_id = request.client.host if request.client else "unknown"
        
        # Check rate limit
        current_time = int(time.time())
        if client_id in self.clients:
            # Clean old entries
            self.clients[client_id] = {
                ts: count for ts, count in self.clients[client_id].items()
                if current_time - int(ts) < self.window
            }
            
            # Calculate current request count in window
            total_requests = sum(self.clients[client_id].values())
            
            if total_requests >= self.rate_limit:
                return JSONResponse(
                    status_code=429,
                    content=ApiResponse(
                        status=ResponseStatus.ERROR,
                        message="Rate limit exceeded. Please try again later."
                    ).dict()
                )
        else:
            self.clients[client_id] = {}
        
        # Update request count
        time_bucket = str(current_time)
        if time_bucket in self.clients[client_id]:
            self.clients[client_id][time_bucket] += 1
        else:
            self.clients[client_id][time_bucket] = 1
        
        # Process request
        return await call_next(request)

# Simple API key authentication
class ApiKeyAuth:
    """Simple API key authentication middleware."""
    def __init__(self, api_key: Optional[str] = None, api_key_header: str = "X-API-Key"):
        """
        Initialize API key authentication.
        
        Args:
            api_key: API key to validate against (if None, no validation)
            api_key_header: Header name for API key
        """
        self.api_key = api_key
        self.api_key_header = api_key_header

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Validate API key if configured."""
        # Skip validation if no API key is set
        if not self.api_key:
            return await call_next(request)
        
        # Skip validation for health check endpoint
        if request.url.path == "/health":
            return await call_next(request)
        
        # Check API key
        api_key = request.headers.get(self.api_key_header)
        if not api_key or api_key != self.api_key:
            return JSONResponse(
                status_code=401,
                content=ApiResponse(
                    status=ResponseStatus.ERROR,
                    message="Invalid or missing API key"
                ).dict()
            )
        
        # API key is valid
        return await call_next(request)

# Function to add common middleware to FastAPI app
def add_middleware(app: FastAPI, api_key: Optional[str] = None) -> None:
    """Add common middleware to FastAPI application."""
    # Add request ID middleware
    app.middleware("http")(request_id_middleware)
    
    # Add logging middleware
    app.middleware("http")(logging_middleware)
    
    # Add rate limiting (disabled by default, enable in production)
    # app.middleware("http")(RateLimiter(rate_limit=100, window=60))
    
    # Add API key authentication if key is provided
    if api_key:
        app.middleware("http")(ApiKeyAuth(api_key=api_key))
    
    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response
