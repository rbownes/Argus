"""
API endpoints for the visualization service.
"""
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, Query, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
from pydantic import BaseModel, Field
import json
from visualization_service.database import VisualizationDB
from visualization_service.dashboard import Dashboard
import pandas as pd

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("visualization_service.api")

# Initialize API
app = FastAPI(
    title="Panopticon Visualization Service",
    description="Visualization service for Panopticon model evaluation system",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Get database connection string from environment variables
db_host = os.getenv("POSTGRES_HOST", "postgres")
db_port = os.getenv("POSTGRES_PORT", "5432")
db_user = os.getenv("POSTGRES_USER", "postgres")
db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
db_name = os.getenv("POSTGRES_DB", "panopticon")
db_url = os.getenv(
    "DATABASE_URL",
    f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

# Initialize database and dashboard
db = VisualizationDB(connection_string=db_url)
dashboard = Dashboard(db)

# API key for authentication
API_KEY = os.getenv("API_KEY", "development_key")

def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key."""
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key

# ---------------------- Models ----------------------

class DateRangeParams(BaseModel):
    """Date range parameters for filtering."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

class FilterParams(DateRangeParams):
    """Filter parameters for dashboard data."""
    models: Optional[List[str]] = None
    themes: Optional[List[str]] = None
    evaluation_prompt_ids: Optional[List[str]] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None

class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, gt=0)
    page_size: int = Field(10, gt=0, le=100)
    
class SortParams(BaseModel):
    """Sort parameters."""
    sort_by: str = "timestamp"
    sort_desc: bool = True

# ---------------------- API Endpoints ----------------------

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Panopticon Visualization Service",
        "version": "0.1.0",
        "description": "Visualization service for Panopticon model evaluation system",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check database connection
    db_healthy = db.check_connection()
    
    if db_healthy:
        return {"status": "healthy", "database": "connected"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": "disconnected"}
        )

@app.get("/api/v1/dashboard/summary")
async def get_dashboard_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    models: Optional[str] = None,
    themes: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get dashboard summary statistics.
    
    Args:
        start_date: Start date for filtering (ISO format)
        end_date: End date for filtering (ISO format)
        models: Comma-separated list of model IDs
        themes: Comma-separated list of themes
    """
    try:
        # Parse date parameters
        start_datetime = datetime.fromisoformat(start_date) if start_date else None
        end_datetime = datetime.fromisoformat(end_date) if end_date else None
        
        # Parse list parameters
        model_list = models.split(",") if models else None
        theme_list = themes.split(",") if themes else None
        
        # Get summary data
        summary = await dashboard.get_summary(
            start_date=start_datetime,
            end_date=end_datetime,
            models=model_list,
            themes=theme_list
        )
        
        return summary
    except Exception as e:
        logger.error(f"Error in get_dashboard_summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dashboard/timeline")
async def get_performance_timeline(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    models: Optional[str] = None,
    themes: Optional[str] = None,
    time_grouping: Optional[str] = "day",
    api_key: str = Depends(verify_api_key)
):
    """
    Get model performance timeline data.
    
    Args:
        start_date: Start date for filtering (ISO format)
        end_date: End date for filtering (ISO format)
        models: Comma-separated list of model IDs
        themes: Comma-separated list of themes
        time_grouping: Time grouping (day, week, month)
    """
    try:
        # Parse date parameters
        start_datetime = datetime.fromisoformat(start_date) if start_date else None
        end_datetime = datetime.fromisoformat(end_date) if end_date else None
        
        # Parse list parameters
        model_list = models.split(",") if models else None
        theme_list = themes.split(",") if themes else None
        
        # Validate time_grouping
        valid_groupings = ["day", "week", "month"]
        if time_grouping not in valid_groupings:
            time_grouping = "day"
        
        # Get timeline data
        timeline_data = await dashboard.get_model_performance_timeline(
            start_date=start_datetime,
            end_date=end_datetime,
            models=model_list,
            themes=theme_list,
            time_grouping=time_grouping
        )
        
        return timeline_data
    except Exception as e:
        logger.error(f"Error in get_performance_timeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dashboard/models")
async def get_model_comparison(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    models: Optional[str] = None,
    themes: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get model comparison data.
    
    Args:
        start_date: Start date for filtering (ISO format)
        end_date: End date for filtering (ISO format)
        models: Comma-separated list of model IDs
        themes: Comma-separated list of themes
    """
    try:
        # Parse date parameters
        start_datetime = datetime.fromisoformat(start_date) if start_date else None
        end_datetime = datetime.fromisoformat(end_date) if end_date else None
        
        # Parse list parameters
        model_list = models.split(",") if models else None
        theme_list = themes.split(",") if themes else None
        
        # Get model comparison data
        comparison_data = await dashboard.get_model_comparison(
            start_date=start_datetime,
            end_date=end_datetime,
            models=model_list,
            themes=theme_list
        )
        
        return comparison_data
    except Exception as e:
        logger.error(f"Error in get_model_comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dashboard/themes")
async def get_theme_analysis(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    models: Optional[str] = None,
    themes: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get theme analysis data.
    
    Args:
        start_date: Start date for filtering (ISO format)
        end_date: End date for filtering (ISO format)
        models: Comma-separated list of model IDs
        themes: Comma-separated list of themes
    """
    try:
        # Parse date parameters
        start_datetime = datetime.fromisoformat(start_date) if start_date else None
        end_datetime = datetime.fromisoformat(end_date) if end_date else None
        
        # Parse list parameters
        model_list = models.split(",") if models else None
        theme_list = themes.split(",") if themes else None
        
        # Get theme analysis data
        theme_data = await dashboard.get_theme_analysis(
            start_date=start_datetime,
            end_date=end_datetime,
            models=model_list,
            themes=theme_list
        )
        
        return theme_data
    except Exception as e:
        logger.error(f"Error in get_theme_analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dashboard/results")
async def get_detailed_results(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    models: Optional[str] = None,
    themes: Optional[str] = None,
    evaluation_prompts: Optional[str] = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    page: int = 1,
    page_size: int = 10,
    sort_by: str = "timestamp",
    sort_desc: bool = True,
    api_key: str = Depends(verify_api_key)
):
    """
    Get detailed evaluation results with pagination.
    
    Args:
        start_date: Start date for filtering (ISO format)
        end_date: End date for filtering (ISO format)
        models: Comma-separated list of model IDs
        themes: Comma-separated list of themes
        evaluation_prompts: Comma-separated list of evaluation prompt IDs
        min_score: Minimum score
        max_score: Maximum score
        page: Page number (1-based)
        page_size: Number of results per page
        sort_by: Column to sort by
        sort_desc: Whether to sort in descending order
    """
    try:
        # Parse date parameters
        start_datetime = datetime.fromisoformat(start_date) if start_date else None
        end_datetime = datetime.fromisoformat(end_date) if end_date else None
        
        # Parse list parameters
        model_list = models.split(",") if models else None
        theme_list = themes.split(",") if themes else None
        prompt_list = evaluation_prompts.split(",") if evaluation_prompts else None
        
        # Get detailed results
        results = await dashboard.get_detailed_results(
            start_date=start_datetime,
            end_date=end_datetime,
            models=model_list,
            themes=theme_list,
            evaluation_prompt_ids=prompt_list,
            min_score=min_score,
            max_score=max_score,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_desc=sort_desc
        )
        
        return results
    except Exception as e:
        logger.error(f"Error in get_detailed_results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dashboard/filters")
async def get_filter_options(
    api_key: str = Depends(verify_api_key)
):
    """
    Get available filter options for the dashboard.
    """
    try:
        # Get filter options
        filters = await dashboard.get_filter_options()
        return filters
    except Exception as e:
        logger.error(f"Error in get_filter_options: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for real-time updates (optional)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()
            
            # Parse client request
            try:
                request = json.loads(data)
                request_type = request.get("type")
                
                # Process different request types
                if request_type == "dashboard_summary":
                    # Get dashboard summary
                    summary = await dashboard.get_summary()
                    await websocket.send_json({"type": "dashboard_summary", "data": summary})
                elif request_type == "performance_timeline":
                    # Get performance timeline
                    timeline_data = await dashboard.get_model_performance_timeline()
                    await websocket.send_json({"type": "performance_timeline", "data": timeline_data})
                else:
                    await websocket.send_json({"error": f"Unknown request type: {request_type}"})
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")

# Mount static files for frontend - this will serve the React frontend
app.mount("/", StaticFiles(directory="visualization_service/frontend/dist", html=True), name="frontend")

# Legacy endpoint compatibility
app.get("/api/dashboard/summary")(get_dashboard_summary)
app.get("/api/dashboard/timeline")(get_performance_timeline)
app.get("/api/dashboard/models")(get_model_comparison)
app.get("/api/dashboard/themes")(get_theme_analysis)
app.get("/api/dashboard/results")(get_detailed_results)
app.get("/api/dashboard/filters")(get_filter_options)
