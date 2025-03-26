"""
API for the Judge Service.
"""
from fastapi import Depends, Query, Body
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import os
import asyncio
import json
import logging
from datetime import datetime

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
from .judge_storage import JudgeStorage, EvaluationResult
from .service_clients import QueryStorageClient, EvaluationStorageClient

# Create FastAPI app
app = create_api_app(
    title="Judge Service",
    description="Run and evaluate LLM outputs for the Panopticon system",
    version="0.1.0"
)

# Initialize configuration
config = ServiceConfig.from_env("judge-service")
add_middleware(app, api_key=os.environ.get("API_KEY"))

# Initialize storage
storage = JudgeStorage()

# API Models
class QueryEvaluationRequest(BaseModel):
    """Request model for evaluating a single query."""
    query: str = Field(..., description="The query text to evaluate")
    model_id: str = Field(..., description="ID of the LLM model to use")
    theme: str = Field(..., description="Theme or category of the query")
    evaluation_prompt_ids: List[str] = Field(..., description="List of evaluation prompt IDs to use")
    judge_model: str = Field("gpt-4", description="LLM model to use for evaluation")
    model_provider: Optional[str] = Field(None, description="Provider of the LLM model (e.g., 'google', 'anthropic', 'openai')")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")
    
    @validator('evaluation_prompt_ids')
    def validate_evaluation_prompts(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one evaluation prompt ID is required")
        return v

class ThemeEvaluationRequest(BaseModel):
    """Request model for evaluating all queries of a theme."""
    theme: str = Field(..., description="Theme to evaluate queries for")
    model_id: str = Field(..., description="ID of the LLM model to use")
    evaluation_prompt_ids: List[str] = Field(..., description="List of evaluation prompt IDs to use")
    judge_model: str = Field("gpt-4", description="LLM model to use for evaluation")
    model_provider: Optional[str] = Field(None, description="Provider of the LLM model (e.g., 'google', 'anthropic', 'openai')")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of queries to evaluate")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")
    
    @validator('evaluation_prompt_ids')
    def validate_evaluation_prompts(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one evaluation prompt ID is required")
        return v

class EvaluationResultFilter(BaseModel):
    """Filter parameters for retrieving evaluation results."""
    theme: Optional[str] = Field(None, description="Filter by theme")
    model_id: Optional[str] = Field(None, description="Filter by model ID")
    evaluation_prompt_id: Optional[str] = Field(None, description="Filter by evaluation prompt ID")
    min_score: Optional[float] = Field(None, ge=0, le=10, description="Minimum score")
    max_score: Optional[float] = Field(None, ge=0, le=10, description="Maximum score")
    start_date: Optional[datetime] = Field(None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(None, description="End date for filtering")

# API Routes
@app.post("/api/v1/evaluate/query", response_model=ApiResponse, tags=["Evaluation"])
async def evaluate_single_query(request: QueryEvaluationRequest):
    """
    Run a single query through an LLM and evaluate it using specified evaluation prompts.
    
    Returns the LLM output and evaluation results.
    """
    try:
        # Check if model exists and auto-register if it's new
        models_config_path = os.path.join(os.path.dirname(__file__), "config", "models.json")
        model_exists = False
        
        if os.path.exists(models_config_path):
            try:
                with open(models_config_path, "r") as f:
                    config = json.load(f)
                    model_exists = any(model["id"] == request.model_id for model in config.get("models", []))
            except Exception as e:
                storage.logger.error(f"Error checking model existence: {str(e)}")
        
        # Auto-register new model if it doesn't exist
        if not model_exists:
            # Detect provider if not provided
            provider = request.model_provider or detect_provider_from_id(request.model_id)
            
            # Prepare new model entry
            new_model = {
                "id": request.model_id,
                "name": request.model_id,  # Use ID as name initially
                "provider": provider,
                "is_judge_compatible": False,  # Default to false until verified
                "auto_registered": True  # Flag as auto-registered
            }
            
            # Register the new model
            await register_new_model(new_model)
            storage.logger.info(f"Auto-registered new model: {request.model_id} (provider: {provider})")
            
        # Initialize service clients
        eval_client = EvaluationStorageClient()
        
        # Run query through LLM
        try:
            llm_output = await storage.run_query_with_llm(
                query=request.query,
                model_id=request.model_id,
                theme=request.theme,
                metadata=request.metadata,
                model_provider=request.model_provider
            )
        except Exception as e:
            raise ApiError(
                status_code=500, 
                message=f"Failed to run query with LLM: {str(e)}"
            )
        
        # Evaluate using each prompt
        evaluation_results = []
        for prompt_id in request.evaluation_prompt_ids:
            # Get the evaluation prompt
            try:
                metric = await eval_client.get_metric_by_id(prompt_id)
                if not metric:
                    raise ApiError(
                        status_code=404, 
                        message=f"Evaluation prompt not found: {prompt_id}"
                    )
            except Exception as e:
                if isinstance(e, ApiError):
                    raise
                raise ApiError(
                    status_code=500, 
                    message=f"Failed to retrieve evaluation prompt: {str(e)}"
                )
                
            evaluation_prompt = metric["prompt"]
            
            # Run evaluation
            try:
                result = await storage.evaluate_output(
                    query=request.query,
                    output=llm_output["output"],
                    evaluation_prompt=evaluation_prompt,
                    evaluation_prompt_id=prompt_id,
                    model_id=request.model_id,
                    theme=request.theme,
                    judge_model=request.judge_model,
                    metadata={
                        "output_id": llm_output["id"],
                        **(request.metadata or {})
                    }
                )
                evaluation_results.append(result)
            except Exception as e:
                raise ApiError(
                    status_code=500, 
                    message=f"Failed to evaluate output: {str(e)}"
                )
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            message="Query evaluated successfully",
            data={
                "output": llm_output,
                "evaluations": evaluation_results
            }
        )
    except Exception as e:
        if isinstance(e, ApiError):
            raise
        raise ApiError(status_code=500, message=str(e))

@app.post("/api/v1/evaluate/theme", response_model=ApiResponse, tags=["Evaluation"])
async def evaluate_theme_queries(request: ThemeEvaluationRequest):
    """
    Run all queries of a theme through an LLM and evaluate them.
    
    Returns a list of outputs and evaluation results.
    """
    try:
        # Get queries by theme using the client
        query_client = QueryStorageClient()
        
        try:
            queries = await query_client.get_queries_by_theme(request.theme, request.limit)
        except Exception as e:
            raise ApiError(
                status_code=500, 
                message=f"Failed to retrieve queries: {str(e)}"
            )
        
        if not queries:
            return ApiResponse(
                status=ResponseStatus.SUCCESS,
                message=f"No queries found for theme: {request.theme}",
                data=[]
            )
        
        # Process each query
        results = []
        for query_data in queries:
            eval_request = QueryEvaluationRequest(
                query=query_data["query"],
                model_id=request.model_id,
                theme=request.theme,
                evaluation_prompt_ids=request.evaluation_prompt_ids,
                judge_model=request.judge_model,
                model_provider=request.model_provider,
                metadata={
                    "query_id": query_data["id"],
                    **(request.metadata or {})
                }
            )
            
            try:
                result = await evaluate_single_query(eval_request)
                results.append(result.data)
            except Exception as e:
                # Log error but continue with other queries
                storage.logger.error(f"Failed to evaluate query {query_data['id']}: {str(e)}")
                results.append({
                    "query": query_data,
                    "error": str(e)
                })
            
            # Small delay to avoid rate limiting on LLM APIs
            await asyncio.sleep(0.5)
            
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Evaluated {len(results)} queries for theme: {request.theme}",
            data=results
        )
    except Exception as e:
        if isinstance(e, ApiError):
            raise
        raise ApiError(status_code=500, message=str(e))

@app.get("/api/v1/results", response_model=ApiResponse, tags=["Results"])
async def get_evaluation_results(
    filter_params: EvaluationResultFilter = Depends(),
    pagination: PaginationParams = Depends()
):
    """
    Get evaluation results with filtering and pagination.
    
    Returns a list of evaluation results matching the specified filters.
    """
    try:
        results, total_count = storage.get_evaluation_results(
            theme=filter_params.theme,
            model_id=filter_params.model_id,
            evaluation_prompt_id=filter_params.evaluation_prompt_id,
            min_score=filter_params.min_score,
            max_score=filter_params.max_score,
            start_date=filter_params.start_date,
            end_date=filter_params.end_date,
            limit=pagination.limit,
            skip=pagination.get_skip()
        )
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=paginate_results(results, pagination, total_count)
        )
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

@app.get("/api/v1/models", response_model=ApiResponse, tags=["Models"])
async def list_available_models():
    """
    List available LLM models for evaluation.
    
    Returns a list of models that can be used for running queries and evaluation.
    """
    try:
        # Get models from Provider Manager
        models = await storage.list_available_models()
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=models
        )
    except Exception as e:
        raise ApiError(status_code=500, message=str(e))

# Helper function to detect provider from model ID
def detect_provider_from_id(model_id: str) -> str:
    """
    Attempt to infer the provider from the model ID.
    
    Args:
        model_id: The model identifier
        
    Returns:
        The inferred provider name, or "unknown" if it can't be determined
    """
    model_id = model_id.lower()
    if any(name in model_id for name in ["gpt", "davinci", "chatgpt", "openai"]):
        return "openai"
    elif any(name in model_id for name in ["claude", "anthropic"]):
        return "anthropic"
    elif any(name in model_id for name in ["gemini", "palm", "bard", "google"]):
        return "google"
    elif any(name in model_id for name in ["llama", "meta"]):
        return "meta"
    elif any(name in model_id for name in ["mistral"]):
        return "mistral"
    else:
        return "unknown"

# Function to register a new model
async def register_new_model(model_data: Dict[str, Any]) -> bool:
    """
    Register a new model with the Provider Manager.
    
    Args:
        model_data: Dictionary containing the model information
        
    Returns:
        True if the model was registered successfully, False otherwise
    """
    try:
        # Use the provider manager to register the model
        return storage.provider_manager.register_model(model_data)
    except Exception as e:
        storage.logger.error(f"Error registering new model: {str(e)}")
        return False

# For backward compatibility, keep the original routes but direct to the new implementations
@app.post("/evaluate/query", tags=["Legacy"])
async def evaluate_single_query_legacy(request: QueryEvaluationRequest):
    """Legacy endpoint for evaluating a single query."""
    response = await evaluate_single_query(request)
    return response.data

@app.post("/evaluate/theme", tags=["Legacy"])
async def evaluate_theme_queries_legacy(request: ThemeEvaluationRequest):
    """Legacy endpoint for evaluating all queries of a theme."""
    response = await evaluate_theme_queries(request)
    return response.data

@app.get("/results", tags=["Legacy"])
async def get_evaluation_results_legacy(
    theme: Optional[str] = None,
    model_id: Optional[str] = None,
    evaluation_prompt_id: Optional[str] = None,
    limit: int = 100
):
    """Legacy endpoint for retrieving evaluation results."""
    filter_params = EvaluationResultFilter(
        theme=theme,
        model_id=model_id,
        evaluation_prompt_id=evaluation_prompt_id
    )
    pagination = PaginationParams(page=1, limit=limit)
    
    response = await get_evaluation_results(
        filter_params=filter_params,
        pagination=pagination
    )
    return response.data["items"]
