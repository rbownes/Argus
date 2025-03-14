"""
FastAPI application for LLM Evaluation System.
"""
import os
from typing import List, Dict, Any, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from llm_eval.core.models import ModelConfig, ThemeCategory, EvaluationMetric, BatchQueryRequest, BatchQueryResponse
from llm_eval.services.llm_service.service import LLMQueryService
from llm_eval.services.evaluation_service.service import EvaluationService
from llm_eval.services.evaluation_service.rule_based import RuleBasedEvaluator
from llm_eval.services.storage_service.service import StorageService
from llm_eval.services.orchestration_service.service import OrchestrationService


app = FastAPI(title="LLM Evaluation System")

# Initialize services using environment variables or defaults
postgres_url = os.environ.get("POSTGRES_URL", "postgresql://user:password@localhost:5432/llm_eval")
chroma_path = os.environ.get("CHROMA_PATH", "./chroma_db")

# Initialize services
storage_service = StorageService(
    postgres_url=postgres_url,
    chroma_path=chroma_path
)

llm_service = LLMQueryService()

evaluation_service = EvaluationService()
evaluation_service.register_evaluator(RuleBasedEvaluator())

orchestration_service = OrchestrationService(
    llm_service=llm_service,
    evaluation_service=evaluation_service,
    storage_service=storage_service
)


class PerformanceQueryRequest(BaseModel):
    """Request model for querying model performance."""
    model_provider: str
    model_id: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search."""
    query_text: str
    n_results: int = 10
    filter_metadata: Optional[Dict[str, Any]] = None


# API endpoints
@app.post("/api/v1/evaluations", response_model=BatchQueryResponse)
async def create_evaluation(request: BatchQueryRequest):
    """Create a new evaluation run."""
    try:
        run_id = await orchestration_service.create_evaluation_run(
            model_configs=request.models,
            themes=request.themes,
            evaluator_ids=request.evaluator_ids if hasattr(request, 'evaluator_ids') else ["rule_based_evaluator"],
            metrics=request.metrics,
            metadata=request.metadata
        )
        
        return BatchQueryResponse(run_id=run_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/evaluations/{run_id}", response_model=Dict[str, Any])
async def get_evaluation_status(run_id: UUID):
    """Get the status of an evaluation run."""
    try:
        status = await orchestration_service.get_run_status(run_id)
        return status
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/evaluations/{run_id}/results", response_model=Dict[str, Any])
async def get_evaluation_results(run_id: UUID):
    """Get the results of an evaluation run."""
    try:
        results = await orchestration_service.get_run_results(run_id)
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/performance", response_model=Dict[str, Any])
async def get_model_performance(request: PerformanceQueryRequest):
    """Get performance metrics for a specific model."""
    try:
        # Convert string timestamps to datetime objects if provided
        start_time = None
        end_time = None
        
        if request.start_time:
            from datetime import datetime
            start_time = datetime.fromisoformat(request.start_time)
            
        if request.end_time:
            from datetime import datetime
            end_time = datetime.fromisoformat(request.end_time)
            
        performance = await storage_service.get_model_performance(
            model_provider=request.model_provider,
            model_id=request.model_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return performance
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/semantic_search", response_model=List[Dict[str, Any]])
async def search_semantically_similar(request: SemanticSearchRequest):
    """Search for semantically similar responses."""
    try:
        results = await storage_service.query_semantically_similar(
            query_text=request.query_text,
            n_results=request.n_results,
            filter_metadata=request.filter_metadata
        )
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/{model_provider}/{model_id}/responses", response_model=List[Dict[str, Any]])
async def get_model_responses(
    model_provider: str,
    model_id: str,
    limit: int = 100,
    offset: int = 0
):
    """Get responses from a specific model."""
    try:
        responses = await storage_service.get_responses_by_model(
            model_provider=model_provider,
            model_id=model_id,
            limit=limit,
            offset=offset
        )
        
        return responses
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
