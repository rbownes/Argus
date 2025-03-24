from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from .evaluation_storage import EvaluationStorage

app = FastAPI()
storage = EvaluationStorage()

class EvaluationMetricRequest(BaseModel):
    prompt: str
    metric_type: str
    metadata: Optional[Dict] = None

class EvaluationMetricResponse(BaseModel):
    id: str
    prompt: str
    metadata: Dict

@app.post("/evaluation-metrics", response_model=EvaluationMetricResponse)
async def store_evaluation_metric(request: EvaluationMetricRequest):
    try:
        metric_id = storage.store_evaluation_metric(
            prompt=request.prompt,
            metric_type=request.metric_type,
            metadata=request.metadata
        )
        
        # Fetch the stored metric to return
        results = storage.get_metrics_by_type(request.metric_type, limit=1)
        if not results:
            raise HTTPException(status_code=500, detail="Failed to store evaluation metric")
            
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluation-metrics/type/{metric_type}", response_model=List[EvaluationMetricResponse])
async def get_metrics_by_type(metric_type: str, limit: int = 10):
    try:
        return storage.get_metrics_by_type(metric_type, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluation-metrics/search", response_model=List[EvaluationMetricResponse])
async def search_similar_metrics(prompt: str, limit: int = 5):
    try:
        return storage.search_similar_metrics(prompt, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 