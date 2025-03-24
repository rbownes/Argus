from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from .query_storage import QueryStorage

app = FastAPI()
storage = QueryStorage()

class QueryRequest(BaseModel):
    query: str
    theme: str
    metadata: Optional[Dict] = None

class QueryResponse(BaseModel):
    id: str
    query: str
    metadata: Dict

@app.post("/queries", response_model=QueryResponse)
async def store_query(request: QueryRequest):
    try:
        query_id = storage.store_query(
            query=request.query,
            theme=request.theme,
            metadata=request.metadata
        )
        
        # Fetch the stored query to return
        results = storage.get_queries_by_theme(request.theme, limit=1)
        if not results:
            raise HTTPException(status_code=500, detail="Failed to store query")
            
        return results[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queries/theme/{theme}", response_model=List[QueryResponse])
async def get_queries_by_theme(theme: str, limit: int = 10):
    try:
        return storage.get_queries_by_theme(theme, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/queries/search", response_model=List[QueryResponse])
async def search_similar_queries(query: str, limit: int = 5):
    try:
        return storage.search_similar_queries(query, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 