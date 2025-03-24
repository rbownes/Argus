from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from .judge_storage import JudgeStorage

app = FastAPI()
storage = JudgeStorage()

class QueryEvaluationRequest(BaseModel):
    query: str
    model_id: str
    theme: str
    evaluation_prompt_ids: List[str]
    judge_model: Optional[str] = "gpt-4"
    metadata: Optional[Dict] = None

class ThemeEvaluationRequest(BaseModel):
    theme: str
    model_id: str
    evaluation_prompt_ids: List[str]
    judge_model: Optional[str] = "gpt-4"
    limit: Optional[int] = 10
    metadata: Optional[Dict] = None

@app.post("/evaluate/query")
async def evaluate_single_query(request: QueryEvaluationRequest):
    """
    Run a single query through an LLM and evaluate it using specified evaluation prompts
    """
    try:
        # Run query through LLM
        llm_output = await storage.run_query_with_llm(
            query=request.query,
            model_id=request.model_id,
            theme=request.theme,
            metadata=request.metadata
        )
        
        # Get evaluation prompts
        from evaluation_storage.evaluation_storage import EvaluationStorage
        eval_storage = EvaluationStorage()
        
        # Evaluate using each prompt
        evaluation_results = []
        for prompt_id in request.evaluation_prompt_ids:
            # Get the evaluation prompt
            metrics = eval_storage.get_metrics_by_type(prompt_id, limit=1)
            if not metrics:
                continue
                
            evaluation_prompt = metrics[0]["prompt"]
            
            # Run evaluation
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
        
        return {
            "output": llm_output,
            "evaluations": evaluation_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/theme")
async def evaluate_theme_queries(request: ThemeEvaluationRequest):
    """
    Run all queries of a theme through an LLM and evaluate them
    """
    try:
        # Get queries by theme
        from query_storage.query_storage import QueryStorage
        query_storage = QueryStorage()
        queries = query_storage.get_queries_by_theme(request.theme, request.limit)
        
        # Process each query
        results = []
        for query_data in queries:
            eval_request = QueryEvaluationRequest(
                query=query_data["query"],
                model_id=request.model_id,
                theme=request.theme,
                evaluation_prompt_ids=request.evaluation_prompt_ids,
                judge_model=request.judge_model,
                metadata={
                    "query_id": query_data["id"],
                    **(request.metadata or {})
                }
            )
            result = await evaluate_single_query(eval_request)
            results.append(result)
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def get_evaluation_results(
    theme: Optional[str] = None,
    model_id: Optional[str] = None,
    evaluation_prompt_id: Optional[str] = None,
    limit: int = 100
):
    """
    Get evaluation results with optional filters
    """
    try:
        return storage.get_evaluation_results(
            theme=theme,
            model_id=model_id,
            evaluation_prompt_id=evaluation_prompt_id,
            limit=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 