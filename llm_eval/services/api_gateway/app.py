"""
API Gateway for the LLM Evaluation Framework.
"""
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from llm_eval.core.models import (
    Prompt, 
    PromptCategory, 
    LLMResponse, 
    EvaluationType,
    EvaluationResult,
    BatchQueryRequest,
    BatchQueryResponse
)
from llm_eval.core.utils import Result, generate_id
from llm_eval.services.prompt_service import InMemoryPromptService
from llm_eval.services.llm_service import MockLLMService
from llm_eval.services.evaluation_service import (
    EvaluationService,
    ToxicityEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator
)
from llm_eval.services.storage_service import InMemoryStorageService


# Initialize the FastAPI app
app = FastAPI(
    title="LLM Evaluation Framework API",
    description="API for evaluating and comparing LLMs",
    version="0.1.0"
)

# Initialize services
prompt_service = None
llm_service = None
evaluation_service = None
storage_service = None


async def init_services():
    """Initialize all services if not already initialized."""
    global prompt_service, llm_service, evaluation_service, storage_service
    
    if prompt_service is None:
        prompt_service = InMemoryPromptService()
    
    if llm_service is None:
        # Use MockLLMService for testing
        # In production, use LiteLLMService with proper API keys
        llm_service = MockLLMService()
    
    if evaluation_service is None:
        evaluation_service = EvaluationService()
        await evaluation_service.register_evaluator(ToxicityEvaluator())
        await evaluation_service.register_evaluator(RelevanceEvaluator())
        await evaluation_service.register_evaluator(CoherenceEvaluator())
    
    if storage_service is None:
        # Use InMemoryStorageService for testing
        # In production, use StorageService with PostgreSQL and ChromaDB
        storage_service = InMemoryStorageService()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await init_services()


# API Models
class PromptCreate(BaseModel):
    """Model for creating a prompt."""
    text: str
    category: PromptCategory = PromptCategory.OTHER
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PromptUpdate(BaseModel):
    """Model for updating a prompt."""
    text: Optional[str] = None
    category: Optional[PromptCategory] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Model for LLM information."""
    name: str
    provider: str
    description: str
    supported: bool


class QueryRequest(BaseModel):
    """Model for a single query request."""
    model_name: str
    prompt_id: str
    parameters: Optional[Dict[str, Any]] = None


class EvaluateRequest(BaseModel):
    """Model for an evaluation request."""
    response_id: str
    evaluation_type: EvaluationType


class CompareRequest(BaseModel):
    """Model for a model comparison request."""
    model_names: List[str]
    evaluation_type: Optional[EvaluationType] = None
    prompt_ids: Optional[List[str]] = None


# Dependency to get services
async def get_services():
    """Get initialized services."""
    await init_services()
    return {
        "prompt_service": prompt_service,
        "llm_service": llm_service,
        "evaluation_service": evaluation_service,
        "storage_service": storage_service
    }


# API Routes - Prompts
@app.get("/prompts", response_model=List[Prompt])
async def list_prompts(
    category: Optional[PromptCategory] = None,
    tags: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    services=Depends(get_services)
):
    """List prompts with optional filtering."""
    tag_list = tags.split(",") if tags else None
    result = await services["prompt_service"].list_prompts(
        category=category,
        tags=tag_list,
        limit=limit,
        offset=offset
    )
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    return result.unwrap()


@app.post("/prompts", response_model=Prompt)
async def create_prompt(
    prompt_create: PromptCreate,
    services=Depends(get_services)
):
    """Create a new prompt."""
    prompt = Prompt(**prompt_create.dict())
    result = await services["prompt_service"].create_prompt(prompt)
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    # Also store in storage service
    prompt_result = result.unwrap()
    await services["storage_service"].store_prompt(prompt_result)
    
    return prompt_result


@app.get("/prompts/{prompt_id}", response_model=Prompt)
async def get_prompt(
    prompt_id: str,
    services=Depends(get_services)
):
    """Get a prompt by ID."""
    result = await services["prompt_service"].get_prompt(prompt_id)
    
    if result.is_err:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    
    return result.unwrap()


@app.put("/prompts/{prompt_id}", response_model=Prompt)
async def update_prompt(
    prompt_id: str,
    prompt_update: PromptUpdate,
    services=Depends(get_services)
):
    """Update a prompt."""
    # Get existing prompt
    get_result = await services["prompt_service"].get_prompt(prompt_id)
    
    if get_result.is_err:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    
    existing_prompt = get_result.unwrap()
    
    # Update fields
    update_data = prompt_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(existing_prompt, field, value)
    
    # Save updated prompt
    result = await services["prompt_service"].update_prompt(prompt_id, existing_prompt)
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    # Also update in storage service
    prompt_result = result.unwrap()
    await services["storage_service"].store_prompt(prompt_result)
    
    return prompt_result


@app.delete("/prompts/{prompt_id}", response_model=bool)
async def delete_prompt(
    prompt_id: str,
    services=Depends(get_services)
):
    """Delete a prompt."""
    result = await services["prompt_service"].delete_prompt(prompt_id)
    
    if result.is_err:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
    
    return result.unwrap()


@app.get("/prompts/search", response_model=List[Prompt])
async def search_prompts(
    query: str,
    services=Depends(get_services)
):
    """Search prompts by text."""
    result = await services["prompt_service"].search_prompts(query)
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    return result.unwrap()


# API Routes - LLMs
@app.get("/models", response_model=List[ModelInfo])
async def list_models(
    services=Depends(get_services)
):
    """List available LLM models."""
    result = await services["llm_service"].get_available_models()
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    return result.unwrap()


@app.post("/query", response_model=LLMResponse)
async def query_model(
    query_request: QueryRequest,
    services=Depends(get_services)
):
    """Query a single model with a prompt."""
    # Get the prompt
    prompt_result = await services["prompt_service"].get_prompt(query_request.prompt_id)
    
    if prompt_result.is_err:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {query_request.prompt_id}")
    
    prompt = prompt_result.unwrap()
    
    # Query the model
    result = await services["llm_service"].query_model(
        model_name=query_request.model_name,
        prompt=prompt,
        parameters=query_request.parameters
    )
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    # Store the response
    response = result.unwrap()
    await services["storage_service"].store_response(response)
    
    return response


@app.post("/batch", response_model=BatchQueryResponse)
async def batch_query(
    batch_request: BatchQueryRequest,
    services=Depends(get_services)
):
    """Run a batch query with multiple models and prompts."""
    # Get all prompts
    prompts = []
    for prompt_id in batch_request.prompt_ids:
        prompt_result = await services["prompt_service"].get_prompt(prompt_id)
        
        if prompt_result.is_err:
            raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")
        
        prompts.append(prompt_result.unwrap())
    
    # Run the batch query
    result = await services["llm_service"].batch_query(
        model_names=batch_request.model_names,
        prompts=prompts,
        parameters=batch_request.metadata.get("parameters")
    )
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    responses = result.unwrap()
    
    # Run evaluations if requested
    evaluations = []
    if batch_request.evaluations:
        # Get evaluators
        evaluators = {}
        for eval_type in batch_request.evaluations:
            evaluator_result = await services["evaluation_service"].get_evaluator(eval_type)
            if evaluator_result.is_ok:
                evaluators[eval_type] = evaluator_result.unwrap()
        
        # Evaluate each response
        for response in responses:
            for eval_type, evaluator in evaluators.items():
                eval_result = await evaluator.evaluate(response)
                if eval_result.is_ok:
                    evaluations.append(eval_result.unwrap())
    
    # Create batch response
    batch_response = BatchQueryResponse(
        batch_id=generate_id(),
        responses=responses,
        evaluations=evaluations if evaluations else None
    )
    
    # Store the batch
    await services["storage_service"].store_batch(batch_request, batch_response)
    
    return batch_response


# API Routes - Evaluations
@app.get("/evaluators", response_model=List[str])
async def list_evaluators(
    services=Depends(get_services)
):
    """List available evaluators."""
    result = await services["evaluation_service"].list_evaluators()
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    evaluators = result.unwrap()
    return [e.evaluation_type for e in evaluators]


@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_response(
    evaluate_request: EvaluateRequest,
    services=Depends(get_services)
):
    """Evaluate a single response."""
    # Get the response
    response_result = await services["storage_service"].get_response(
        evaluate_request.response_id
    )
    
    if response_result.is_err:
        raise HTTPException(status_code=404, detail=f"Response not found: {evaluate_request.response_id}")
    
    response = response_result.unwrap()
    
    # Evaluate the response
    result = await services["evaluation_service"].evaluate_response(
        response=response,
        evaluation_type=evaluate_request.evaluation_type
    )
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    # Store the evaluation
    evaluation = result.unwrap()
    await services["storage_service"].store_evaluation(evaluation)
    
    return evaluation


@app.post("/batch-evaluate", response_model=List[EvaluationResult])
async def batch_evaluate(
    response_ids: List[str],
    evaluation_types: List[EvaluationType],
    services=Depends(get_services)
):
    """Perform batch evaluations on multiple responses."""
    # Get all responses
    responses = []
    for response_id in response_ids:
        response_result = await services["storage_service"].get_response(response_id)
        
        if response_result.is_err:
            raise HTTPException(status_code=404, detail=f"Response not found: {response_id}")
        
        responses.append(response_result.unwrap())
    
    # Run batch evaluations
    result = await services["evaluation_service"].batch_evaluate(
        responses=responses,
        evaluation_types=evaluation_types
    )
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    # Store all evaluations
    evaluations = result.unwrap()
    for evaluation in evaluations:
        await services["storage_service"].store_evaluation(evaluation)
    
    return evaluations


# API Routes - Results
@app.get("/results", response_model=List[LLMResponse])
async def get_results(
    model_name: Optional[str] = None,
    prompt_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    services=Depends(get_services)
):
    """Get LLM responses with filtering."""
    if batch_id:
        # Get responses from a specific batch
        batch_result = await services["storage_service"].get_batch(batch_id)
        
        if batch_result.is_err:
            raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
        
        batch = batch_result.unwrap()
        responses = batch.responses
        
        # Apply additional filters if needed
        if model_name:
            responses = [r for r in responses if r.model_name == model_name]
        
        if prompt_id:
            responses = [r for r in responses if r.prompt_id == prompt_id]
        
        # Apply pagination
        responses = responses[offset:offset + limit]
        
        return responses
    else:
        # Query storage service directly
        result = await services["storage_service"].query_responses(
            model_name=model_name,
            prompt_id=prompt_id,
            limit=limit,
            offset=offset
        )
        
        if result.is_err:
            raise HTTPException(status_code=500, detail=str(result.error))
        
        return result.unwrap()


@app.post("/results/compare", response_model=Dict[str, Any])
async def compare_models(
    compare_request: CompareRequest,
    services=Depends(get_services)
):
    """Compare the performance of multiple models."""
    result = await services["storage_service"].compare_models(
        model_names=compare_request.model_names,
        evaluation_type=compare_request.evaluation_type,
        prompt_ids=compare_request.prompt_ids
    )
    
    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.error))
    
    return result.unwrap()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is healthy."""
    return {"status": "ok"}


# Run the app
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8080))
    
    uvicorn.run(app, host="0.0.0.0", port=port)
