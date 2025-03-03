"""
Module for querying multiple LLM models using litellm.

This module provides functionality to send prompts to various LLM models
and collect their responses in a structured format.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import uuid
from pydantic import BaseModel, Field, validator, ConfigDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import litellm
    from litellm.utils import ModelResponse
except ImportError:
    logger.error("litellm package not found. Please install it with 'pip install litellm'")
    raise


class LLMQueryParams(BaseModel):
    """Pydantic model for validating LLM query parameters."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    models: List[str] = Field(..., description="List of model identifiers compatible with litellm")
    prompts: List[str] = Field(..., description="List of text prompts to send to each model")
    max_tokens: int = Field(default=1000, description="Maximum number of tokens to generate in responses")
    temperature: float = Field(default=0.7, description="Controls randomness in the model output (0.0-1.0)")
    batch_id: Optional[str] = Field(default=None, description="Identifier for this batch of queries")
    
    @validator('models')
    def validate_models(cls, v):
        if not v:
            raise ValueError("At least one model must be provided")
        return v
    
    @validator('prompts')
    def validate_prompts(cls, v):
        if not v:
            raise ValueError("At least one prompt must be provided")
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v
    
    @validator('batch_id', pre=True, always=True)
    def validate_batch_id(cls, v):
        return v or f"batch-{uuid.uuid4()}"


class LLMResponse(BaseModel):
    """Structured response from an LLM query."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model: str = Field(..., description="Name of the model that generated the response")
    prompt: str = Field(..., description="The prompt that was sent to the model")
    response_text: str = Field(..., description="The text response from the model")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    raw_response: Optional[Any] = Field(default=None, description="Raw response object")
    error: Optional[str] = Field(default=None, description="Error message if query failed")


def query_llm_models(
    models: List[str], 
    prompts: List[str], 
    max_tokens: int = 1000, 
    temperature: float = 0.7,
    batch_id: Optional[str] = None,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, List[LLMResponse]]:
    """
    Query multiple LLM models with a list of prompts and return their responses.
    
    Args:
        models: A list of model identifiers compatible with litellm.
        prompts: A list of text prompts to send to each model.
        max_tokens: Maximum number of tokens to generate in responses.
        temperature: Controls randomness in the model output (0.0-1.0).
        batch_id: Optional identifier for this batch of queries.
        custom_metadata: Optional additional metadata to include with each response.
        
    Returns:
        A dictionary mapping model names to lists of LLMResponse objects,
        where each response corresponds to a prompt in the input list.
        
    Raises:
        ValueError: If parameter validation fails.
        Exception: If there's an error contacting the model provider.
    """
    # Validate inputs with Pydantic
    try:
        params = LLMQueryParams(
            models=models,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            batch_id=batch_id
        )
    except Exception as e:
        logger.error(f"Parameter validation failed: {e}")
        raise ValueError(f"Parameter validation failed: {e}")
    
    # Assert input validity
    assert len(params.models) > 0, "At least one model must be provided"
    assert len(params.prompts) > 0, "At least one prompt must be provided"
    
    # Initialize the results dictionary
    results: Dict[str, List[LLMResponse]] = {}
    
    # Initialize metadata if not provided
    if custom_metadata is None:
        custom_metadata = {}
    
    # Add batch_id to metadata
    metadata = {
        "batch_id": params.batch_id,
        "query_timestamp": datetime.now().isoformat(),
        **custom_metadata
    }
    
    logger.info(f"Starting batch {params.batch_id} with {len(params.models)} models and {len(params.prompts)} prompts")
    
    # Process each model
    for model_name in params.models:
        model_responses: List[LLMResponse] = []
        results[model_name] = model_responses
        
        logger.info(f"Querying model: {model_name}")
        
        # Process each prompt for the current model
        for prompt in params.prompts:
            try:
                # Call the model using litellm
                logger.debug(f"Sending prompt to {model_name}: {prompt[:50]}...")
                
                response = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=params.max_tokens,
                    temperature=params.temperature
                )
                
                # Extract the response text with safeguards
                response_text = ""
                if (response is not None and 
                    hasattr(response, 'choices') and 
                    response.choices and 
                    hasattr(response.choices[0], 'message') and
                    hasattr(response.choices[0].message, 'content')):
                    response_text = response.choices[0].message.content
                
                # Create a structured response object
                llm_response = LLMResponse(
                    model=model_name,
                    prompt=prompt,
                    response_text=response_text,
                    timestamp=datetime.now(),
                    metadata=metadata.copy(),
                    raw_response=response
                )
                
                # Add to the list of responses for this model
                model_responses.append(llm_response)
                logger.debug(f"Received response of length {len(response_text)}")
                
            except Exception as e:
                logger.error(f"Error querying {model_name}: {str(e)}")
                
                # Create an error response object
                error_response = LLMResponse(
                    model=model_name,
                    prompt=prompt,
                    response_text=f"Error: {str(e)}",
                    timestamp=datetime.now(),
                    metadata=metadata.copy(),
                    error=str(e)
                )
                
                model_responses.append(error_response)
    
    # Verify results are complete
    assert len(results) == len(params.models), "Results dictionary should contain entries for all models"
    for model, responses in results.items():
        assert len(responses) == len(params.prompts), f"Model {model} should have responses for all prompts"
    
    logger.info(f"Completed batch {params.batch_id}")
    return results


def flatten_model_responses(model_responses: Dict[str, List[LLMResponse]]) -> Dict[str, List[str]]:
    """
    Flatten structured LLMResponse objects into simple text responses.
    
    This is useful for compatibility with other functions that expect
    a simple dictionary of model names to lists of text responses.
    
    Args:
        model_responses: Dictionary mapping model names to lists of LLMResponse objects.
        
    Returns:
        Dictionary mapping model names to lists of response text strings.
    """
    flattened = {}
    for model, responses in model_responses.items():
        flattened[model] = [r.response_text for r in responses]
    return flattened


def get_batch_summary(model_responses: Dict[str, List[LLMResponse]]) -> Dict[str, Any]:
    """
    Generate a summary of a batch of model responses.
    
    Args:
        model_responses: Dictionary mapping model names to lists of LLMResponse objects.
        
    Returns:
        Dictionary containing summary statistics for the batch.
    """
    if not model_responses:
        return {}
    
    # Get batch_id from the first response
    first_model = next(iter(model_responses))
    if not model_responses[first_model]:
        return {}
    
    batch_id = model_responses[first_model][0].metadata.get("batch_id", "unknown")
    
    total_responses = sum(len(responses) for responses in model_responses.values())
    error_count = sum(
        1 for responses in model_responses.values() 
        for response in responses if response.error is not None
    )
    
    # Calculate average response length per model
    avg_lengths = {}
    for model, responses in model_responses.items():
        if responses:
            avg_lengths[model] = sum(len(r.response_text) for r in responses) / len(responses)
    
    return {
        "batch_id": batch_id,
        "total_models": len(model_responses),
        "total_responses": total_responses,
        "error_count": error_count,
        "success_rate": (total_responses - error_count) / total_responses if total_responses > 0 else 0,
        "average_response_lengths": avg_lengths,
        "timestamp": datetime.now().isoformat()
    }