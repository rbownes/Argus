"""
API for the Model Registry service.
"""
from fastapi import Depends, Query, Body, HTTPException
from typing import Dict, List, Optional, Any
import os
import importlib
import logging
from datetime import datetime
import json
import traceback

from shared.utils import create_api_app, ApiResponse, ResponseStatus, ApiError, setup_logging

# Call logging setup early
setup_logging(log_level=os.environ.get("LOG_LEVEL", "INFO"))
from shared.middleware import add_middleware
from .models import ProviderConfig, ModelConfig, CompletionRequest, CompletionResponse
from .registry_storage import ModelRegistryStorage

# Create FastAPI app
app = create_api_app(
    title="Model Registry Service",
    description="Central registry for LLM models and providers",
    version="0.1.0"
)

# Initialize configuration
api_key = os.environ.get("API_KEY")
add_middleware(app, api_key=api_key)

# Initialize storage
storage = ModelRegistryStorage()

# Load default configurations at startup
@app.on_event("startup")
async def load_defaults():
    """Load default models and providers at startup."""
    try:
        await storage._load_defaults()
        logger.info("Loaded default models and providers")
    except Exception as e:
        logger.error(f"Error loading default configurations: {str(e)}")

# Dictionary to cache adapter instances
adapter_instances = {}

# Logger
logger = logging.getLogger("model_registry")

@app.get("/api/v1/models", response_model=ApiResponse, tags=["Models"])
async def list_models():
    """List all registered models."""
    try:
        models = await storage.get_all_models()
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=models
        )
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise ApiError(status_code=500, message=str(e))

@app.get("/api/v1/models/{model_id}", response_model=ApiResponse, tags=["Models"])
async def get_model(model_id: str):
    """Get model by ID."""
    try:
        model = await storage.get_model(model_id)
        if not model:
            raise ApiError(status_code=404, message=f"Model not found: {model_id}")
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=model
        )
    except ApiError:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {str(e)}")
        raise ApiError(status_code=500, message=str(e))

@app.post("/api/v1/models", response_model=ApiResponse, tags=["Models"])
async def add_model(model_config: ModelConfig):
    """Add a new model to the registry."""
    try:
        # Check if provider exists
        provider = await storage.get_provider(model_config.provider_id)
        if not provider:
            raise ApiError(status_code=404, message=f"Provider not found: {model_config.provider_id}")
        
        # Add the model
        model = await storage.add_model(model_config.dict())
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Model {model_config.id} added successfully",
            data=model
        )
    except ApiError:
        raise
    except Exception as e:
        logger.error(f"Error adding model: {str(e)}")
        raise ApiError(status_code=500, message=str(e))

@app.get("/api/v1/providers", response_model=ApiResponse, tags=["Providers"])
async def list_providers():
    """List all registered providers."""
    try:
        providers = await storage.get_all_providers()
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=providers
        )
    except Exception as e:
        logger.error(f"Error listing providers: {str(e)}")
        raise ApiError(status_code=500, message=str(e))

@app.get("/api/v1/providers/{provider_id}", response_model=ApiResponse, tags=["Providers"])
async def get_provider(provider_id: str):
    """Get provider by ID."""
    try:
        provider = await storage.get_provider(provider_id)
        if not provider:
            raise ApiError(status_code=404, message=f"Provider not found: {provider_id}")
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            data=provider
        )
    except ApiError:
        raise
    except Exception as e:
        logger.error(f"Error getting provider {provider_id}: {str(e)}")
        raise ApiError(status_code=500, message=str(e))

@app.post("/api/v1/providers", response_model=ApiResponse, tags=["Providers"])
async def add_provider(provider_config: ProviderConfig):
    """Add a new provider to the registry."""
    try:
        provider = await storage.add_provider(provider_config.dict())
        
        return ApiResponse(
            status=ResponseStatus.SUCCESS,
            message=f"Provider {provider_config.id} added successfully",
            data=provider
        )
    except Exception as e:
        logger.error(f"Error adding provider: {str(e)}")
        raise ApiError(status_code=500, message=str(e))

@app.post("/api/v1/completion", response_model=ApiResponse, tags=["Completion"])
async def generate_completion(request: CompletionRequest):
    """Generate a completion using the specified model."""
    try:
        # Get model information
        model = await storage.get_model(request.model_id)
        if not model:
            raise ApiError(status_code=404, message=f"Model not found: {request.model_id}")
        
        # Get provider information
        provider = await storage.get_provider(model["provider_id"])
        if not provider:
            raise ApiError(status_code=404, message=f"Provider not found: {model['provider_id']}")
        
        # Get or create adapter instance
        adapter = await get_adapter_instance(provider)
        
        # Generate completion
        try:
            completion = await adapter.complete(
                model_id=model["id"],
                messages=request.messages,
                model_config=model["config"],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Create response object
            response = CompletionResponse(
                id=completion["id"],
                model_id=model["id"],
                provider_id=provider["id"],
                content=completion["content"],
                usage=completion["usage"],
                created_at=datetime.utcnow()
            )
            
            # Log the completion
            await storage.log_completion(
                model_id=model["id"],
                query=json.dumps(request.messages),
                response=completion["content"],
                metadata={
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "usage": completion["usage"]
                }
            )
            
            return ApiResponse(
                status=ResponseStatus.SUCCESS,
                data=response.dict()
            )
        except Exception as e:
            # Log the error
            error_msg = str(e)
            trace = traceback.format_exc()
            logger.error(f"Completion error: {error_msg}\n{trace}")
            
            await storage.log_completion(
                model_id=model["id"],
                query=json.dumps(request.messages),
                error=error_msg,
                metadata={
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "traceback": trace
                }
            )
            
            raise ApiError(status_code=500, message=f"Completion error: {error_msg}")
            
    except ApiError:
        raise
    except Exception as e:
        logger.error(f"Error in completion endpoint: {str(e)}")
        raise ApiError(status_code=500, message=str(e))

async def get_adapter_instance(provider: Dict[str, Any]):
    """
    Get or create an adapter instance for a provider.
    
    Args:
        provider: Provider configuration
        
    Returns:
        Adapter instance
    """
    provider_id = provider["id"]
    
    # Check if adapter instance already exists
    if provider_id in adapter_instances:
        return adapter_instances[provider_id]
    
    try:
        # Import the adapter module
        adapter_name = provider["adapter"]
        module_path = f".adapters.{adapter_name}" if not "." in adapter_name else adapter_name
        module = importlib.import_module(module_path, package="model_registry")
        
        # Find the adapter class (assumed to be the only class that extends ModelAdapter)
        adapter_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr.__name__.endswith("Adapter"):
                adapter_class = attr
                break
                
        if not adapter_class:
            raise ValueError(f"No adapter class found in module {module_path}")
            
        # Create and cache the adapter instance
        adapter = adapter_class(provider)
        adapter_instances[provider_id] = adapter
        
        logger.info(f"Created adapter instance for provider {provider_id} using {adapter_class.__name__}")
        
        return adapter
    except Exception as e:
        logger.error(f"Error creating adapter for provider {provider_id}: {str(e)}")
        raise

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "model-registry"}
