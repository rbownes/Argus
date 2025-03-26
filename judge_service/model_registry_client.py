"""
Model Registry service client for integration with judge service.
"""
import os
import logging
import json
from typing import Dict, List, Optional, Any
from shared.service_client import ServiceClient
from shared.utils import ApiError

class ModelRegistryClient:
    """Client for Model Registry service."""
    
    def __init__(self):
        """Initialize client with service URL from environment."""
        base_url = os.environ.get("MODEL_REGISTRY_URL", "http://model-registry:8000")
        self.client = ServiceClient(base_url)
        self.logger = logging.getLogger("model_registry_client")
    
    async def get_models(self) -> List[Dict]:
        """
        Get all available models.
        
        Returns:
            List of models
        """
        try:
            self.logger.info("Getting all models from registry")
            response = await self.client.get("/api/v1/models")
            return response.get("data", [])
        except Exception as e:
            self.logger.error(f"Error getting models: {str(e)}")
            raise ApiError(status_code=500, message=f"Failed to retrieve models: {str(e)}")
    
    async def get_model(self, model_id: str) -> Optional[Dict]:
        """
        Get model by ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model data or None if not found
        """
        try:
            self.logger.info(f"Getting model by ID: {model_id}")
            response = await self.client.get(f"/api/v1/models/{model_id}")
            return response.get("data")
        except Exception as e:
            self.logger.error(f"Error getting model {model_id}: {str(e)}")
            return None
    
    async def get_providers(self) -> List[Dict]:
        """
        Get all providers.
        
        Returns:
            List of providers
        """
        try:
            self.logger.info("Getting all providers from registry")
            response = await self.client.get("/api/v1/providers")
            return response.get("data", [])
        except Exception as e:
            self.logger.error(f"Error getting providers: {str(e)}")
            raise ApiError(status_code=500, message=f"Failed to retrieve providers: {str(e)}")
    
    async def complete(
        self,
        model_id: str,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """
        Generate a completion using the model registry.
        
        Args:
            model_id: ID of the model to use
            messages: List of messages in the format [{"role": "user", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with completion results
        """
        try:
            self.logger.info(f"Generating completion using model {model_id}")
            
            # Prepare the request
            request_data = {
                "model_id": model_id,
                "messages": messages
            }
            
            # Add optional parameters if provided
            if temperature is not None:
                request_data["temperature"] = temperature
            
            if max_tokens is not None:
                request_data["max_tokens"] = max_tokens
            
            # Check if this is a Gemini model and use direct client if needed
            if "gemini" in model_id.lower():
                self.logger.info(f"Using direct Gemini client for model: {model_id}")
                try:
                    # Import the direct client
                    from .direct_gemini_client import DirectGeminiClient
                    
                    # Initialize and use the direct client
                    gemini_client = DirectGeminiClient()
                    response_data = await gemini_client.generate_content(
                        model_id=model_id,
                        messages=messages,
                        temperature=temperature or 0.7,
                        max_tokens=max_tokens or 1024
                    )
                    
                    return response_data
                except Exception as e:
                    self.logger.error(f"Error using direct Gemini client: {str(e)}")
                    raise ApiError(
                        status_code=500, 
                        message=f"Failed to generate completion with direct Gemini client: {str(e)}"
                    )
            
            # For non-Gemini models, use the regular Model Registry service
            # Make the POST request
            response = await self.client.post("/api/v1/completion", data=request_data)
            return response.get("data", {})
        except Exception as e:
            self.logger.error(f"Error generating completion with model {model_id}: {str(e)}")
            raise ApiError(
                status_code=500, 
                message=f"Failed to generate completion: {str(e)}"
            )
