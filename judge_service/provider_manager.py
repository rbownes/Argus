"""
Provider Manager for handling direct API integrations with LLM providers.
"""
import os
import logging
import json
from typing import Dict, List, Optional, Any, Type, Union
from datetime import datetime
import uuid
import re

from .clients.base_client import BaseProviderClient
from .clients.openai_client import OpenAIClient
from .clients.anthropic_client import AnthropicClient
from .clients.gemini_client import GeminiClient
from shared.utils import ApiError

class ProviderManager:
    """
    Manager for LLM provider clients.
    Handles routing requests to the appropriate provider based on model ID.
    """
    
    def __init__(self):
        """Initialize the provider manager with client instances."""
        self.logger = logging.getLogger("provider_manager")
        
        # Initialize clients
        self.openai_client = OpenAIClient()
        self.anthropic_client = AnthropicClient()
        self.gemini_client = GeminiClient()
        
        # Load model configurations
        self.models = self._load_model_config()
        
        self.logger.info("Provider Manager initialized")
        
    def _load_model_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Load model configurations from the models.json file.
        
        Returns:
            Dictionary of model configurations
        """
        models_config_path = os.path.join(os.path.dirname(__file__), "config", "models.json")
        
        try:
            if os.path.exists(models_config_path):
                with open(models_config_path, "r") as f:
                    config = json.load(f)
                    return {model["id"]: model for model in config.get("models", [])}
            else:
                # Return default models if config doesn't exist
                default_models = [
                    {"id": "gpt-4", "name": "GPT-4", "provider": "openai", "is_judge_compatible": True},
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai", "is_judge_compatible": True},
                    {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "provider": "anthropic", "is_judge_compatible": True},
                    {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet", "provider": "anthropic", "is_judge_compatible": True},
                    {"id": "gemini-pro", "name": "Gemini Pro", "provider": "gemini", "is_judge_compatible": False},
                    {"id": "chatgpt-4o-latest", "name": "ChatGPT-4o (Latest)", "provider": "openai", "is_judge_compatible": True}
                ]
                return {model["id"]: model for model in default_models}
        except Exception as e:
            self.logger.error(f"Error loading model configuration: {str(e)}")
            # Return empty dict as fallback
            return {}
        
    def _detect_provider_from_id(self, model_id: str) -> str:
        """
        Attempt to infer the provider from the model ID.
        
        Args:
            model_id: The model identifier
            
        Returns:
            The inferred provider name, or "unknown" if it can't be determined
        """
        model_id = model_id.lower()
        
        # Check if model is in our configuration
        if model_id in self.models:
            return self.models[model_id]["provider"]
        
        # Otherwise, try to infer from model name patterns
        if any(name in model_id for name in ["gpt", "davinci", "chatgpt", "openai", "text-embedding"]):
            return "openai"
        elif any(name in model_id for name in ["claude", "anthropic"]):
            return "anthropic"
        elif any(name in model_id for name in ["gemini", "palm", "bard", "google"]):
            return "gemini"
        else:
            return "unknown"
            
    def _get_client_for_provider(self, provider: str) -> BaseProviderClient:
        """
        Get the appropriate client for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider client instance
            
        Raises:
            ValueError: If provider is unknown or not supported
        """
        provider = provider.lower()
        
        if provider in ["openai", "azure"]:
            return self.openai_client
        elif provider in ["anthropic", "claude"]:
            return self.anthropic_client
        elif provider in ["gemini", "google"]:
            return self.gemini_client
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
    def register_model(self, model_data: Dict[str, Any]) -> bool:
        """
        Register a new model in the models.json configuration file.
        
        Args:
            model_data: Dictionary containing the model information
            
        Returns:
            True if the model was registered successfully, False otherwise
        """
        try:
            models_config_path = os.path.join(os.path.dirname(__file__), "config", "models.json")
            
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(models_config_path), exist_ok=True)
            
            # Load existing config or create a new one
            if os.path.exists(models_config_path):
                with open(models_config_path, "r") as f:
                    config = json.load(f)
            else:
                config = {"models": []}
            
            # Check if model already exists
            model_exists = False
            for i, existing_model in enumerate(config["models"]):
                if existing_model["id"] == model_data["id"]:
                    # Update existing model
                    config["models"][i] = model_data
                    model_exists = True
                    break
            
            # Add new model if it doesn't exist
            if not model_exists:
                config["models"].append(model_data)
            
            # Save updated config
            with open(models_config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            # Update local model cache
            self.models[model_data["id"]] = model_data
            
            return True
        except Exception as e:
            self.logger.error(f"Error registering new model: {str(e)}")
            return False
            
    async def complete(
        self,
        model_id: str,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion using the appropriate provider.
        
        Args:
            model_id: ID of the model to use
            messages: List of messages for the conversation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with completion results
            
        Raises:
            ApiError: If an error occurs during completion
        """
        try:
            # Determine provider from model ID
            provider = self._detect_provider_from_id(model_id)
            
            if provider == "unknown":
                raise ValueError(f"Unknown model: {model_id}. Cannot determine provider.")
                
            self.logger.info(f"Routing completion request for model {model_id} to {provider} provider")
            
            # Get the appropriate client
            client = self._get_client_for_provider(provider)
            
            # Generate content
            response = await client.generate_content(
                model_id=model_id,
                messages=messages,
                temperature=temperature or 0.7,
                max_tokens=max_tokens or 1024
            )
            
            return response
        except Exception as e:
            self.logger.error(f"Error in provider manager completion: {str(e)}")
            raise ApiError(status_code=500, message=f"Provider error: {str(e)}")
            
    async def get_models(self) -> List[Dict]:
        """
        Get all available models.
        
        Returns:
            List of models
        """
        try:
            models_list = list(self.models.values())
            return models_list
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
            return self.models.get(model_id)
        except Exception as e:
            self.logger.error(f"Error getting model {model_id}: {str(e)}")
            return None
