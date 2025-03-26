"""
Direct client for OpenAI API.
"""
import os
import json
import aiohttp
from typing import Dict, List, Optional, Any
import uuid
import logging
from datetime import datetime

from .base_client import BaseProviderClient

class OpenAIClient(BaseProviderClient):
    """Direct client for OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        super().__init__(api_key=api_key, api_key_env_var="OPENAI_API_KEY")
        
        self.base_url = "https://api.openai.com/v1"
        
    async def generate_content(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using OpenAI API.
        
        Args:
            model_id: Model ID (e.g., "gpt-4", "gpt-3.5-turbo")
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with response content
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Format messages (OpenAI already uses the standard format)
        formatted_messages = self.format_messages(messages)
        
        # Build request payload
        payload = {
            "model": model_id,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload and value is not None:
                payload[key] = value
        
        # URL for chat completions
        url = f"{self.base_url}/chat/completions"
        
        self.logger.info(f"Making request to OpenAI API for model: {model_id}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    response_text = await response.text()
                    
                    if response.status != 200:
                        self.logger.error(f"OpenAI API error ({response.status}): {response_text}")
                        
                        try:
                            response_data = json.loads(response_text)
                            error_msg = response_data.get("error", {}).get("message", "Unknown error")
                        except:
                            error_msg = response_text
                            
                        raise ValueError(f"OpenAI API error: {error_msg}")
                        
                    response_data = json.loads(response_text)
                    
                    # Extract the generated content
                    content = response_data["choices"][0]["message"]["content"]
                    
                    # Extract usage statistics
                    usage = {
                        "prompt_tokens": response_data["usage"]["prompt_tokens"],
                        "completion_tokens": response_data["usage"]["completion_tokens"],
                        "total_tokens": response_data["usage"]["total_tokens"]
                    }
                    
                    # Create and return standardized response
                    return self.create_response_object(
                        model_id=model_id,
                        content=content,
                        usage=usage,
                        provider="openai",
                        response_id=response_data.get("id")
                    )
        except aiohttp.ClientError as e:
            self.logger.error(f"OpenAI API request error: {str(e)}")
            raise ValueError(f"Failed to connect to OpenAI API: {str(e)}")
            
    def format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format messages for OpenAI API.
        OpenAI already uses the standard format with 'role' and 'content' keys.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted messages for OpenAI API
        """
        # Validate and ensure correct format
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user").lower()
            
            # Ensure role is one of: 'system', 'user', 'assistant'
            if role not in ["system", "user", "assistant"]:
                role = "user"  # Default to user if unknown role
                
            formatted_messages.append({
                "role": role,
                "content": msg.get("content", "")
            })
                
        return formatted_messages
