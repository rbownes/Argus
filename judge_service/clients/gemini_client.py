"""
Direct client for Google's Gemini API (Compatible with base provider interface).
"""
import os
import json
import aiohttp
from typing import Dict, List, Optional, Any
import uuid
import logging
from datetime import datetime

from .base_client import BaseProviderClient

class GeminiClient(BaseProviderClient):
    """Direct client for Google's Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google API key (optional, will use env var if not provided)
        """
        super().__init__(api_key=api_key, api_key_env_var="GOOGLE_API_KEY")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
    async def generate_content(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini models.
        
        Args:
            model_id: Model ID (e.g., "gemini-pro", "gemini-pro-vision")
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with response content
        """
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        # Format messages for Gemini API
        formatted_messages = self.format_messages(messages)
        
        # Build request payload
        payload = {
            "contents": formatted_messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        # Add any additional parameters from kwargs to generation config
        for key, value in kwargs.items():
            if key not in payload["generationConfig"] and value is not None:
                payload["generationConfig"][key] = value
        
        # Format the URL
        url = f"{self.base_url}/models/{model_id}:generateContent?key={self.api_key}"
        
        self.logger.info(f"Making request to Gemini API for model: {model_id}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response_text = await response.text()
                    
                    if response.status != 200:
                        self.logger.error(f"Gemini API error ({response.status}): {response_text}")
                        
                        try:
                            response_data = json.loads(response_text)
                            error_msg = response_data.get("error", {}).get("message", "Unknown error")
                        except:
                            error_msg = response_text
                            
                        raise ValueError(f"Gemini API error: {error_msg}")
                        
                    response_data = json.loads(response_text)
                    
                    # Extract the generated content
                    content = response_data["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # Create usage stats (Gemini doesn't provide these directly)
                    usage = {
                        "prompt_tokens": sum(len(m.get("content", "")) for m in messages) // 4,  # Rough estimate
                        "completion_tokens": len(content) // 4,  # Rough estimate
                        "total_tokens": 0  # Will be calculated below
                    }
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    
                    # Create and return standardized response
                    return self.create_response_object(
                        model_id=model_id,
                        content=content,
                        usage=usage,
                        provider="gemini",
                        response_id=str(uuid.uuid4())  # Gemini doesn't provide a response ID
                    )
        except aiohttp.ClientError as e:
            self.logger.error(f"Gemini API request error: {str(e)}")
            raise ValueError(f"Failed to connect to Gemini API: {str(e)}")
            
    def format_messages(self, messages: List[Dict[str, str]]) -> List[Dict]:
        """
        Format messages for Gemini API.
        Gemini uses a different format with 'role' and 'parts'.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted messages for Gemini API
        """
        formatted_messages = []
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            # Map roles to Gemini format
            if role == "system":
                # Gemini doesn't have system messages, convert to user
                formatted_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                formatted_messages.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
            else:
                # Default to user role
                formatted_messages.append({
                    "role": "user", 
                    "parts": [{"text": content}]
                })
                
        return formatted_messages
