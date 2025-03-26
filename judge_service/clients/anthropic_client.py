"""
Direct client for Anthropic API.
"""
import os
import json
import aiohttp
from typing import Dict, List, Optional, Any
import uuid
import logging
from datetime import datetime

from .base_client import BaseProviderClient

class AnthropicClient(BaseProviderClient):
    """Direct client for Anthropic API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Anthropic client.
        
        Args:
            api_key: Anthropic API key (optional, will use env var if not provided)
        """
        super().__init__(api_key=api_key, api_key_env_var="ANTHROPIC_API_KEY")
        
        self.base_url = "https://api.anthropic.com/v1"
        self.anthropic_version = "2023-06-01" # Current API version as of now, can be updated
        
    async def generate_content(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate content using Anthropic API.
        
        Args:
            model_id: Model ID (e.g., "claude-3-opus-20240229")
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with response content
        """
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        # Format messages for Anthropic API
        formatted_messages = self.format_messages(messages)
        
        # Build request payload
        payload = {
            "model": model_id,
            "messages": formatted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload and value is not None:
                payload[key] = value
        
        # URL for chat completions
        url = f"{self.base_url}/messages"
        
        self.logger.info(f"Making request to Anthropic API for model: {model_id}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": self.anthropic_version,
                        "Content-Type": "application/json"
                    }
                ) as response:
                    response_text = await response.text()
                    
                    if response.status != 200:
                        self.logger.error(f"Anthropic API error ({response.status}): {response_text}")
                        
                        try:
                            response_data = json.loads(response_text)
                            error_type = response_data.get("error", {}).get("type", "Unknown error")
                            error_msg = response_data.get("error", {}).get("message", "Unknown error")
                            full_error = f"{error_type}: {error_msg}"
                        except:
                            full_error = response_text
                            
                        raise ValueError(f"Anthropic API error: {full_error}")
                        
                    response_data = json.loads(response_text)
                    
                    # Extract the generated content from the first message
                    content = response_data["content"][0]["text"]
                    
                    # Create usage statistics (Anthropic provides usage info in response)
                    # If usage info isn't in the response, we'll estimate it
                    if "usage" in response_data:
                        usage = {
                            "prompt_tokens": response_data["usage"]["input_tokens"],
                            "completion_tokens": response_data["usage"]["output_tokens"],
                            "total_tokens": response_data["usage"]["input_tokens"] + response_data["usage"]["output_tokens"]
                        }
                    else:
                        # Estimate usage based on text length (roughly 4 chars per token)
                        input_text = "".join(msg.get("content", "") for msg in messages)
                        prompt_tokens = len(input_text) // 4
                        completion_tokens = len(content) // 4
                        
                        usage = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    
                    # Create and return standardized response
                    return self.create_response_object(
                        model_id=model_id,
                        content=content,
                        usage=usage,
                        provider="anthropic",
                        response_id=response_data.get("id")
                    )
        except aiohttp.ClientError as e:
            self.logger.error(f"Anthropic API request error: {str(e)}")
            raise ValueError(f"Failed to connect to Anthropic API: {str(e)}")
            
    def format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format messages for Anthropic API.
        Anthropic expects messages with 'role' and 'content' where role is one of:
        'user', 'assistant', or 'system' (only one system message is allowed and it should be first).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted messages for Anthropic API
        """
        formatted_messages = []
        system_message = None
        
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            
            # Handle system message (Claude can only have one system message)
            if role == "system":
                # Save the system message to add it at the beginning
                if system_message is None:
                    system_message = {"role": "system", "content": content}
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
            else:
                # Default to user role for any other role
                formatted_messages.append({"role": "user", "content": content})
        
        # Add system message at the beginning if it exists
        if system_message is not None:
            formatted_messages.insert(0, system_message)
            
        return formatted_messages
