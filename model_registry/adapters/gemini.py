"""
Adapter for Google's Gemini API.
"""
from .base import ModelAdapter
from typing import Dict, List, Optional, Any
import aiohttp
import uuid
import json
import logging

class GeminiAdapter(ModelAdapter):
    """Adapter for Google's Gemini API."""
    
    async def complete(
        self, 
        model_id: str,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion using Gemini API.
        
        Args:
            model_id: ID of the model to use
            messages: List of message dictionaries with 'role' and 'content'
            model_config: Model-specific configuration
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with completion results
        """
        if not self.api_key:
            raise ValueError(f"API key not found in environment variable {self.provider_config['env_var_key']}")
        
        # Format messages for Gemini API
        formatted_messages = self._format_messages_for_gemini(messages)
        
        # Build request payload
        payload = {
            "contents": formatted_messages,
            "generationConfig": {
                "temperature": temperature or model_config.get("temperature_default", 0.7),
                "maxOutputTokens": max_tokens or model_config.get("max_tokens_default", 1024),
                "topP": 0.95,
                "topK": 40
            }
        }
        
        # Determine the API endpoint format from model config or use default
        api_format = model_config.get("api_format", "models/{model_id}:generateContent")
        endpoint = api_format.format(model_id=model_id)
        
        # Ensure URL has proper structure
        if not endpoint.startswith("http"):
            base_url = self.provider_config.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
            url = f"{base_url}/{endpoint}"
        else:
            url = endpoint
        
        # Add API key as query parameter
        if "?" in url:
            url += f"&key={self.api_key}"
        else:
            url += f"?key={self.api_key}"
            
        self.logger.info(f"Making request to Gemini API: {url}")
        
        # Make the API request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    response_data = await response.json()
                    
                    if response.status != 200:
                        error_msg = response_data.get("error", {}).get("message", "Unknown error")
                        self.logger.error(f"Gemini API error ({response.status}): {error_msg}")
                        raise ValueError(f"Gemini API error: {error_msg}")
                    
                    # Extract the generated content
                    try:
                        content = response_data["candidates"][0]["content"]["parts"][0]["text"]
                        
                        # Create usage stats (Gemini doesn't provide these directly)
                        usage = {
                            "prompt_tokens": sum(len(m.get("content", "")) for m in messages) // 4,  # Rough estimate
                            "completion_tokens": len(content) // 4,  # Rough estimate
                            "total_tokens": 0  # Will be calculated below
                        }
                        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                        
                        return {
                            "id": str(uuid.uuid4()),
                            "model_id": model_id,
                            "provider_id": self.provider_config["id"],
                            "content": content,
                            "usage": usage,
                            "raw_response": response_data
                        }
                    except (KeyError, IndexError) as e:
                        self.logger.error(f"Error parsing Gemini response: {str(e)}, Response: {response_data}")
                        raise ValueError(f"Error parsing Gemini response: {str(e)}")
        except aiohttp.ClientError as e:
            self.logger.error(f"Gemini API request error: {str(e)}")
            raise ValueError(f"Failed to connect to Gemini API: {str(e)}")
    
    def _format_messages_for_gemini(self, messages: List[Dict[str, str]]) -> List[Dict]:
        """
        Format messages for Gemini API structure.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Gemini-formatted message list
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user").lower()
            
            # Map OpenAI roles to Gemini roles
            if role == "system":
                # Gemini doesn't have system messages, convert to user
                formatted.append({
                    "role": "user",
                    "parts": [{"text": msg.get("content", "")}]
                })
            elif role == "assistant":
                formatted.append({
                    "role": "model",
                    "parts": [{"text": msg.get("content", "")}]
                })
            else:
                # Default to user role
                formatted.append({
                    "role": "user",
                    "parts": [{"text": msg.get("content", "")}]
                })
                
        return formatted
