"""
Direct client for Google's Generative AI (Gemini) models.
"""
import os
import json
import aiohttp
from typing import List, Dict, Any, Optional
import uuid
import logging

class DirectGeminiClient:
    """Direct client for Google's Generative AI (Gemini) models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google API key (optional, will use env var if not provided)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.logger = logging.getLogger("direct_gemini_client")
        
    async def generate_content(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate content using Gemini models.
        
        Args:
            model_id: Model ID (e.g., "gemini-pro", "gemini-2.5-pro-exp-03-25")
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with response content
        """
        # Format messages for Gemini API
        gemini_messages = []
        for msg in messages:
            role = msg.get("role", "user").lower()
            
            # Map roles to Gemini format
            if role == "system":
                # Gemini doesn't have system messages, convert to user
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": msg.get("content", "")}]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": msg.get("content", "")}]
                })
            else:
                # Default to user role
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": msg.get("content", "")}]
                })
        
        # Build request payload
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 40
            }
        }
        
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
                    
                    # Create simulated usage stats since Gemini doesn't provide these
                    usage = {
                        "prompt_tokens": sum(len(m.get("content", "")) for m in messages) // 4,  # Rough estimate
                        "completion_tokens": len(content) // 4,  # Rough estimate
                        "total_tokens": 0  # Will be calculated below
                    }
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    
                    return {
                        "id": str(uuid.uuid4()),
                        "model_id": model_id,
                        "content": content,
                        "usage": usage,
                        "raw_response": response_data
                    }
        except aiohttp.ClientError as e:
            self.logger.error(f"Gemini API request error: {str(e)}")
            raise ValueError(f"Failed to connect to Gemini API: {str(e)}")
