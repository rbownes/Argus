"""
Shared HTTP client for inter-service communication.
"""
import os
import aiohttp
import json
import logging
from typing import Dict, Any, Optional

class ServiceClient:
    """HTTP client for inter-service communication."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize service client.
        
        Args:
            base_url: Base URL for the service
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key
        }
        self.logger = logging.getLogger("service_client")
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make GET request to service.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"
        self.logger.info(f"Making GET request to {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make POST request to service.
        
        Args:
            endpoint: API endpoint
            data: Request data
            
        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"
        self.logger.info(f"Making POST request to {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()
