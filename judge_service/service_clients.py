"""
Service client wrappers for inter-service communication.
"""
import os
import logging
from typing import Dict, List, Optional, Any
from shared.service_client import ServiceClient
from shared.utils import ApiError

class QueryStorageClient:
    """Client for Query Storage service."""
    
    def __init__(self):
        """Initialize client with service URL from environment."""
        base_url = os.environ.get("QUERY_STORAGE_URL", "http://item-storage-queries:8000")
        self.client = ServiceClient(base_url)
        self.logger = logging.getLogger("query_storage_client")
    
    async def get_queries_by_theme(self, theme: str, limit: int = 10) -> List[Dict]:
        """
        Get queries by theme.
        
        Args:
            theme: Theme to filter by
            limit: Maximum number of results
            
        Returns:
            List of queries
        """
        try:
            self.logger.info(f"Getting queries for theme: {theme}")
            response = await self.client.get(
                f"/api/v1/queries/theme/{theme}",
                params={"limit": limit, "page": 1}
            )
            return response.get("data", {}).get("items", [])
        except Exception as e:
            self.logger.error(f"Error getting queries by theme: {str(e)}")
            raise ApiError(status_code=500, message=f"Failed to retrieve queries: {str(e)}")

class EvaluationStorageClient:
    """Client for Evaluation Storage service."""
    
    def __init__(self):
        """Initialize client with service URL from environment."""
        base_url = os.environ.get("EVALUATION_STORAGE_URL", "http://item-storage-metrics:8000")
        self.client = ServiceClient(base_url)
        self.logger = logging.getLogger("evaluation_storage_client")
    
    async def get_metric_by_id(self, metric_id: str) -> Optional[Dict]:
        """
        Get evaluation metric by ID.
        
        Args:
            metric_id: ID of the metric or metric type
            
        Returns:
            Metric data or None if not found
        """
        self.logger.info(f"Getting evaluation metric: {metric_id}")
        
        try:
            # Try to get by ID directly
            response = await self.client.get(f"/api/v1/evaluation-metrics/{metric_id}")
            return response.get("data")
        except Exception as e:
            self.logger.info(f"Metric not found by ID, trying as metric type: {str(e)}")
            
            try:
                # Try as metric type
                response = await self.client.get(
                    f"/api/v1/evaluation-metrics/type/{metric_id}",
                    params={"limit": 1}
                )
                items = response.get("data", {}).get("items", [])
                if not items:
                    self.logger.warning(f"No metrics found for type: {metric_id}")
                    return None
                    
                return items[0]
            except Exception as e2:
                self.logger.error(f"Error getting metric by type: {str(e2)}")
                raise ApiError(status_code=500, message=f"Failed to retrieve evaluation metric: {str(e2)}")
