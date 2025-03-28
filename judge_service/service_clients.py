"""
Service client wrappers for inter-service communication.
"""
import os
import logging
from typing import Dict, List, Optional, Any
from shared.service_client import ServiceClient
from shared.utils import ApiError

class ItemStorageClient:
    """Client for Item Storage service."""
    
    def __init__(self, service_type="queries"):
        """
        Initialize client with service URL from environment.
        
        Args:
            service_type: Type of item storage ("queries" or "metrics")
        """
        if service_type == "queries":
            base_url = os.environ.get("QUERY_STORAGE_URL", "http://item-storage-queries:8000")
            self.logger = logging.getLogger("query_storage_client")
        elif service_type == "metrics":
            base_url = os.environ.get("EVALUATION_STORAGE_URL", "http://item-storage-metrics:8000")
            self.logger = logging.getLogger("evaluation_storage_client")
        else:
            raise ValueError(f"Unknown service type: {service_type}")
            
        self.client = ServiceClient(base_url)
        self.service_type = service_type
    
    async def get_items_by_type(self, item_type: str, limit: int = 10) -> List[Dict]:
        """
        Get items by type.
        
        Args:
            item_type: Type or category of the items
            limit: Maximum number of results
            
        Returns:
            List of items
        """
        try:
            self.logger.info(f"Getting {self.service_type} for type: {item_type}")
            response = await self.client.get(
                f"/api/v1/items/type/{item_type}",
                params={"limit": limit, "page": 1}
            )
            return response.get("data", {}).get("items", [])
        except Exception as e:
            self.logger.error(f"Error getting {self.service_type} by type: {str(e)}")
            raise ApiError(status_code=500, message=f"Failed to retrieve {self.service_type}: {str(e)}")
    
    async def get_item_by_id(self, item_id: str) -> Optional[Dict]:
        """
        Get item by ID.
        
        Args:
            item_id: ID of the item
            
        Returns:
            Item data or None if not found
        """
        self.logger.info(f"Getting {self.service_type} item: {item_id}")
        
        try:
            response = await self.client.get(f"/api/v1/items/{item_id}")
            return response.get("data")
        except Exception as e:
            self.logger.error(f"Error getting {self.service_type} item: {str(e)}")
            raise ApiError(status_code=500, message=f"Failed to retrieve {self.service_type} item: {str(e)}")

# For backward compatibility
class QueryStorageClient(ItemStorageClient):
    """Client for Query Storage service."""
    
    def __init__(self):
        """Initialize client with service URL from environment."""
        super().__init__(service_type="queries")
    
    async def get_queries_by_theme(self, theme: str, limit: int = 10) -> List[Dict]:
        """
        Get queries by theme.
        
        Args:
            theme: Theme to filter by
            limit: Maximum number of results
            
        Returns:
            List of queries
        """
        return await self.get_items_by_type(theme, limit)

class EvaluationStorageClient(ItemStorageClient):
    """Client for Evaluation Storage service."""
    
    def __init__(self):
        """Initialize client with service URL from environment."""
        super().__init__(service_type="metrics")
    
    async def get_metric_by_id(self, metric_id: str) -> Optional[Dict]:
        """
        Get evaluation metric by ID.
        
        Args:
            metric_id: ID of the metric 
            
        Returns:
            Metric data or None if not found
        """
        try:
            # Try to get by ID directly
            return await self.get_item_by_id(metric_id)
        except Exception as e:
            self.logger.info(f"Metric not found by ID, trying as metric type: {str(e)}")
            
            try:
                # Try as metric type
                items = await self.get_items_by_type(metric_id, limit=1)
                if not items:
                    self.logger.warning(f"No metrics found for type: {metric_id}")
                    return None
                    
                return items[0]
            except Exception as e2:
                self.logger.error(f"Error getting metric by type: {str(e2)}")
                raise ApiError(status_code=500, message=f"Failed to retrieve evaluation metric: {str(e2)}")
