"""
Evaluation storage implementation using ChromaDB.
"""
import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict, Optional
import uuid
import logging

class EvaluationStorage:
    """Storage for evaluation metrics using ChromaDB vector database."""
    
    def __init__(self, persist_directory: str = "./evaluation_storage_data"):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.logger = logging.getLogger("evaluation_storage")
        
        try:
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
            self.collection = self.client.get_or_create_collection(
                name="evaluation_metrics",
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"ChromaDB collection 'evaluation_metrics' initialized at {persist_directory}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def store_evaluation_metric(self, prompt: str, metric_type: str, metadata: Optional[Dict] = None) -> str:
        """
        Store an evaluation metric with its type and optional metadata.
        
        Args:
            prompt: The evaluation prompt text
            metric_type: Type or category of the evaluation metric
            metadata: Additional metadata for the metric
            
        Returns:
            The metric ID (UUID)
        """
        try:
            metric_id = str(uuid.uuid4())
            
            # Prepare metadata
            metric_metadata = {
                "metric_type": metric_type,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
            
            # Store in ChromaDB
            self.collection.add(
                documents=[prompt],
                metadatas=[metric_metadata],
                ids=[metric_id]
            )
            
            self.logger.info(f"Stored evaluation metric with ID {metric_id} and type '{metric_type}'")
            return metric_id
        except Exception as e:
            self.logger.error(f"Failed to store evaluation metric: {str(e)}")
            raise

    def get_metrics_by_type(self, metric_type: str, limit: int = 10, skip: int = 0) -> List[Dict]:
        """
        Retrieve evaluation metrics by type with pagination.
        
        Args:
            metric_type: Type to filter by
            limit: Maximum number of results to return
            skip: Number of results to skip for pagination
            
        Returns:
            List of metrics with their metadata
        """
        try:
            # First get all IDs matching the metric type
            all_ids_result = self.collection.query(
                query_texts=[""], 
                where={"metric_type": metric_type},
                n_results=1000
            )
            
            if not all_ids_result["ids"][0]:
                return []
            
            # Apply pagination
            paginated_ids = all_ids_result["ids"][0][skip:skip+limit]
            
            if not paginated_ids:
                return []
            
            # Get full documents for paginated IDs
            results = []
            for i, id in enumerate(paginated_ids):
                # Get the document by ID
                get_result = self.collection.get(ids=[id])
                
                if get_result["ids"]:
                    results.append({
                        "id": get_result["ids"][0],
                        "prompt": get_result["documents"][0],
                        "metadata": get_result["metadatas"][0]
                    })
            
            self.logger.info(f"Retrieved {len(results)} metrics with type '{metric_type}'")
            return results
        except Exception as e:
            self.logger.error(f"Failed to get metrics by type: {str(e)}")
            raise

    def count_metrics_by_type(self, metric_type: str) -> int:
        """
        Count the number of metrics for a type.
        
        Args:
            metric_type: Type to count metrics for
            
        Returns:
            Number of metrics for the type
        """
        try:
            results = self.collection.query(
                query_texts=[""],
                where={"metric_type": metric_type},
                n_results=1000
            )
            
            count = len(results["ids"][0]) if results["ids"] and results["ids"][0] else 0
            self.logger.info(f"Counted {count} metrics with type '{metric_type}'")
            return count
        except Exception as e:
            self.logger.error(f"Failed to count metrics by type: {str(e)}")
            raise

    def search_similar_metrics(self, prompt: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar evaluation metrics using semantic similarity.
        
        Args:
            prompt: Prompt text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of similar metrics with their metadata and distance score
        """
        try:
            results = self.collection.query(
                query_texts=[prompt],
                n_results=limit
            )
            
            metrics = []
            for i in range(len(results["ids"][0])):
                metrics.append({
                    "id": results["ids"][0][i],
                    "prompt": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                })
            
            self.logger.info(f"Found {len(metrics)} similar metrics")
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to search similar metrics: {str(e)}")
            raise
            
    def get_metric_by_id(self, metric_id: str) -> Optional[Dict]:
        """
        Retrieve a metric by its ID.
        
        Args:
            metric_id: ID of the metric to retrieve
            
        Returns:
            Metric data or None if not found
        """
        try:
            result = self.collection.get(ids=[metric_id])
            
            if not result["ids"]:
                return None
                
            return {
                "id": result["ids"][0],
                "prompt": result["documents"][0],
                "metadata": result["metadatas"][0]
            }
        except Exception as e:
            self.logger.error(f"Failed to get metric by ID: {str(e)}")
            raise
