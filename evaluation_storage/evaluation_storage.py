import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict, Optional
import uuid

class EvaluationStorage:
    def __init__(self, persist_directory: str = "./evaluation_data"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name="evaluation_metrics",
            metadata={"hnsw:space": "cosine"}
        )

    def store_evaluation_metric(self, prompt: str, metric_type: str, metadata: Optional[Dict] = None) -> str:
        """
        Store an evaluation metric/prompt with its type and optional metadata
        Returns the metric ID
        """
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
        
        return metric_id

    def get_metrics_by_type(self, metric_type: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve evaluation metrics by type
        Returns a list of metrics with their metadata
        """
        results = self.collection.query(
            query_texts=[""],  # Empty query to get all documents
            where={"metric_type": metric_type},
            n_results=limit
        )
        
        metrics = []
        for i in range(len(results['ids'][0])):
            metrics.append({
                "id": results['ids'][0][i],
                "prompt": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        
        return metrics

    def search_similar_metrics(self, prompt: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar evaluation metrics using semantic similarity
        """
        results = self.collection.query(
            query_texts=[prompt],
            n_results=limit
        )
        
        metrics = []
        for i in range(len(results['ids'][0])):
            metrics.append({
                "id": results['ids'][0][i],
                "prompt": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return metrics 