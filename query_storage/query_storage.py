import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict, Optional
import uuid

class QueryStorage:
    def __init__(self, persist_directory: str = "./data"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name="queries",
            metadata={"hnsw:space": "cosine"}
        )

    def store_query(self, query: str, theme: str, metadata: Optional[Dict] = None) -> str:
        """
        Store a query with its theme and optional metadata
        Returns the query ID
        """
        query_id = str(uuid.uuid4())
        
        # Prepare metadata
        query_metadata = {
            "theme": theme,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {})
        }
        
        # Store in ChromaDB
        self.collection.add(
            documents=[query],
            metadatas=[query_metadata],
            ids=[query_id]
        )
        
        return query_id

    def get_queries_by_theme(self, theme: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve queries by theme
        Returns a list of queries with their metadata
        """
        results = self.collection.query(
            query_texts=[""],  # Empty query to get all documents
            where={"theme": theme},
            n_results=limit
        )
        
        queries = []
        for i in range(len(results['ids'][0])):
            queries.append({
                "id": results['ids'][0][i],
                "query": results['documents'][0][i],
                "metadata": results['metadatas'][0][i]
            })
        
        return queries

    def search_similar_queries(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar queries using semantic similarity
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        queries = []
        for i in range(len(results['ids'][0])):
            queries.append({
                "id": results['ids'][0][i],
                "query": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
        
        return queries
