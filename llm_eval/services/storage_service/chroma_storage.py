"""
ChromaDB implementation for vector storage.
"""
from typing import Dict, List, Any, Optional
import json

from llm_eval.core.utils import Result, generate_id

# Import chromadb conditionally to handle environments where it's not installed
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


class ChromaStorage:
    """
    Vector database implementation using ChromaDB.
    
    This handles storing and retrieving embeddings for semantic search.
    """
    
    def __init__(
        self, 
        collection_name: str = "llm_eval_embeddings",
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """
        Initialize the ChromaDB storage.
        
        Args:
            collection_name: Name of the collection to use.
            persist_directory: Directory to persist data (for local mode).
            host: ChromaDB server host (for client mode).
            port: ChromaDB server port (for client mode).
        """
        if chromadb is None:
            raise ImportError(
                "ChromaDB is not installed. Install it with 'pip install chromadb'"
            )
        
        # Initialize ChromaDB client
        if host and port:
            # Use HTTP client
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            # Use persistent client
            self.client = chromadb.PersistentClient(
                path=persist_directory or "./chroma_data"
            )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "LLM evaluation embeddings"}
        )
    
    async def store_embedding(
        self,
        id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> Result[str]:
        """
        Store a text embedding.
        
        Args:
            id: Unique identifier for the embedding.
            text: The original text.
            embedding: The vector embedding of the text.
            metadata: Additional metadata to store with the embedding.
            
        Returns:
            Result containing the embedding ID.
        """
        try:
            # Convert metadata to JSON-serializable format
            serialized_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, list, dict)) and key != "embedding":
                    serialized_metadata[key] = value
                else:
                    # Convert non-serializable types to string
                    serialized_metadata[key] = str(value)
            
            # Add the embedding
            self.collection.add(
                ids=[id],
                embeddings=[embedding],
                metadatas=[serialized_metadata],
                documents=[text]
            )
            
            return Result.ok(id)
        except Exception as e:
            return Result.err(e)
    
    async def query_embeddings(
        self,
        query_embedding: List[float],
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        """
        Query for similar embeddings.
        
        Args:
            query_embedding: The query vector embedding.
            filter_metadata: Filter by metadata if provided.
            limit: Maximum number of results to return.
            
        Returns:
            Result containing a list of similar embeddings with metadata.
        """
        try:
            # Convert filter to ChromaDB format if provided
            where_filter = None
            if filter_metadata:
                where_filter = {}
                for key, value in filter_metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        where_filter[key] = value
            
            # Perform the query
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_filter
            )
            
            # Process results
            processed_results = []
            if results["ids"] and results["ids"][0]:
                for i, id in enumerate(results["ids"][0]):
                    processed_results.append({
                        "id": id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": float(results["distances"][0][i]) if "distances" in results else None
                    })
            
            return Result.ok(processed_results)
        except Exception as e:
            return Result.err(e)
    
    async def delete_embedding(self, id: str) -> Result[bool]:
        """
        Delete an embedding by ID.
        
        Args:
            id: The ID of the embedding to delete.
            
        Returns:
            Result containing True if successful.
        """
        try:
            self.collection.delete(ids=[id])
            return Result.ok(True)
        except Exception as e:
            return Result.err(e)
    
    async def update_embedding(
        self,
        id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[str]:
        """
        Update an existing embedding.
        
        Args:
            id: The ID of the embedding to update.
            text: The new text if provided.
            embedding: The new embedding if provided.
            metadata: The new metadata if provided.
            
        Returns:
            Result containing the embedding ID.
        """
        try:
            # Prepare update data
            update_kwargs = {"ids": [id]}
            
            if text is not None:
                update_kwargs["documents"] = [text]
            
            if embedding is not None:
                update_kwargs["embeddings"] = [embedding]
            
            if metadata is not None:
                # Convert metadata to JSON-serializable format
                serialized_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool, list, dict)) and key != "embedding":
                        serialized_metadata[key] = value
                    else:
                        # Convert non-serializable types to string
                        serialized_metadata[key] = str(value)
                
                update_kwargs["metadatas"] = [serialized_metadata]
            
            # Perform the update
            self.collection.update(**update_kwargs)
            
            return Result.ok(id)
        except Exception as e:
            return Result.err(e)
