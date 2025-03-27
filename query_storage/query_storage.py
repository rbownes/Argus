"""
Query storage implementation using ChromaDB.
"""
import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict, Optional
import uuid
import logging

class QueryStorage:
    """Storage for queries using ChromaDB vector database."""
    
    def __init__(self, persist_directory: str = "/app/query_storage_data"):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.logger = logging.getLogger("query_storage")
        
        try:
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
            self.collection = self.client.get_or_create_collection(
                name="queries",
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"ChromaDB collection 'queries' initialized at {persist_directory}")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def store_query(self, query: str, theme: str, metadata: Optional[Dict] = None) -> str:
        """
        Store a query with its theme and optional metadata.
        
        Args:
            query: The query text
            theme: Theme or category of the query
            metadata: Additional metadata for the query
            
        Returns:
            The query ID (UUID)
        """
        try:
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
            
            self.logger.info(f"Stored query with ID {query_id} and theme '{theme}'")
            return query_id
        except Exception as e:
            self.logger.error(f"Failed to store query: {str(e)}")
            raise

    def get_queries_by_theme(self, theme: str, limit: int = 10, skip: int = 0) -> List[Dict]:
        """
        Retrieve queries by theme with pagination.
        
        Args:
            theme: Theme to filter by
            limit: Maximum number of results to return
            skip: Number of results to skip for pagination
            
        Returns:
            List of queries with their metadata
        """
        try:
            # First get all IDs matching the theme
            all_ids_result = self.collection.query(
                query_texts=[""], 
                where={"theme": theme},
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
                        "query": get_result["documents"][0],
                        "metadata": get_result["metadatas"][0]
                    })
            
            self.logger.info(f"Retrieved {len(results)} queries with theme '{theme}'")
            return results
        except Exception as e:
            self.logger.error(f"Failed to get queries by theme: {str(e)}")
            raise

    def count_queries_by_theme(self, theme: str) -> int:
        """
        Count the number of queries for a theme.
        
        Args:
            theme: Theme to count queries for
            
        Returns:
            Number of queries for the theme
        """
        try:
            results = self.collection.query(
                query_texts=[""],
                where={"theme": theme},
                n_results=1000
            )
            
            count = len(results["ids"][0]) if results["ids"] and results["ids"][0] else 0
            self.logger.info(f"Counted {count} queries with theme '{theme}'")
            return count
        except Exception as e:
            self.logger.error(f"Failed to count queries by theme: {str(e)}")
            raise

    def search_similar_queries(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar queries using semantic similarity.
        
        Args:
            query: Query text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of similar queries with their metadata and distance score
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            queries = []
            for i in range(len(results["ids"][0])):
                queries.append({
                    "id": results["ids"][0][i],
                    "query": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                })
            
            self.logger.info(f"Found {len(queries)} similar queries")
            return queries
        except Exception as e:
            self.logger.error(f"Failed to search similar queries: {str(e)}")
            raise
            
    def get_query_by_id(self, query_id: str) -> Optional[Dict]:
        """
        Retrieve a query by its ID.
        
        Args:
            query_id: ID of the query to retrieve
            
        Returns:
            Query data or None if not found
        """
        try:
            result = self.collection.get(ids=[query_id])
            
            if not result["ids"]:
                return None
                
            return {
                "id": result["ids"][0],
                "query": result["documents"][0],
                "metadata": result["metadatas"][0]
            }
        except Exception as e:
            self.logger.error(f"Failed to get query by ID: {str(e)}")
            raise
