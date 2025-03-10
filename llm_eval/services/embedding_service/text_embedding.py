"""
Text embedding service for generating vector embeddings from text.
"""
from typing import List, Dict, Any, Optional, Tuple
import os

from llm_eval.core.utils import Result, measure_latency

# Import sentence-transformers conditionally
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class TextEmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None
    ):
        """
        Initialize the text embedding service.
        
        Args:
            model_name: Name of the embedding model to use.
            device: Device to run the model on (cpu, cuda, etc.).
        """
        if SentenceTransformer is None:
            raise ImportError(
                "SentenceTransformer is not installed. Install it with 'pip install sentence-transformers'"
            )
        
        # Set device (use CUDA if available)
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        # Load the embedding model
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        
    @measure_latency
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed.
            
        Returns:
            A list of floats representing the embedding.
        """
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        return embedding
    
    @measure_latency
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            A list of embeddings.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        return embeddings

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            The dimension of the embeddings.
        """
        return self.model.get_sentence_embedding_dimension()
