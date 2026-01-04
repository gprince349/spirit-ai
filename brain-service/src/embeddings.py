"""Embedding model handler using sentence-transformers."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from config import settings


class EmbeddingModel:
    """
    Singleton embedding model handler.
    
    Loads the model once and reuses it for all embedding operations.
    Uses LRU cache for repeated query embeddings.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EmbeddingModel._model is None:
            print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            EmbeddingModel._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            print(f"Model loaded. Dimension: {self.dimension}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the underlying model."""
        return EmbeddingModel._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (n_texts, dimension)
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity with dot product
            show_progress_bar=len(texts) > 100,
            batch_size=32
        )
        return embeddings.astype('float32')
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        
        Uses caching for repeated queries.
        
        Args:
            query: Query text to embed
            
        Returns:
            numpy array of shape (dimension,)
        """
        return self._embed_query_cached(query)
    
    @lru_cache(maxsize=1000)
    def _embed_query_cached(self, query: str) -> np.ndarray:
        """Cached query embedding."""
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype('float32')
    
    def clear_cache(self):
        """Clear the query embedding cache."""
        self._embed_query_cached.cache_clear()


# Global singleton instance
embedder = EmbeddingModel()

