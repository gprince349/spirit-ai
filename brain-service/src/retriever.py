"""Retriever module - combines embeddings and vector store."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional

from .embeddings import embedder
from .vector_store import vector_store
from config import settings


class Retriever:
    """
    Retriever that combines embedding and vector search.
    
    Provides a simple interface for semantic search over documents.
    """
    
    def __init__(self, top_k: int = None):
        """
        Initialize retriever.
        
        Args:
            top_k: Default number of results to return
        """
        self.top_k = top_k or settings.TOP_K
        self.embedder = embedder
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None
    ) -> list[dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query text
            top_k: Number of results (overrides default)
            min_score: Minimum similarity score threshold
            
        Returns:
            List of dicts with 'text', 'score', and 'metadata'
        """
        top_k = top_k or self.top_k
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k)
        
        # Filter by minimum score if specified
        if min_score is not None:
            results = [r for r in results if r['score'] >= min_score]
        
        return results
    
    def retrieve_as_context(
        self,
        query: str,
        top_k: int = None,
        separator: str = "\n\n---\n\n"
    ) -> tuple[str, list[dict]]:
        """
        Retrieve documents and format as context string.
        
        Args:
            query: Search query text
            top_k: Number of results
            separator: String to join chunks
            
        Returns:
            Tuple of (context_string, source_documents)
        """
        results = self.retrieve(query, top_k)
        
        # Join texts into context
        context = separator.join([r['text'] for r in results])
        
        return context, results
    
    def is_ready(self) -> bool:
        """Check if retriever is ready (index loaded)."""
        return self.vector_store.total_vectors > 0
    
    @property
    def stats(self) -> dict:
        """Get retriever statistics."""
        return {
            "total_vectors": self.vector_store.total_vectors,
            "total_chunks": self.vector_store.total_chunks,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": self.embedder.dimension
        }


# Global singleton instance
retriever = Retriever()

