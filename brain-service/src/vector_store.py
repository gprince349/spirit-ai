"""FAISS Vector Store for document embeddings."""

import sys
from pathlib import Path as FilePath

# Add parent directory to path for imports
sys.path.insert(0, str(FilePath(__file__).parent.parent))

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import Optional

from config import settings


class VectorStore:
    """
    FAISS-based vector store for fast similarity search.
    
    Uses IndexFlatIP (Inner Product) for cosine similarity
    when vectors are normalized.
    """
    
    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[str] = []  # Original text chunks
        self.metadata: list[dict] = []  # Chunk metadata (source, page, etc.)
    
    def create_index(self, dimension: int):
        """
        Create a new FAISS index.
        
        Args:
            dimension: Embedding dimension
        """
        # IndexFlatIP: Exact inner product search
        # With normalized vectors, this gives cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []
        self.metadata = []
        print(f"Created FAISS index with dimension {dimension}")
    
    def add(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadata: Optional[list[dict]] = None
    ):
        """
        Add documents to the index.
        
        Args:
            texts: List of text chunks
            embeddings: Numpy array of embeddings (n_texts, dimension)
            metadata: Optional list of metadata dicts
        """
        if self.index is None:
            self.create_index(embeddings.shape[1])
        
        # Ensure float32 for FAISS
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store texts and metadata
        self.chunks.extend(texts)
        self.metadata.extend(metadata or [{}] * len(texts))
        
        print(f"Added {len(texts)} chunks. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> list[dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'text', 'score', and 'metadata'
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Ensure correct shape and type
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Limit top_k to available documents
        top_k = min(top_k, self.index.ntotal)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "score": float(score),
                    "metadata": self.metadata[idx]
                })
        
        return results
    
    def save(self, index_path: Optional[str] = None, chunks_path: Optional[str] = None):
        """
        Save index and chunks to disk.
        
        Args:
            index_path: Path for FAISS index file
            chunks_path: Path for chunks pickle file
        """
        index_path = index_path or settings.FAISS_INDEX_PATH
        chunks_path = chunks_path or settings.CHUNKS_PATH
        
        # Ensure directories exist
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save chunks and metadata
        with open(chunks_path, 'wb') as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata
            }, f)
        
        print(f"Saved index to {index_path}")
        print(f"Saved {len(self.chunks)} chunks to {chunks_path}")
    
    def load(self, index_path: Optional[str] = None, chunks_path: Optional[str] = None) -> bool:
        """
        Load index and chunks from disk.
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks pickle file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = index_path or settings.FAISS_INDEX_PATH
        chunks_path = chunks_path or settings.CHUNKS_PATH
        
        if not Path(index_path).exists() or not Path(chunks_path).exists():
            print(f"Index files not found at {index_path}")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load chunks and metadata
        with open(chunks_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
        
        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.chunks)} chunks")
        
        return True
    
    @property
    def total_vectors(self) -> int:
        """Get total number of vectors in index."""
        return self.index.ntotal if self.index else 0
    
    @property
    def total_chunks(self) -> int:
        """Get total number of text chunks."""
        return len(self.chunks)


# Global singleton instance
vector_store = VectorStore()

