"""
Vector Store Implementation for RAG-based Chatbot

This module provides vector storage and similarity search capabilities using:
- FAISS for high-performance similarity search
- ChromaDB as an alternative vector database
- Persistent storage and retrieval
- Metadata management
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

# Vector database imports
import faiss
import chromadb
from chromadb.config import Settings

# Local imports
try:
    from .document_processor import DocumentChunk
except ImportError:
    from document_processor import DocumentChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Data class for search results"""
    chunk: DocumentChunk
    score: float
    rank: int


@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    store_type: str = "faiss"  # "faiss" or "chromadb"
    index_path: str = "data/vector_index"
    collection_name: str = "documents"
    embedding_dim: int = 384
    distance_metric: str = "cosine"  # "cosine", "euclidean", "inner_product"


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> None:
        """Save the index to disk"""
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> None:
        """Load the index from disk"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize FAISS vector store
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self.index = None
        self.chunks = []  # Store chunks for metadata retrieval
        self.chunk_ids = []  # Store chunk IDs for mapping
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index based on configuration"""
        try:
            if self.config.distance_metric == "cosine":
                # For cosine similarity, normalize vectors and use inner product
                self.index = faiss.IndexFlatIP(self.config.embedding_dim)
            elif self.config.distance_metric == "euclidean":
                self.index = faiss.IndexFlatL2(self.config.embedding_dim)
            elif self.config.distance_metric == "inner_product":
                self.index = faiss.IndexFlatIP(self.config.embedding_dim)
            else:
                raise ValueError(f"Unsupported distance metric: {self.config.distance_metric}")
            
            logger.info(f"Initialized FAISS index with {self.config.distance_metric} distance metric")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        if self.config.distance_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return embeddings / norms
        return embeddings
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the FAISS index
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
        
        try:
            # Extract embeddings and prepare for FAISS
            embeddings = []
            valid_chunks = []
            
            for chunk in chunks:
                if chunk.embedding is not None:
                    embeddings.append(chunk.embedding)
                    valid_chunks.append(chunk)
                else:
                    logger.warning(f"Chunk without embedding skipped: {chunk.metadata.get('filename', 'unknown')}")
            
            if not embeddings:
                logger.error("No valid embeddings found in chunks")
                return
            
            # Convert to numpy array and normalize if needed
            embeddings_array = np.array(embeddings).astype(np.float32)
            embeddings_array = self._normalize_embeddings(embeddings_array)
            
            # Add to FAISS index
            start_id = len(self.chunks)
            self.index.add(embeddings_array)
            
            # Store chunks and IDs for metadata retrieval
            self.chunks.extend(valid_chunks)
            self.chunk_ids.extend(range(start_id, start_id + len(valid_chunks)))
            
            logger.info(f"Added {len(valid_chunks)} documents to FAISS index. Total: {len(self.chunks)}")
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS index: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        """
        Search for similar documents using FAISS
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None or len(self.chunks) == 0:
            logger.warning("Vector store is empty or not initialized")
            return []
        
        try:
            # Prepare query embedding
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            query_vector = self._normalize_embeddings(query_vector)
            
            # Ensure k doesn't exceed available documents
            k = min(k, len(self.chunks))
            
            # Search FAISS index
            scores, indices = self.index.search(query_vector, k)
            
            # Prepare results
            results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):  # Valid index
                    chunk = self.chunks[idx]
                    
                    # Convert FAISS score to similarity score
                    if self.config.distance_metric == "cosine" or self.config.distance_metric == "inner_product":
                        # Higher score is better for cosine/inner product
                        similarity_score = float(score)
                    else:
                        # Lower score is better for L2 distance, convert to similarity
                        similarity_score = 1.0 / (1.0 + float(score))
                    
                    results.append(SearchResult(
                        chunk=chunk,
                        score=similarity_score,
                        rank=rank
                    ))
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            raise
    
    def save_index(self, path: str) -> None:
        """
        Save FAISS index and metadata to disk
        
        Args:
            path: Directory path to save the index
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save FAISS index
            index_file = os.path.join(path, "faiss_index.bin")
            faiss.write_index(self.index, index_file)
            
            # Save chunks metadata
            chunks_file = os.path.join(path, "chunks_metadata.pkl")
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save configuration
            config_file = os.path.join(path, "config.json")
            config_dict = {
                'store_type': self.config.store_type,
                'embedding_dim': self.config.embedding_dim,
                'distance_metric': self.config.distance_metric,
                'total_chunks': len(self.chunks)
            }
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Saved FAISS index with {len(self.chunks)} documents to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
    
    def load_index(self, path: str) -> None:
        """
        Load FAISS index and metadata from disk
        
        Args:
            path: Directory path to load the index from
        """
        try:
            # Load FAISS index
            index_file = os.path.join(path, "faiss_index.bin")
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"FAISS index file not found: {index_file}")
            
            self.index = faiss.read_index(index_file)
            
            # Load chunks metadata
            chunks_file = os.path.join(path, "chunks_metadata.pkl")
            if os.path.exists(chunks_file):
                with open(chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
            else:
                logger.warning("Chunks metadata file not found")
                self.chunks = []
            
            # Load configuration
            config_file = os.path.join(path, "config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                logger.info(f"Loaded configuration: {config_dict}")
            
            logger.info(f"Loaded FAISS index with {len(self.chunks)} documents from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS vector store"""
        return {
            'store_type': 'FAISS',
            'total_documents': len(self.chunks),
            'embedding_dimension': self.config.embedding_dim,
            'distance_metric': self.config.distance_metric,
            'index_size': self.index.ntotal if self.index else 0,
            'is_trained': self.index.is_trained if self.index else False
        }


class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB-based vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize ChromaDB vector store
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self.client = None
        self.collection = None
        self.chunks = []
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create ChromaDB client with persistent storage
            os.makedirs(self.config.index_path, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=self.config.index_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name
                )
                logger.info(f"Loaded existing ChromaDB collection: {self.config.collection_name}")
            except:
                # Collection doesn't exist, create it
                distance_function = "cosine" if self.config.distance_metric == "cosine" else "l2"
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": distance_function}
                )
                logger.info(f"Created new ChromaDB collection: {self.config.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to ChromaDB
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
        """
        if not chunks:
            logger.warning("No chunks provided to add to vector store")
            return
        
        try:
            # Prepare data for ChromaDB
            embeddings = []
            documents = []
            metadatas = []
            ids = []
            
            start_id = len(self.chunks)
            
            for i, chunk in enumerate(chunks):
                if chunk.embedding is not None:
                    embeddings.append(chunk.embedding.tolist())
                    documents.append(chunk.text)
                    
                    # Prepare metadata (ChromaDB requires string values)
                    metadata = {}
                    for key, value in chunk.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = str(value)
                        else:
                            metadata[key] = str(value)
                    
                    metadatas.append(metadata)
                    ids.append(f"chunk_{start_id + i}")
                    self.chunks.append(chunk)
                else:
                    logger.warning(f"Chunk without embedding skipped: {chunk.metadata.get('filename', 'unknown')}")
            
            if not embeddings:
                logger.error("No valid embeddings found in chunks")
                return
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(embeddings)} documents to ChromaDB. Total: {len(self.chunks)}")
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        """
        Search for similar documents using ChromaDB
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if not self.collection:
            logger.warning("ChromaDB collection is not initialized")
            return []
        
        # Check collection count
        collection_count = self.collection.count()
        if collection_count == 0:
            logger.warning("ChromaDB collection is empty")
            return []
        
        try:
            # Ensure k doesn't exceed available documents
            k = min(k, collection_count)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            
            # Process results
            search_results = []
            
            if results['ids'] and len(results['ids'][0]) > 0:
                for rank, (doc_id, distance, document, metadata) in enumerate(zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['documents'][0],
                    results['metadatas'][0]
                )):
                    # Create a DocumentChunk from the retrieved data
                    # Convert metadata back to proper types
                    chunk_metadata = {}
                    for key, value in metadata.items():
                        if key in ['page_number', 'chunk_index', 'chunk_size', 'total_chunks']:
                            try:
                                chunk_metadata[key] = int(value)
                            except (ValueError, TypeError):
                                chunk_metadata[key] = value
                        else:
                            chunk_metadata[key] = value
                    
                    # Create DocumentChunk
                    chunk = DocumentChunk(
                        text=document,
                        metadata=chunk_metadata,
                        embedding=None  # We don't need the embedding for search results
                    )
                    
                    # Convert distance to similarity score
                    if self.config.distance_metric == "cosine":
                        # ChromaDB returns distance, convert to similarity
                        similarity_score = 1.0 - float(distance)
                    else:
                        # For L2 distance
                        similarity_score = 1.0 / (1.0 + float(distance))
                    
                    search_results.append(SearchResult(
                        chunk=chunk,
                        score=similarity_score,
                        rank=rank
                    ))
            
            logger.info(f"Found {len(search_results)} similar documents")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            raise
    
    def save_index(self, path: str) -> None:
        """
        Save ChromaDB index (already persistent)
        
        Args:
            path: Directory path (ChromaDB is already persistent)
        """
        try:
            # ChromaDB is already persistent, but save additional metadata
            os.makedirs(path, exist_ok=True)
            
            config_file = os.path.join(path, "chromadb_config.json")
            config_dict = {
                'store_type': self.config.store_type,
                'collection_name': self.config.collection_name,
                'embedding_dim': self.config.embedding_dim,
                'distance_metric': self.config.distance_metric,
                'total_chunks': len(self.chunks)
            }
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"ChromaDB configuration saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save ChromaDB configuration: {e}")
            raise
    
    def load_index(self, path: str) -> None:
        """
        Load ChromaDB index (already persistent)
        
        Args:
            path: Directory path to load configuration from
        """
        try:
            # ChromaDB loads automatically on initialization
            # Load additional configuration if available
            config_file = os.path.join(path, "chromadb_config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                logger.info(f"Loaded ChromaDB configuration: {config_dict}")
            
            # Get current collection count
            count = self.collection.count()
            logger.info(f"ChromaDB collection contains {count} documents")
            
        except Exception as e:
            logger.error(f"Failed to load ChromaDB configuration: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB vector store"""
        count = self.collection.count() if self.collection else 0
        return {
            'store_type': 'ChromaDB',
            'total_documents': count,
            'embedding_dimension': self.config.embedding_dim,
            'distance_metric': self.config.distance_metric,
            'collection_name': self.config.collection_name
        }


class VectorStoreFactory:
    """Factory class for creating vector store instances"""
    
    @staticmethod
    def create_vector_store(config: VectorStoreConfig) -> BaseVectorStore:
        """
        Create a vector store instance based on configuration
        
        Args:
            config: Vector store configuration
            
        Returns:
            Vector store instance
        """
        if config.store_type.lower() == "faiss":
            return FAISSVectorStore(config)
        elif config.store_type.lower() == "chromadb":
            return ChromaDBVectorStore(config)
        else:
            raise ValueError(f"Unsupported vector store type: {config.store_type}")


def test_vector_store():
    """Test function for vector store functionality"""
    try:
        from .document_processor import DocumentProcessor, ProcessingConfig
    except ImportError:
        from document_processor import DocumentProcessor, ProcessingConfig
    
    # Initialize document processor
    doc_config = ProcessingConfig(
        min_chunk_size=500,
        max_chunk_size=1000,
        overlap_size=100
    )
    doc_processor = DocumentProcessor(doc_config)
    
    # Test with sample document
    documents_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'documents')
    
    if os.path.exists(documents_dir):
        pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
        
        if pdf_files:
            test_file = os.path.join(documents_dir, pdf_files[0])
            chunks = doc_processor.process_document(test_file)
            
            # Test FAISS vector store
            print("Testing FAISS Vector Store...")
            faiss_config = VectorStoreConfig(
                store_type="faiss",
                index_path="test_data/faiss_index",
                embedding_dim=384
            )
            
            faiss_store = VectorStoreFactory.create_vector_store(faiss_config)
            faiss_store.add_documents(chunks)
            
            # Test search
            query_embedding = chunks[0].embedding
            results = faiss_store.search(query_embedding, k=3)
            
            print(f"FAISS search results: {len(results)}")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: Score={result.score:.4f}")
            
            # Test save/load
            faiss_store.save_index("test_data/faiss_index")
            
            # Test ChromaDB vector store
            print("\nTesting ChromaDB Vector Store...")
            chroma_config = VectorStoreConfig(
                store_type="chromadb",
                index_path="test_data/chromadb",
                embedding_dim=384
            )
            
            chroma_store = VectorStoreFactory.create_vector_store(chroma_config)
            chroma_store.add_documents(chunks)
            
            # Test search
            results = chroma_store.search(query_embedding, k=3)
            
            print(f"ChromaDB search results: {len(results)}")
            for i, result in enumerate(results):
                print(f"  Result {i+1}: Score={result.score:.4f}")
            
            print("Vector store tests completed successfully!")
            return True
        else:
            print("No PDF files found for testing")
            return False
    else:
        print("Documents directory not found for testing")
        return False


if __name__ == "__main__":
    test_vector_store()
