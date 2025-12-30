"""
Retrieval System for RAG-based Chatbot

This module provides:
- Query embedding generation
- Similarity search with ranking and filtering
- Context-aware retrieval
- Metadata-rich results
- Query preprocessing and optimization
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

# Local imports
try:
    from .vector_store import BaseVectorStore, SearchResult, VectorStoreFactory, VectorStoreConfig
    from .document_processor import DocumentProcessor, ProcessingConfig
except ImportError:
    from vector_store import BaseVectorStore, SearchResult, VectorStoreFactory, VectorStoreConfig
    from document_processor import DocumentProcessor, ProcessingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system"""
    vector_store_type: str = "faiss"
    index_path: str = "data/vector_index"
    embedding_model: str = 'all-MiniLM-L6-v2'
    default_k: int = 5
    max_k: int = 20
    min_score_threshold: float = 0.1
    rerank_results: bool = True
    filter_duplicates: bool = True
    diversity_threshold: float = 0.8


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with additional context"""
    chunk_text: str
    source_file: str
    page_number: int
    chunk_index: int
    score: float
    rank: int
    metadata: Dict[str, Any]
    context_chunks: Optional[List[str]] = None
    
    @property
    def similarity_score(self) -> float:
        """Alias for score to maintain compatibility"""
        return self.score


class QueryProcessor:
    """Handles query preprocessing and optimization"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        """
        Initialize query processor
        
        Args:
            embedding_model: SentenceTransformer model for embedding generation
        """
        self.embedding_model = embedding_model
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess and clean the query
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query string
        """
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase for consistency (optional, depends on model)
        # query = query.lower()
        
        # Remove special characters that might interfere with embedding
        query = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', query)
        
        # Remove multiple spaces
        query = re.sub(r' +', ' ', query)
        
        return query.strip()
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for the query
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        try:
            processed_query = self.preprocess_query(query)
            logger.info(f"Generating embedding for query: '{processed_query[:50]}...'")
            
            embedding = self.embedding_model.encode(
                processed_query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms or related terms (basic implementation)
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        # This is a basic implementation - in production, you might use
        # more sophisticated query expansion techniques
        expanded_queries = [query]
        
        # Add some basic expansions
        query_lower = query.lower()
        
        # Add variations for common AI/ML terms
        expansions = {
            'ai': ['artificial intelligence', 'machine learning', 'AI'],
            'ml': ['machine learning', 'artificial intelligence', 'ML'],
            'deep learning': ['neural networks', 'deep learning', 'DL'],
            'nlp': ['natural language processing', 'text processing', 'NLP'],
            'computer vision': ['image processing', 'visual recognition', 'CV']
        }
        
        for term, variants in expansions.items():
            if term in query_lower:
                for variant in variants:
                    if variant.lower() != query_lower:
                        expanded_queries.append(query.replace(term, variant))
        
        return list(set(expanded_queries))  # Remove duplicates


class ResultRanker:
    """Handles result ranking and filtering"""
    
    def __init__(self, config: RetrievalConfig):
        """
        Initialize result ranker
        
        Args:
            config: Retrieval configuration
        """
        self.config = config
    
    def filter_by_score(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Filter results by minimum score threshold
        
        Args:
            results: List of search results
            
        Returns:
            Filtered results
        """
        filtered = [r for r in results if r.score >= self.config.min_score_threshold]
        logger.info(f"Filtered {len(results)} -> {len(filtered)} results by score threshold")
        return filtered
    
    def remove_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results based on text similarity
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated results
        """
        if not self.config.filter_duplicates:
            return results
        
        deduplicated = []
        
        for result in results:
            is_duplicate = False
            
            for existing in deduplicated:
                # Simple text similarity check
                similarity = self._text_similarity(result.chunk.text, existing.chunk.text)
                if similarity > self.config.diversity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        logger.info(f"Removed {len(results) - len(deduplicated)} duplicate results")
        return deduplicated
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using simple overlap
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Rerank results based on additional criteria
        
        Args:
            results: List of search results
            query: Original query
            
        Returns:
            Reranked results
        """
        if not self.config.rerank_results:
            return results
        
        # Simple reranking based on query term presence
        query_terms = set(query.lower().split())
        
        for result in results:
            # Calculate term overlap bonus
            text_terms = set(result.chunk.text.lower().split())
            term_overlap = len(query_terms.intersection(text_terms))
            
            # Calculate recency bonus (prefer more recent pages/chunks)
            recency_bonus = 1.0 / (1.0 + result.chunk.metadata.get('page_number', 1))
            
            # Combine scores
            enhanced_score = (
                result.score * 0.7 +  # Original similarity score
                (term_overlap / len(query_terms)) * 0.2 +  # Term overlap
                recency_bonus * 0.1  # Recency bonus
            )
            
            result.score = enhanced_score
        
        # Re-sort by enhanced score
        reranked = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i
        
        logger.info("Reranked results based on enhanced scoring")
        return reranked


class DocumentRetriever:
    """Main retrieval system class"""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        Initialize document retriever
        
        Args:
            config: Retrieval configuration
        """
        self.config = config or RetrievalConfig()
        self.vector_store = None
        self.query_processor = None
        self.result_ranker = None
        self.embedding_model = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize retrieval components"""
        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Initialize components
            self.query_processor = QueryProcessor(self.embedding_model)
            self.result_ranker = ResultRanker(self.config)
            
            # Initialize vector store
            vector_config = VectorStoreConfig(
                store_type=self.config.vector_store_type,
                index_path=self.config.index_path,
                embedding_dim=384  # all-MiniLM-L6-v2 dimension
            )
            
            self.vector_store = VectorStoreFactory.create_vector_store(vector_config)
            
            logger.info("Document retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize document retriever: {e}")
            raise
    
    def load_index(self) -> None:
        """Load the vector index from disk"""
        try:
            if self.vector_store:
                self.vector_store.load_index(self.config.index_path)
                logger.info("Vector index loaded successfully")
            else:
                raise ValueError("Vector store not initialized")
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            raise
    
    def retrieve(
        self, 
        query: str, 
        k: Optional[int] = None,
        include_context: bool = False
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of results to return (uses default if None)
            include_context: Whether to include surrounding context chunks
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            # Use default k if not specified
            k = k or self.config.default_k
            k = min(k, self.config.max_k)
            
            logger.info(f"Retrieving documents for query: '{query[:50]}...'")
            
            # Generate query embedding
            query_embedding = self.query_processor.generate_query_embedding(query)
            
            # Search vector store
            search_results = self.vector_store.search(query_embedding, k=k * 2)  # Get more for filtering
            
            # Filter and rank results
            filtered_results = self.result_ranker.filter_by_score(search_results)
            deduplicated_results = self.result_ranker.remove_duplicates(filtered_results)
            reranked_results = self.result_ranker.rerank_results(deduplicated_results, query)
            
            # Take top k results
            final_results = reranked_results[:k]
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in final_results:
                context_chunks = None
                if include_context:
                    context_chunks = self._get_context_chunks(result.chunk)
                
                retrieval_result = RetrievalResult(
                    chunk_text=result.chunk.text,
                    source_file=result.chunk.metadata.get('filename', 'unknown'),
                    page_number=result.chunk.metadata.get('page_number', 0),
                    chunk_index=result.chunk.metadata.get('chunk_index', 0),
                    score=result.score,
                    rank=result.rank,
                    metadata=result.chunk.metadata,
                    context_chunks=context_chunks
                )
                retrieval_results.append(retrieval_result)
            
            logger.info(f"Retrieved {len(retrieval_results)} relevant documents")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise
    
    def _get_context_chunks(self, target_chunk) -> List[str]:
        """
        Get surrounding context chunks for a target chunk
        
        Args:
            target_chunk: The target DocumentChunk
            
        Returns:
            List of context chunk texts
        """
        # This is a simplified implementation
        # In practice, you'd need to store chunk relationships
        # or query the vector store for chunks from the same document
        
        context_chunks = []
        
        # For now, just return the chunk itself
        # In a full implementation, you'd find adjacent chunks
        # from the same document based on chunk_index
        
        return context_chunks
    
    def batch_retrieve(
        self, 
        queries: List[str], 
        k: Optional[int] = None
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve documents for multiple queries
        
        Args:
            queries: List of search queries
            k: Number of results per query
            
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        logger.info(f"Processing {len(queries)} queries")
        
        for query in queries:
            try:
                query_results = self.retrieve(query, k=k)
                results[query] = query_results
            except Exception as e:
                logger.error(f"Failed to process query '{query}': {e}")
                results[query] = []
        
        return results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system"""
        vector_stats = self.vector_store.get_stats() if self.vector_store else {}
        
        return {
            'vector_store_stats': vector_stats,
            'embedding_model': self.config.embedding_model,
            'default_k': self.config.default_k,
            'score_threshold': self.config.min_score_threshold,
            'reranking_enabled': self.config.rerank_results,
            'duplicate_filtering': self.config.filter_duplicates
        }


def create_ingestion_pipeline(
    documents_dir: str,
    index_path: str,
    vector_store_type: str = "faiss"
) -> None:
    """
    Create end-to-end ingestion pipeline
    
    Args:
        documents_dir: Directory containing PDF documents
        index_path: Path to save the vector index
        vector_store_type: Type of vector store ("faiss" or "chromadb")
    """
    logger.info("Starting end-to-end document ingestion pipeline")
    
    try:
        # Initialize document processor
        doc_config = ProcessingConfig(
            min_chunk_size=500,
            max_chunk_size=1000,
            overlap_size=100,
            embedding_model='all-MiniLM-L6-v2',
            batch_size=32
        )
        doc_processor = DocumentProcessor(doc_config)
        
        # Get all PDF files
        if not os.path.exists(documents_dir):
            raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
        
        pdf_files = [
            os.path.join(documents_dir, f) 
            for f in os.listdir(documents_dir) 
            if f.endswith('.pdf')
        ]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {documents_dir}")
        
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        # Process all documents
        all_chunks = doc_processor.process_documents_batch(pdf_files)
        
        # Get processing stats
        stats = doc_processor.get_embedding_stats(all_chunks)
        logger.info(f"Processing completed: {stats}")
        
        # Initialize vector store
        vector_config = VectorStoreConfig(
            store_type=vector_store_type,
            index_path=index_path,
            embedding_dim=384
        )
        
        vector_store = VectorStoreFactory.create_vector_store(vector_config)
        
        # Add documents to vector store
        vector_store.add_documents(all_chunks)
        
        # Save index
        vector_store.save_index(index_path)
        
        # Get final stats
        vector_stats = vector_store.get_stats()
        logger.info(f"Vector store created: {vector_stats}")
        
        logger.info("Ingestion pipeline completed successfully!")
        
        return {
            'processing_stats': stats,
            'vector_stats': vector_stats,
            'total_documents': len(pdf_files),
            'index_path': index_path
        }
        
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        raise


def test_retrieval_system():
    """Test the complete retrieval system"""
    try:
        # First, create the ingestion pipeline
        documents_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'documents')
        index_path = "test_data/retrieval_index"
        
        print("Creating ingestion pipeline...")
        pipeline_stats = create_ingestion_pipeline(
            documents_dir=documents_dir,
            index_path=index_path,
            vector_store_type="faiss"
        )
        
        print(f"Pipeline completed: {pipeline_stats}")
        
        # Initialize retrieval system
        print("\nInitializing retrieval system...")
        retrieval_config = RetrievalConfig(
            vector_store_type="faiss",
            index_path=index_path,
            default_k=5,
            rerank_results=True,
            filter_duplicates=True
        )
        
        retriever = DocumentRetriever(retrieval_config)
        retriever.load_index()
        
        # Test queries
        test_queries = [
            "What is artificial intelligence?",
            "How is AI used in finance and banking?",
            "Machine learning algorithms and applications",
            "Computer vision and image processing",
            "Natural language processing techniques"
        ]
        
        print("\nTesting retrieval with sample queries...")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = retriever.retrieve(query, k=3)
            
            for i, result in enumerate(results):
                print(f"  Result {i+1}: Score={result.score:.4f}")
                print(f"    Source: {result.source_file} (Page {result.page_number})")
                print(f"    Text: {result.chunk_text[:100]}...")
        
        # Get system stats
        stats = retriever.get_retrieval_stats()
        print(f"\nRetrieval system stats: {stats}")
        
        print("\nRetrieval system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Retrieval system test failed: {e}")
        return False


if __name__ == "__main__":
    test_retrieval_system()
