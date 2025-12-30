#!/usr/bin/env python3
"""
End-to-End Document Ingestion Pipeline

This script processes all documents in the data directory and creates a searchable vector index.
It handles the complete pipeline from PDF parsing to vector storage.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.retriever import create_ingestion_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main ingestion pipeline function"""
    parser = argparse.ArgumentParser(description='Document Ingestion Pipeline for RAG Chatbot')
    
    parser.add_argument(
        '--documents-dir',
        default='data/documents',
        help='Directory containing PDF documents to process'
    )
    
    parser.add_argument(
        '--index-path',
        default='data/vector_index',
        help='Path to save the vector index'
    )
    
    parser.add_argument(
        '--vector-store',
        choices=['faiss', 'chromadb'],
        default='faiss',
        help='Vector store type to use'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Maximum chunk size in characters'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=100,
        help='Overlap size between chunks'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation'
    )
    
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild of existing index'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting document ingestion pipeline")
        logger.info(f"Documents directory: {args.documents_dir}")
        logger.info(f"Index path: {args.index_path}")
        logger.info(f"Vector store: {args.vector_store}")
        
        # Check if index already exists
        if os.path.exists(args.index_path) and not args.force_rebuild:
            logger.warning(f"Index already exists at {args.index_path}")
            logger.warning("Use --force-rebuild to rebuild the index")
            return
        
        # Create output directory
        os.makedirs(args.index_path, exist_ok=True)
        
        # Run ingestion pipeline
        pipeline_stats = create_ingestion_pipeline(
            documents_dir=args.documents_dir,
            index_path=args.index_path,
            vector_store_type=args.vector_store
        )
        
        logger.info("=" * 60)
        logger.info("INGESTION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {pipeline_stats['total_documents']}")
        logger.info(f"Total chunks: {pipeline_stats['processing_stats']['total_chunks']}")
        logger.info(f"Average chunk size: {pipeline_stats['processing_stats']['avg_chunk_size']:.2f}")
        logger.info(f"Embedding dimension: {pipeline_stats['processing_stats']['embedding_dimension']}")
        logger.info(f"Index saved to: {args.index_path}")
        logger.info("=" * 60)
        
        # Test the created index
        logger.info("Testing the created index...")
        test_index(args.index_path, args.vector_store)
        
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        sys.exit(1)


def test_index(index_path: str, vector_store_type: str):
    """Test the created index with sample queries"""
    try:
        from src.retriever import DocumentRetriever, RetrievalConfig
        
        # Initialize retrieval system
        retrieval_config = RetrievalConfig(
            vector_store_type=vector_store_type,
            index_path=index_path,
            default_k=3
        )
        
        retriever = DocumentRetriever(retrieval_config)
        retriever.load_index()
        
        # Test queries
        test_queries = [
            "What is artificial intelligence?",
            "How is machine learning used in healthcare?",
            "Computer vision applications"
        ]
        
        logger.info("Testing index with sample queries:")
        
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            results = retriever.retrieve(query, k=2)
            
            for i, result in enumerate(results):
                logger.info(f"  Result {i+1}: Score={result.score:.4f}")
                logger.info(f"    Source: {result.source_file}")
                logger.info(f"    Text: {result.chunk_text[:100]}...")
        
        # Get system stats
        stats = retriever.get_retrieval_stats()
        logger.info(f"\nIndex statistics: {stats['vector_store_stats']}")
        
        logger.info("Index testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Index testing failed: {e}")


if __name__ == "__main__":
    main()
