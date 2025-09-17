"""
Document Ingestion Module for RAG-based Chatbot

This module handles:
- PDF parsing using PyMuPDF
- Text chunking with overlap strategy
- Metadata extraction
- Embedding generation using SentenceTransformers
- Batch processing with progress tracking
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Data class to represent a document chunk with metadata"""
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    min_chunk_size: int = 500
    max_chunk_size: int = 1000
    overlap_size: int = 100
    embedding_model: str = 'all-MiniLM-L6-v2'
    batch_size: int = 32


class DocumentProcessor:
    """
    Main class for processing documents and generating embeddings
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the document processor
        
        Args:
            config: Processing configuration, uses default if None
        """
        self.config = config or ProcessingConfig()
        self.embedding_model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the SentenceTransformers model"""
        try:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF using PyMuPDF
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text and page metadata
        """
        try:
            doc = fitz.open(file_path)
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Skip empty pages
                if text.strip():
                    pages_data.append({
                        'text': text,
                        'page_number': page_num + 1,
                        'filename': os.path.basename(file_path)
                    })
            
            doc.close()
            logger.info(f"Extracted text from {len(pages_data)} pages in {file_path}")
            return pages_data
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def create_chunks_with_overlap(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Create text chunks with overlap strategy
        
        Args:
            text: Text to chunk
            metadata: Base metadata for the text
            
        Returns:
            List of DocumentChunk objects
        """
        # Preprocess the text
        clean_text = self.preprocess_text(text)
        
        if len(clean_text) < self.config.min_chunk_size:
            # If text is smaller than min chunk size, return as single chunk
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': 0,
                'chunk_size': len(clean_text),
                'total_chunks': 1
            })
            return [DocumentChunk(text=clean_text, metadata=chunk_metadata)]
        
        chunks = []
        chunk_index = 0
        start = 0
        
        while start < len(clean_text):
            # Determine end position
            end = start + self.config.max_chunk_size
            
            # If this would be the last chunk and it's too small, extend the previous chunk
            if end >= len(clean_text):
                end = len(clean_text)
            else:
                # Try to break at a sentence or word boundary
                break_point = self._find_break_point(clean_text, start, end)
                if break_point > start:
                    end = break_point
            
            chunk_text = clean_text[start:end].strip()
            
            # Create chunk metadata
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': chunk_index,
                'chunk_size': len(chunk_text),
                'start_char': start,
                'end_char': end
            })
            
            chunks.append(DocumentChunk(text=chunk_text, metadata=chunk_metadata))
            
            # Move start position with overlap
            if end >= len(clean_text):
                break
            
            start = end - self.config.overlap_size
            chunk_index += 1
        
        # Update total chunks count in all chunks
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(clean_text)}")
        return chunks
    
    def _find_break_point(self, text: str, start: int, max_end: int) -> int:
        """
        Find a good break point for chunking (sentence or word boundary)
        
        Args:
            text: The text to analyze
            start: Start position
            max_end: Maximum end position
            
        Returns:
            Optimal break point position
        """
        # Look for sentence endings first
        sentence_endings = ['.', '!', '?']
        for i in range(max_end - 1, start + self.config.min_chunk_size, -1):
            if text[i] in sentence_endings and i + 1 < len(text) and text[i + 1] == ' ':
                return i + 1
        
        # Look for word boundaries
        for i in range(max_end - 1, start + self.config.min_chunk_size, -1):
            if text[i] == ' ':
                return i
        
        # If no good break point found, use max_end
        return max_end
    
    def generate_embeddings_batch(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for chunks in batches
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of DocumentChunk objects with embeddings
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Process in batches with progress tracking
        for i in tqdm(range(0, len(chunks), self.config.batch_size), 
                     desc="Generating embeddings"):
            batch_chunks = chunks[i:i + self.config.batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]
            
            try:
                # Generate embeddings for the batch
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Assign embeddings to chunks
                for j, chunk in enumerate(batch_chunks):
                    chunk.embedding = batch_embeddings[j]
                    chunk.metadata['embedding_dim'] = len(batch_embeddings[j])
                    
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                raise
        
        logger.info("Embedding generation completed")
        return chunks
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a single document: extract text, chunk, and generate embeddings
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed DocumentChunk objects with embeddings
        """
        logger.info(f"Processing document: {file_path}")
        
        # Validate file exists and is PDF
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"Only PDF files are supported. Got: {file_path}")
        
        # Extract text from PDF
        pages_data = self.extract_text_from_pdf(file_path)
        
        all_chunks = []
        
        # Process each page
        for page_data in pages_data:
            base_metadata = {
                'filename': page_data['filename'],
                'page_number': page_data['page_number'],
                'file_path': file_path
            }
            
            # Create chunks for this page
            page_chunks = self.create_chunks_with_overlap(
                page_data['text'], 
                base_metadata
            )
            
            all_chunks.extend(page_chunks)
        
        # Generate embeddings for all chunks
        all_chunks = self.generate_embeddings_batch(all_chunks)
        
        logger.info(f"Document processing completed. Generated {len(all_chunks)} chunks")
        return all_chunks
    
    def process_documents_batch(self, file_paths: List[str]) -> List[DocumentChunk]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            List of all processed DocumentChunk objects
        """
        all_chunks = []
        
        logger.info(f"Processing {len(file_paths)} documents")
        
        for file_path in tqdm(file_paths, desc="Processing documents"):
            try:
                document_chunks = self.process_document(file_path)
                all_chunks.extend(document_chunks)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        logger.info(f"Batch processing completed. Total chunks: {len(all_chunks)}")
        return all_chunks
    
    def get_embedding_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Get statistics about the processed chunks and embeddings
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Dictionary containing statistics
        """
        if not chunks:
            return {}
        
        chunk_sizes = [chunk.metadata.get('chunk_size', 0) for chunk in chunks]
        embedding_dims = [chunk.metadata.get('embedding_dim', 0) for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_size': np.mean(chunk_sizes) if chunk_sizes else 0,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'embedding_dimension': embedding_dims[0] if embedding_dims else 0,
            'unique_documents': len(set(chunk.metadata.get('filename', '') for chunk in chunks)),
            'total_pages': len(set((chunk.metadata.get('filename', ''), 
                                 chunk.metadata.get('page_number', 0)) for chunk in chunks))
        }
        
        return stats


def test_document_processor():
    """
    Test function to validate document processing functionality
    """
    # Initialize processor
    config = ProcessingConfig(
        min_chunk_size=500,
        max_chunk_size=1000,
        overlap_size=100
    )
    processor = DocumentProcessor(config)
    
    # Test with sample documents (if they exist)
    documents_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'documents')
    
    if os.path.exists(documents_dir):
        pdf_files = [f for f in os.listdir(documents_dir) if f.endswith('.pdf')]
        
        if pdf_files:
            # Test with first PDF file
            test_file = os.path.join(documents_dir, pdf_files[0])
            logger.info(f"Testing with file: {test_file}")
            
            try:
                chunks = processor.process_document(test_file)
                stats = processor.get_embedding_stats(chunks)
                
                logger.info("Test Results:")
                logger.info(f"  Total chunks: {stats['total_chunks']}")
                logger.info(f"  Average chunk size: {stats['avg_chunk_size']:.2f}")
                logger.info(f"  Embedding dimension: {stats['embedding_dimension']}")
                logger.info("Document processing test completed successfully!")
                
                # Validate chunk sizes and overlap
                for i, chunk in enumerate(chunks[:5]):  # Check first 5 chunks
                    size = chunk.metadata['chunk_size']
                    logger.info(f"  Chunk {i}: size={size}, page={chunk.metadata.get('page_number')}")
                
                return True
                
            except Exception as e:
                logger.error(f"Test failed: {e}")
                return False
        else:
            logger.warning("No PDF files found for testing")
            return False
    else:
        logger.warning("Documents directory not found for testing")
        return False


if __name__ == "__main__":
    # Run tests when executed directly
    test_document_processor()
