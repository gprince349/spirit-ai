"""
RAG-based Chatbot - Core Package

This package contains the core components for the RAG (Retrieval-Augmented Generation) chatbot:
- Document processing and embedding generation
- Vector store management and retrieval
- LLM integration and response generation
- FastAPI server and endpoints
"""

# Version information
__version__ = "1.0.0"
__author__ = "RAG Chatbot Team"

# Core component imports for easier access
from .document_processor import DocumentProcessor, ProcessingConfig
from .vector_store import VectorStoreFactory, VectorStoreConfig
from .retriever import RetrievalConfig
from .llm_handler import RAGPipeline, LLMConfig, RAGResponse

# Make key classes available at package level
__all__ = [
    "DocumentProcessor",
    "ProcessingConfig", 
    "VectorStoreFactory",
    "VectorStoreConfig",
    "RetrievalConfig",
    "RAGPipeline", 
    "LLMConfig",
    "RAGResponse"
]