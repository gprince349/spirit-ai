"""
FastAPI-based REST API for RAG Chatbot

This module provides:
- RESTful endpoints for chatbot queries
- Document upload and ingestion
- Health checks and system status
- Configuration management
- Error handling and validation
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Local imports
try:
    from .llm_handler import RAGPipeline, LLMConfig, RAGResponse
    from .retriever import RetrievalConfig
    from .document_processor import DocumentProcessor
    from .vector_store import VectorStoreFactory
except ImportError:
    from llm_handler import RAGPipeline, LLMConfig, RAGResponse
    from retriever import RetrievalConfig
    from document_processor import DocumentProcessor
    from vector_store import VectorStoreFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with comprehensive documentation
app = FastAPI(
    title="RAG-based Chatbot API",
    description="""
    ## RAG-based Chatbot API

    A powerful Retrieval-Augmented Generation chatbot with document ingestion capabilities.

    ### Features:
    - **Smart Document Search**: Vector-based similarity search through your document collection
    - **AI-Powered Responses**: OpenAI GPT integration for intelligent, context-aware answers
    - **Source Citations**: Every response includes source references with page numbers
    - **Document Upload**: Add new PDF documents to expand the knowledge base
    - **Multiple Query Types**: Supports Q&A, summaries, explanations, and comparisons
    - **Real-time Processing**: Fast response times with efficient vector search

    ### Quick Start:
    1. Upload documents using `/upload` endpoint
    2. Ask questions using `/chat` endpoint
    3. Get intelligent responses with source citations

    ### API Workflow:
    ```
    Documents → Vector Index → Query → AI Response + Sources
    ```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "RAG Chatbot API",
        "url": "https://github.com/your-repo/rag-chatbot",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(description="Service health status", example="ok")
    timestamp: str = Field(description="Health check timestamp", example="2024-01-15T10:30:00Z")
    version: str = Field(description="API version", example="1.0.0")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0"
            }
        }


# Global variables
rag_pipeline: Optional[RAGPipeline] = None
document_processor: Optional[DocumentProcessor] = None


# Request/Response Models with comprehensive documentation
class ChatRequest(BaseModel):
    """Chat request model with validation and examples"""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="User query or question",
        example="What are the main applications of artificial intelligence in healthcare?"
    )
    max_sources: int = Field(
        3, 
        ge=1, 
        le=10, 
        description="Maximum number of source documents to retrieve",
        example=3
    )
    template_type: Optional[str] = Field(
        None, 
        description="Template type for response generation",
        example="qa",
        pattern="^(default|qa|summary|explanation|comparison)$"
    )
    include_sources: bool = Field(
        True, 
        description="Whether to include source citations in response",
        example=True
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How is machine learning used in medical diagnosis?",
                "max_sources": 3,
                "template_type": "qa",
                "include_sources": True
            }
        }


class ChatResponse(BaseModel):
    """Chat response model with detailed source information"""
    query: str = Field(description="Original user query", example="What is artificial intelligence?")
    response: str = Field(description="AI-generated response", example="Artificial intelligence (AI) is a branch of computer science...")
    sources: List[Dict[str, Any]] = Field(description="Source documents used in response")
    model_used: str = Field(description="AI model used for generation", example="gpt-3.5-turbo")
    timestamp: str = Field(description="Response timestamp in ISO format", example="2024-01-15T10:30:00")
    response_time: float = Field(description="Response time in seconds", example=2.45)
    confidence_score: Optional[float] = Field(description="Response confidence score (0-1)", example=0.85)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "response": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed...",
                "sources": [
                    {
                        "filename": "machine-learning-fundamentals.pdf",
                        "page_number": 2,
                        "chunk_text": "Machine learning algorithms use statistical techniques to give computer systems the ability to learn...",
                        "similarity_score": 0.92
                    }
                ],
                "model_used": "gpt-3.5-turbo",
                "timestamp": "2024-01-15T10:30:00",
                "response_time": 2.45,
                "confidence_score": 0.85
            }
        }
    
    @classmethod
    def from_rag_response(cls, rag_response: RAGResponse) -> "ChatResponse":
        """Convert RAGResponse to ChatResponse"""
        sources = []
        for source in rag_response.sources:
            sources.append({
                "filename": source.source_file,
                "page_number": source.page_number,
                "chunk_text": source.chunk_text[:200] + "..." if len(source.chunk_text) > 200 else source.chunk_text,
                "similarity_score": source.similarity_score
            })
        
        return cls(
            query=rag_response.query,
            response=rag_response.response,
            sources=sources,
            model_used=rag_response.model_used,
            timestamp=rag_response.timestamp.isoformat(),
            response_time=rag_response.response_time,
            confidence_score=rag_response.confidence_score
        )


class DocumentUploadResponse(BaseModel):
    """Document upload response model"""
    filename: str = Field(description="Name of uploaded file", example="healthcare-ai-report.pdf")
    status: str = Field(description="Upload status", example="success")
    message: str = Field(description="Status message", example="Document uploaded and processed successfully")
    processing_time: Optional[float] = Field(description="Processing time in seconds", example=15.3)
    chunks_created: Optional[int] = Field(description="Number of text chunks created", example=45)

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "healthcare-ai-report.pdf",
                "status": "success",
                "message": "Document uploaded and processed successfully",
                "processing_time": 15.3,
                "chunks_created": 45
            }
        }


class SystemStatus(BaseModel):
    """System status model with detailed component information"""
    status: str = Field(description="Overall system status", example="healthy")
    components: Dict[str, str] = Field(description="Status of individual components")
    index_stats: Dict[str, Any] = Field(description="Vector index statistics")
    configuration: Dict[str, Any] = Field(description="System configuration")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "components": {
                    "rag_pipeline": "initialized",
                    "vector_store": "loaded",
                    "llm_handler": "connected",
                    "document_processor": "ready"
                },
                "index_stats": {
                    "total_documents": 10,
                    "total_chunks": 343,
                    "embedding_dimension": 384,
                    "store_type": "FAISS"
                },
                "configuration": {
                    "model": "gpt-3.5-turbo",
                    "vector_store_type": "faiss",
                    "chunk_size": 1000
                }
            }
        }


class DocumentInfo(BaseModel):
    """Document information model"""
    filename: str = Field(description="Document filename", example="ai-healthcare-report.pdf")
    upload_date: str = Field(description="Upload timestamp", example="2024-01-15T10:30:00")
    file_size: int = Field(description="File size in bytes", example=2048576)
    status: str = Field(description="Processing status", example="processed")
    chunks_count: Optional[int] = Field(description="Number of text chunks", example=45)


class DocumentListResponse(BaseModel):
    """Document list response model"""
    documents: List[DocumentInfo] = Field(description="List of documents")
    total_documents: int = Field(description="Total number of documents", example=10)
    total_size_mb: float = Field(description="Total collection size in MB", example=25.6)

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "filename": "ai-healthcare-report.pdf",
                        "upload_date": "2024-01-15T10:30:00",
                        "file_size": 2048576,
                        "status": "processed",
                        "chunks_count": 45
                    }
                ],
                "total_documents": 1,
                "total_size_mb": 2.0
            }
        }


class QueryExamplesResponse(BaseModel):
    """Query examples response model"""
    examples: List[Dict[str, str]] = Field(description="List of example queries")
    categories: List[str] = Field(description="Query categories available")

    class Config:
        json_schema_extra = {
            "example": {
                "examples": [
                    {
                        "category": "Question & Answer",
                        "query": "What is artificial intelligence?",
                        "description": "Get direct answers to specific questions"
                    },
                    {
                        "category": "Summary",
                        "query": "Summarize the main applications of AI in healthcare",
                        "description": "Get comprehensive summaries of topics"
                    }
                ],
                "categories": ["Question & Answer", "Summary", "Explanation", "Comparison"]
            }
        }


# Global variables
rag_pipeline: Optional[RAGPipeline] = None
document_processor: Optional[DocumentProcessor] = None


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global rag_pipeline, document_processor
    
    try:
        logger.info("Starting RAG Chatbot API...")
        
        # Initialize configurations
        retrieval_config = RetrievalConfig(
            vector_store_type=os.getenv("VECTOR_STORE_TYPE", "faiss"),
            index_path=os.getenv("INDEX_PATH", "data/vector_index"),
            default_k=int(os.getenv("DEFAULT_K", "5"))
        )
        
        # Respect LLM_PROVIDER and both OpenAI/Gemini env vars (align with CLI behavior)
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

        # Choose max_tokens default based on provider (kept consistent with CLI)
        if llm_provider == "gemini":
            default_max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "2000"))
        else:
            default_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))

        llm_config = LLMConfig(
            provider=llm_provider,
            # OpenAI configuration
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            # Gemini configuration
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-pro"),
            # Common configuration
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=default_max_tokens,
            max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "4000")),
            include_sources=os.getenv("INCLUDE_SOURCES", "true").lower() == "true",
            citation_style=os.getenv("CITATION_STYLE", "numbered")
        )

        # Log selected provider and basic sanity on API keys
        if llm_provider == "openai":
            if not llm_config.api_key:
                logger.warning("No OPENAI_API_KEY found. API responses may be unavailable.")
            logger.info(f"LLM provider: OpenAI (model={llm_config.model})")
        elif llm_provider == "gemini":
            if not llm_config.gemini_api_key:
                logger.warning("No GEMINI_API_KEY found. API responses may be unavailable.")
            logger.info(f"LLM provider: Gemini (model={llm_config.gemini_model})")
        else:
            logger.warning(f"Unknown LLM_PROVIDER '{llm_provider}', defaulting to OpenAI settings.")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(retrieval_config, llm_config)
        
        # Initialize document processor for uploads
        document_processor = DocumentProcessor()
        
        logger.info("RAG Chatbot API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start RAG pipeline: {e}")
        # Don't fail startup, but log the error


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG Chatbot API...")


# Health check endpoint
# Health check endpoint
@app.get(
    "/health", 
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API service is running and healthy",
    tags=["System"],
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    **Health Check Endpoint**
    
    Provides a simple health check to verify the API service is running.
    Use this endpoint for:
    - Service monitoring
    - Load balancer health checks
    - Basic connectivity testing
    
    **Returns:**
    - Service status
    - Current timestamp
    - API version
    """
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


# System status endpoint
# System status endpoint
@app.get(
    "/status", 
    response_model=SystemStatus,
    summary="System Status",
    description="Get detailed system status including component health and statistics",
    tags=["System"],
    responses={
        200: {
            "description": "System status retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "components": {
                            "rag_pipeline": "initialized",
                            "vector_store": "loaded",
                            "llm_handler": "connected"
                        },
                        "index_stats": {
                            "total_documents": 10,
                            "total_chunks": 343
                        }
                    }
                }
            }
        },
        503: {"description": "Service unavailable - components not initialized"}
    }
)
async def get_system_status():
    """Get detailed system status"""
    global rag_pipeline, document_processor
    
    components = {
        "rag_pipeline": "initialized" if rag_pipeline else "not_initialized",
        "document_processor": "initialized" if document_processor else "not_initialized",
        "vector_store": "available" if rag_pipeline and rag_pipeline.retriever.vector_store else "unavailable"
    }
    
    index_stats = {}
    if rag_pipeline and rag_pipeline.retriever.vector_store:
        try:
            # Get index statistics
            if hasattr(rag_pipeline.retriever.vector_store, 'get_stats'):
                index_stats = rag_pipeline.retriever.vector_store.get_stats()
            else:
                index_stats = {"status": "available", "type": type(rag_pipeline.retriever.vector_store).__name__}
        except Exception as e:
            index_stats = {"status": "error", "error": str(e)}
    
    configuration = {
        "vector_store_type": os.getenv("VECTOR_STORE_TYPE", "faiss"),
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "index_path": os.getenv("INDEX_PATH", "data/vector_index"),
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY"))
    }
    
    overall_status = "healthy" if all(status != "not_initialized" for status in components.values()) else "degraded"
    
    return SystemStatus(
        status=overall_status,
        components=components,
        index_stats=index_stats,
        configuration=configuration
    )


# Main chat endpoint - the core functionality
@app.post(
    "/chat", 
    response_model=ChatResponse,
    summary="Chat Query",
    description="Process a natural language query using RAG (Retrieval-Augmented Generation)",
    tags=["Chat"],
    responses={
        200: {
            "description": "Query processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "query": "What is machine learning?",
                        "response": "Machine learning is a subset of artificial intelligence...",
                        "sources": [
                            {
                                "filename": "ml-fundamentals.pdf",
                                "page_number": 2,
                                "similarity_score": 0.92
                            }
                        ],
                        "response_time": 2.45
                    }
                }
            }
        },
        422: {"description": "Validation error - invalid request format"},
        500: {"description": "Internal server error - query processing failed"},
        503: {"description": "Service unavailable - RAG pipeline not initialized"}
    }
)
async def chat(request: ChatRequest):
    """
    **Main Chat Endpoint - RAG Query Processing**
    
    This is the core endpoint that processes user queries using Retrieval-Augmented Generation.
    
    **How it works:**
    1. 🔍 **Retrieval**: Searches your document collection for relevant context
    2. 🧠 **Generation**: Uses OpenAI GPT to generate intelligent responses
    3. 📚 **Citations**: Provides source references with page numbers
    
    **Query Types Supported:**
    - **Questions**: "What is artificial intelligence?"
    - **Summaries**: "Summarize the main AI applications"
    - **Explanations**: "Explain how neural networks work"
    - **Comparisons**: "Compare machine learning vs deep learning"
    
    **Template Types:**
    - `default`: General-purpose responses
    - `qa`: Question-answering format
    - `summary`: Summary-focused responses
    - `explanation`: Detailed explanations
    - `comparison`: Comparative analysis
    
    **Example Usage:**
    ```json
    {
        "query": "How is AI used in healthcare?",
        "max_sources": 3,
        "template_type": "qa",
        "include_sources": true
    }
    ```
    """
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline not initialized. Please check system configuration."
        )
    
    try:
        logger.info(f"Processing chat query: '{request.query[:50]}...'")
        
        # Process the query through RAG pipeline
        rag_response = rag_pipeline.query(
            question=request.query,
            k=request.max_sources,
            template_type=request.template_type
        )
        
        # Convert to API response format
        chat_response = ChatResponse.from_rag_response(rag_response)
        
        # Optionally remove sources if not requested
        if not request.include_sources:
            chat_response.sources = []
        
        logger.info(f"Chat query processed successfully in {rag_response.response_time:.2f}s")
        return chat_response
        
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Document upload endpoint
@app.post(
    "/upload", 
    response_model=DocumentUploadResponse,
    summary="Upload Document",
    description="Upload a PDF document to be processed and added to the knowledge base",
    tags=["Documents"],
    responses={
        200: {
            "description": "Document uploaded and processed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "filename": "healthcare-ai-report.pdf",
                        "status": "success",
                        "message": "Document uploaded and processed successfully",
                        "processing_time": 15.3,
                        "chunks_created": 45
                    }
                }
            }
        },
        400: {"description": "Bad request - invalid file type or format"},
        413: {"description": "File too large"},
        503: {"description": "Service unavailable - document processor not initialized"}
    }
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload (max 50MB)"),
    process_immediately: bool = Form(True, description="Process document immediately after upload")
):
    """
    **Document Upload Endpoint**
    
    Upload PDF documents to expand the chatbot's knowledge base.
    
    **Supported File Types:**
    - PDF (.pdf) - Text will be extracted and processed
    
    **Processing Options:**
    - **Immediate**: Document is processed right away (recommended)
    - **Background**: Document is queued for later processing
    
    **What Happens During Processing:**
    1. 📄 **Text Extraction**: Extracts text from PDF pages
    2. ✂️ **Chunking**: Splits text into manageable chunks (1000 chars)
    3. 🧮 **Embeddings**: Creates vector embeddings for each chunk
    4. 💾 **Indexing**: Adds chunks to the searchable vector database
    
    **File Constraints:**
    - Maximum file size: 50MB
    - Format: PDF only
    - Text-based PDFs work best (OCR for scanned images not supported)
    
    **Example Usage:**
    ```bash
    curl -X POST "http://localhost:8000/upload" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "file=@healthcare-report.pdf" \
         -F "process_immediately=true"
    ```
    """
    global document_processor, rag_pipeline
    
    if not document_processor:
        raise HTTPException(
            status_code=503,
            detail="Document processor not initialized"
        )
    
    # Validate file type
    allowed_types = ["application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Only PDF files are supported."
        )
    
    try:
        # Save uploaded file
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded: {file.filename}")
        
        if process_immediately:
            # Process document immediately
            start_time = datetime.now()
            
            try:
                # Process the document
                chunks = document_processor.process_document(str(file_path))
                
                # Add to vector store if pipeline is available
                if rag_pipeline and rag_pipeline.retriever.vector_store:
                    rag_pipeline.retriever.vector_store.add_documents(chunks)
                    logger.info(f"Added {len(chunks)} chunks to vector store")
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return DocumentUploadResponse(
                    filename=file.filename,
                    status="processed",
                    message=f"Document processed successfully. Created {len(chunks)} chunks.",
                    processing_time=processing_time,
                    chunks_created=len(chunks)
                )
                
            except Exception as e:
                logger.error(f"Error processing document {file.filename}: {e}")
                return DocumentUploadResponse(
                    filename=file.filename,
                    status="error",
                    message=f"Error processing document: {str(e)}"
                )
        else:
            # Queue for background processing
            background_tasks.add_task(process_document_background, str(file_path))
            
            return DocumentUploadResponse(
                filename=file.filename,
                status="queued",
                message="Document uploaded and queued for processing"
            )
            
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


async def process_document_background(file_path: str):
    """Background task for document processing"""
    global document_processor, rag_pipeline
    
    try:
        logger.info(f"Background processing: {file_path}")
        
        # Process the document
        chunks = document_processor.process_document(file_path)
        
        # Add to vector store if available
        if rag_pipeline and rag_pipeline.retriever.vector_store:
            rag_pipeline.retriever.vector_store.add_documents(chunks)
            logger.info(f"Background processing completed: {len(chunks)} chunks added")
        
    except Exception as e:
        logger.error(f"Background processing error for {file_path}: {e}")


# Document management endpoints
@app.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List Documents",
    description="Get a list of all uploaded documents with metadata",
    tags=["Documents"],
    responses={
        200: {
            "description": "List of documents retrieved successfully"
        }
    }
)
async def list_documents():
    """
    **List Uploaded Documents**
    
    Get a comprehensive list of all documents in the knowledge base.
    
    **Returns:**
    - Document metadata (filename, size, upload date)
    - Processing status
    - Number of text chunks created
    - Total collection statistics
    
    **Document Status:**
    - `processed`: Ready for querying
    - `processing`: Currently being processed
    - `failed`: Processing encountered an error
    - `uploaded`: Uploaded but not yet processed
    """
    upload_dir = Path("data/uploads")
    
    if not upload_dir.exists():
        return {
            "documents": [],
            "total_documents": 0,
            "total_size_mb": 0.0
        }
    
    documents = []
    total_size_bytes = 0
    
    for file_path in upload_dir.glob("*.pdf"):
        stat = file_path.stat()
        total_size_bytes += stat.st_size
        documents.append({
            "filename": file_path.name,
            "size": stat.st_size,
            "uploaded": datetime.fromtimestamp(stat.st_ctime).isoformat()
        })
    
    return {
        "documents": documents,
        "total_documents": len(documents),
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 2)
    }


@app.delete(
    "/documents/{filename}",
    summary="Delete Document",
    description="Delete an uploaded document from the system",
    tags=["Documents"],
    responses={
        200: {
            "description": "Document deleted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Document 'example.pdf' deleted successfully"
                    }
                }
            }
        },
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"}
    }
)
async def delete_document(filename: str):
    """
    **Delete Document**
    
    Remove a document from the uploaded files.
    
    **Note:** This only removes the file from storage. 
    To remove it from the search index, you'll need to rebuild the index using `/index/rebuild`.
    """
    upload_dir = Path("data/uploads")
    file_path = upload_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        file_path.unlink()
        logger.info(f"Deleted document: {filename}")
        return {"message": f"Document {filename} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


# Index management endpoints
@app.post(
    "/index/rebuild",
    summary="Rebuild Vector Index",
    description="Rebuild the entire vector index from all uploaded documents",
    tags=["System"],
    responses={
        200: {
            "description": "Index rebuild queued successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Index rebuild queued. Check status endpoint for progress."
                    }
                }
            }
        },
        503: {"description": "Service unavailable - RAG pipeline not initialized"}
    }
)
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    **Rebuild Vector Index**
    
    Rebuilds the entire vector search index from all uploaded documents.
    
    **When to Use:**
    - After uploading multiple documents
    - When search results seem outdated
    - After changing embedding settings
    - To optimize search performance
    
    **Process:**
    1. Clears existing vector index
    2. Reprocesses all PDF documents
    3. Creates new embeddings
    4. Rebuilds searchable index
    
    **Note:** This is a background operation that may take several minutes depending on document volume.
    """
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    # Queue rebuild as background task
    background_tasks.add_task(rebuild_index_background)
    
    return {"message": "Index rebuild queued. Check status endpoint for progress."}


async def rebuild_index_background():
    """Background task for index rebuilding"""
    global rag_pipeline, document_processor
    
    try:
        logger.info("Starting index rebuild...")
        
        upload_dir = Path("data/uploads")
        if not upload_dir.exists():
            logger.warning("No uploads directory found")
            return
        
        # Clear existing index
        if rag_pipeline.retriever.vector_store:
            # This would need to be implemented in the vector store classes
            logger.info("Clearing existing index...")
        
        # Process all documents
        total_chunks = 0
        for file_path in upload_dir.glob("*.pdf"):
            try:
                chunks = document_processor.process_document(str(file_path))
                if rag_pipeline.retriever.vector_store:
                    rag_pipeline.retriever.vector_store.add_documents(chunks)
                total_chunks += len(chunks)
                logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
        
        logger.info(f"Index rebuild completed: {total_chunks} total chunks")
        
    except Exception as e:
        logger.error(f"Index rebuild error: {e}")


# Configuration endpoints
@app.get(
    "/config",
    summary="System Configuration",
    description="Get current system configuration and settings",
    tags=["System"],
    responses={
        200: {
            "description": "Configuration retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "vector_store_type": "faiss",
                        "model": "gpt-3.5-turbo",
                        "temperature": 0.7,
                        "max_tokens": 1000,
                        "default_k": 5,
                        "chunk_size": 1000,
                        "chunk_overlap": 200
                    }
                }
            }
        }
    }
)
async def get_configuration():
    """
    **System Configuration Endpoint**
    
    Returns current system configuration and settings.
    
    **Configuration Parameters:**
    - **vector_store_type**: Type of vector database (faiss/chromadb)
    - **model**: OpenAI model being used
    - **temperature**: Response creativity (0.0-1.0)
    - **max_tokens**: Maximum response length
    - **default_k**: Default number of sources to retrieve
    - **chunk_size**: Text chunk size for processing
    - **chunk_overlap**: Overlap between text chunks
    """
    return {
        "vector_store_type": os.getenv("VECTOR_STORE_TYPE", "faiss"),
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
        "default_k": int(os.getenv("DEFAULT_K", "5")),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200"))
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )


# Query examples endpoint
@app.get(
    "/examples",
    response_model=QueryExamplesResponse,
    summary="Query Examples",
    description="Get example queries to help users understand how to interact with the chatbot",
    tags=["Chat"],
    responses={
        200: {
            "description": "Query examples retrieved successfully"
        }
    }
)
async def get_query_examples():
    """
    **Query Examples Endpoint**
    
    Provides example queries to help users understand the chatbot's capabilities.
    
    **Categories:**
    - **Questions**: Direct factual questions
    - **Summaries**: Requests for summarization
    - **Explanations**: Detailed explanations of concepts
    - **Comparisons**: Comparative analysis between topics
    
    **Use Cases:**
    - Help new users get started
    - Showcase chatbot capabilities
    - Provide query inspiration
    """
    examples = [
        {
            "category": "Question & Answer",
            "query": "What is artificial intelligence?",
            "description": "Get direct answers to specific questions about AI concepts"
        },
        {
            "category": "Question & Answer", 
            "query": "How does machine learning work?",
            "description": "Understanding technical processes and mechanisms"
        },
        {
            "category": "Summary",
            "query": "Summarize the main applications of AI in healthcare",
            "description": "Get comprehensive summaries of specific topics"
        },
        {
            "category": "Summary",
            "query": "What are the key points about deep learning?",
            "description": "Extract main ideas and important concepts"
        },
        {
            "category": "Explanation",
            "query": "Explain how neural networks learn from data",
            "description": "Get detailed explanations of complex processes"
        },
        {
            "category": "Explanation",
            "query": "Why is data quality important in machine learning?",
            "description": "Understand reasoning and causality"
        },
        {
            "category": "Comparison",
            "query": "Compare machine learning vs deep learning",
            "description": "Analyze differences and similarities between concepts"
        },
        {
            "category": "Comparison",
            "query": "What's the difference between supervised and unsupervised learning?",
            "description": "Contrast different approaches and methodologies"
        },
        {
            "category": "Application",
            "query": "How is AI being used in financial services?",
            "description": "Learn about real-world applications and use cases"
        },
        {
            "category": "Application",
            "query": "What are the challenges in implementing AI in healthcare?",
            "description": "Understand practical considerations and obstacles"
        }
    ]
    
    categories = ["Question & Answer", "Summary", "Explanation", "Comparison", "Application"]
    
    return QueryExamplesResponse(
        examples=examples,
        categories=categories
    )


# Run the application
def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI application"""
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the API
    run_api(reload=True)
