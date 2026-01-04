"""Brain Service - FastAPI Application with REST + WebSocket endpoints."""

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from src.rag_chain import get_rag_chain
from src.vector_store import vector_store


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    question: str = Field(..., description="The question to ask Osho")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    language: Optional[str] = Field(default=None, description="Response language: 'en', 'hi', 'bilingual', or None for auto-detect")
    include_sources: bool = Field(default=True, description="Include source documents in response")


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    answer: str
    sources: Optional[list] = None
    timings: dict
    language: str
    model: str
    provider: str


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    ready: bool
    index_loaded: bool
    total_chunks: int


class StatsResponse(BaseModel):
    """Response model for /stats endpoint."""
    retriever: dict
    llm: dict
    ready: bool


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load index on startup."""
    print("=" * 60)
    print("Starting Brain Service...")
    print("=" * 60)
    
    # Load the RAG chain and index
    rag_chain = get_rag_chain()
    
    if rag_chain.load_index():
        print(f"✓ Index loaded: {vector_store.total_vectors} vectors")
    else:
        print("⚠ Warning: No index found. Run 'python ingest.py' first.")
    
    # Try to initialize LLM provider (optional - will fail gracefully)
    try:
        llm = rag_chain.llm
        print(f"✓ LLM Provider: {llm.provider_name} ({llm.model_name})")
    except ValueError as e:
        print(f"⚠ Warning: No LLM configured. Set GROQ_API_KEY or OPENAI_API_KEY in .env")
        print(f"  Error: {e}")
    
    print("=" * 60)
    print(f"Server ready at http://{settings.HOST}:{settings.PORT}")
    print("=" * 60)
    
    yield
    
    print("Shutting down Brain Service...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Spirit AI - Brain Service",
    description="RAG + LLM service for Osho AI assistant",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    rag_chain = get_rag_chain()
    return HealthResponse(
        status="healthy",
        ready=rag_chain.is_ready(),
        index_loaded=vector_store.total_vectors > 0,
        total_chunks=vector_store.total_chunks
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get service statistics."""
    rag_chain = get_rag_chain()
    return StatsResponse(**rag_chain.stats)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system (non-streaming).
    
    Returns a complete response with sources and timing information.
    """
    rag_chain = get_rag_chain()
    
    if not rag_chain.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run 'python ingest.py' first."
        )
    
    try:
        result = await rag_chain.query(
            question=request.question,
            top_k=request.top_k,
            language=request.language,
            include_sources=request.include_sources
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Sentence Buffer for TTS
# =============================================================================

class SentenceBuffer:
    """Buffer tokens into complete sentences for TTS."""
    
    def __init__(self, min_chars: int = 30):
        self.buffer = ""
        self.min_chars = min_chars
        self.sentence_endings = {'.', '!', '?', '।'}  # Including Hindi purna viram
        self.pause_markers = {'...', '—', '–'}
    
    def add(self, token: str) -> Optional[str]:
        """Add token, return sentence if complete."""
        self.buffer += token
        
        # Check for sentence endings
        for i, char in enumerate(self.buffer):
            if char in self.sentence_endings:
                # Check if followed by space or end of buffer
                if i + 1 >= len(self.buffer) or self.buffer[i + 1] in ' \n':
                    sentence = self.buffer[:i + 1].strip()
                    # Only return if we have enough content
                    if len(sentence) >= self.min_chars:
                        self.buffer = self.buffer[i + 1:].lstrip()
                        return sentence
        
        # Check for pause markers (like "...")
        for marker in self.pause_markers:
            if marker in self.buffer:
                idx = self.buffer.index(marker) + len(marker)
                sentence = self.buffer[:idx].strip()
                if len(sentence) >= self.min_chars:
                    self.buffer = self.buffer[idx:].lstrip()
                    return sentence
        
        return None
    
    def flush(self) -> Optional[str]:
        """Get remaining buffer content."""
        if self.buffer.strip():
            remaining = self.buffer.strip()
            self.buffer = ""
            return remaining
        return None


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat.
    
    Protocol:
    - Client sends: {"question": "...", "top_k": 5, "language": "en", "stream_mode": "sentence"}
    - stream_mode: "token" (default, individual tokens) or "sentence" (buffered for TTS)
    
    Response (sentence mode - for TTS):
    - Server sends: {"type": "sources", "sources": [...]}
    - Server sends: {"type": "sentence", "text": "Complete sentence.", "index": 0}
    - Server sends: {"type": "sentence", "text": "Another sentence.", "index": 1}
    - Server sends: {"type": "done", "full_response": "...", "timings": {...}}
    
    Response (token mode - raw):
    - Server sends: {"type": "sources", "sources": [...]}
    - Server sends: {"type": "token", "token": "..."} (multiple)
    - Server sends: {"type": "done", "full_response": "...", "timings": {...}}
    """
    await websocket.accept()
    
    rag_chain = get_rag_chain()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            
            question = data.get("question", "")
            top_k = data.get("top_k", settings.TOP_K)
            language = data.get("language", None)
            stream_mode = data.get("stream_mode", "sentence")  # "token" or "sentence"
            
            if not question:
                await websocket.send_json({
                    "type": "error",
                    "message": "Question is required"
                })
                continue
            
            if not rag_chain.is_ready():
                await websocket.send_json({
                    "type": "error",
                    "message": "Index not loaded. Run 'python ingest.py' first."
                })
                continue
            
            # Stream response
            try:
                if stream_mode == "sentence":
                    # Sentence-buffered streaming for TTS
                    buffer = SentenceBuffer(min_chars=30)
                    sentence_index = 0
                    
                    async for chunk in rag_chain.stream(
                        question=question,
                        top_k=top_k,
                        language=language
                    ):
                        if chunk["type"] == "sources":
                            await websocket.send_json(chunk)
                        elif chunk["type"] == "token":
                            # Buffer tokens into sentences
                            sentence = buffer.add(chunk["token"])
                            if sentence:
                                await websocket.send_json({
                                    "type": "sentence",
                                    "text": sentence,
                                    "index": sentence_index
                                })
                                sentence_index += 1
                        elif chunk["type"] == "done":
                            # Flush remaining buffer
                            remaining = buffer.flush()
                            if remaining:
                                await websocket.send_json({
                                    "type": "sentence",
                                    "text": remaining,
                                    "index": sentence_index
                                })
                            await websocket.send_json(chunk)
                else:
                    # Token-level streaming (original behavior)
                    async for chunk in rag_chain.stream(
                        question=question,
                        top_k=top_k,
                        language=language
                    ):
                        await websocket.send_json(chunk)
                        
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

