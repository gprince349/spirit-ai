"""Brain Service - FastAPI Application with REST + WebSocket endpoints."""

import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from src.rag_chain import get_rag_chain
from src.vector_store import vector_store


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for /query and /query/stream endpoints."""
    query: str = Field(..., description="The user's query/question")
    language: str = Field(default="en", description="Response language: 'en' or 'hi'")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation history (future use)")


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    type: str = "response"
    answer: str
    language: str


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
    
    Returns a complete response with timing information.
    """
    rag_chain = get_rag_chain()
    
    if not rag_chain.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run 'python ingest.py' first."
        )
    
    try:
        result = await rag_chain.query(
            question=request.query,
            language=request.language,
            include_sources=False
        )
        return QueryResponse(
            answer=result["answer"],
            language=result["language"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    SSE streaming endpoint for orchestrator consumption.
    
    Returns Server-Sent Events with tokens as they are generated.
    
    Event types:
    - event: token  -> data: {"token": "...", "index": 0}
    - event: done   -> data: {"total_tokens": N, "finish_reason": "stop"}
    - event: error  -> data: {"error": "...", "code": "..."}
    """
    rag_chain = get_rag_chain()
    
    if not rag_chain.is_ready():
        async def error_generator():
            error_data = json.dumps({
                "type": "error",
                "message": "Index not loaded",
                "code": "INDEX_ERROR"
            })
            yield f"event: error\ndata: {error_data}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    async def event_generator():
        token_index = 0
        try:
            async for chunk in rag_chain.stream(
                question=request.query,
                language=request.language
            ):
                if chunk["type"] == "token":
                    token_data = json.dumps({
                        "type": "token",
                        "token": chunk["token"],
                        "index": token_index
                    })
                    yield f"event: token\ndata: {token_data}\n\n"
                    token_index += 1
                elif chunk["type"] == "done":
                    done_data = json.dumps({
                        "type": "done",
                        "total_tokens": token_index,
                        "finish_reason": "stop"
                    })
                    yield f"event: done\ndata: {done_data}\n\n"
        except Exception as e:
            error_data = json.dumps({
                "type": "error",
                "message": str(e),
                "code": "LLM_ERROR"
            })
            yield f"event: error\ndata: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat.
    
    Protocol:
    - Client sends: {"query": "...", "language": "en", "session_id": "optional"}
    
    Response:
    - Server sends: {"type": "token", "token": "...", "index": 0}
    - Server sends: {"type": "done", "total_tokens": N, "finish_reason": "stop"}
    - Server sends: {"type": "error", "message": "...", "code": "..."}
    """
    await websocket.accept()
    
    rag_chain = get_rag_chain()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            
            query = data.get("query", "")
            language = data.get("language", "en")
            # session_id = data.get("session_id")  # For future use
            
            if not query:
                await websocket.send_json({
                    "type": "error",
                    "message": "Query is required",
                    "code": "VALIDATION_ERROR"
                })
                continue
            
            if not rag_chain.is_ready():
                await websocket.send_json({
                    "type": "error",
                    "message": "Index not loaded. Run 'python ingest.py' first.",
                    "code": "INDEX_ERROR"
                })
                continue
            
            # Stream response
            try:
                token_index = 0
                async for chunk in rag_chain.stream(
                    question=query,
                    language=language
                ):
                    if chunk["type"] == "token":
                        await websocket.send_json({
                            "type": "token",
                            "token": chunk["token"],
                            "index": token_index
                        })
                        token_index += 1
                    elif chunk["type"] == "done":
                        await websocket.send_json({
                            "type": "done",
                            "total_tokens": token_index,
                            "finish_reason": "stop"
                        })
                        
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "code": "LLM_ERROR"
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

