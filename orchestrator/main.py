"""Orchestrator Service - Connects UI, Brain, and TTS."""

import asyncio
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import HOST, PORT, LOG_LEVEL
from src.clients import stream_brain_response
from src.sentence_buffer import SentenceBuffer
from src.tts_pipeline import TTSPipeline
from src.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(level=LOG_LEVEL)
logger = get_logger("orchestrator.main")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Spirit AI - Orchestrator",
    description="Connects UI clients to Brain and TTS services",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "orchestrator"}


# =============================================================================
# WebSocket Handler
# =============================================================================

@app.websocket("/conversation")
async def conversation(websocket: WebSocket):
    """
    WebSocket endpoint for voice conversations.
    
    Protocol:
    - Client sends: {"type": "query", "query": "...", "language": "en"}
    - Server sends: {"type": "sentence", "index": 0, "caption": "...", "audio": "base64...", "duration_ms": 3500}
    - Server sends: {"type": "complete", "total_sentences": 5, "total_duration_ms": 18500}
    - Server sends: {"type": "error", "message": "...", "code": "..."}
    """
    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"Client connected [client_id={client_id}]")
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            
            msg_type = data.get("type")
            if msg_type != "query":
                logger.warning(f"Invalid message type received: {msg_type} [client_id={client_id}]")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "code": "INVALID_TYPE"
                })
                continue
            
            query = data.get("query", "").strip()
            language = data.get("language", "en")
            session_id = data.get("session_id")
            
            if not query:
                logger.warning(f"Empty query received [client_id={client_id}]")
                await websocket.send_json({
                    "type": "error",
                    "message": "Query is required",
                    "code": "VALIDATION_ERROR"
                })
                continue
            
            logger.info(f"Query received: '{query[:50]}...' lang={language} [client_id={client_id}]")
            
            # Process the query
            await process_query(websocket, query, language, session_id)
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected [client_id={client_id}]")
    except Exception as e:
        logger.error(f"WebSocket error: {e} [client_id={client_id}]", exc_info=True)


async def process_query(
    websocket: WebSocket,
    query: str,
    language: str,
    session_id: Optional[str]
):
    """
    Process a single query: Brain → Buffer → TTS → Client.
    
    Uses push-based TTS pipeline - results are pushed to a queue as they complete.
    """
    start_time = time.time()
    buffer = SentenceBuffer()
    pipeline = TTSPipeline(voice="osho", language=language)
    
    sentence_index = 0
    total_duration_ms = 0
    token_count = 0
    
    logger.info(f"Processing query: '{query[:30]}...' lang={language}")
    
    async def result_sender():
        """Background task that sends TTS results as they are pushed to queue."""
        nonlocal total_duration_ms
        sent_count = 0
        
        while True:
            # Block until next result is pushed to queue
            result = await pipeline.get_result()
            
            # None sentinel means no more results
            if result is None:
                logger.debug("Result sender received sentinel, stopping")
                break
            
            total_duration_ms += result.duration_ms
            await send_result(websocket, result)
            sent_count += 1
            logger.info(f"📤 Sent result [{result.index}] to client")
    
    try:
        # Start the result sender in background
        sender_task = asyncio.create_task(result_sender())
        
        # Stream tokens from Brain Service
        logger.debug("Starting Brain service stream")
        async for event in stream_brain_response(query, language, session_id):
            event_type = event.get("type")
            
            if event_type == "error":
                logger.error(f"Brain service error: {event.get('message')}")
                await websocket.send_json({
                    "type": "error",
                    "message": event.get("message", "Brain service error"),
                    "code": event.get("code", "BRAIN_ERROR")
                })
                sender_task.cancel()
                return
            
            if event_type == "token":
                token = event.get("token", "")
                token_count += 1
                
                # Add token to buffer, check if sentence is ready
                sentence = buffer.add(token)
                if sentence:
                    # Submit sentence to TTS pipeline (non-blocking)
                    logger.info(f"Sentence ready [{sentence_index}]: '{sentence[:40]}...' ({len(sentence.split())} words)")
                    pipeline.submit(sentence_index, sentence)
                    sentence_index += 1
            
            elif event_type == "done":
                logger.info(f"Brain stream complete: {token_count} tokens received")
                
                # Flush remaining buffer
                remaining = buffer.flush()
                if remaining:
                    logger.info(f"Flushing remaining buffer [{sentence_index}]: '{remaining[:40]}...'")
                    pipeline.submit(sentence_index, remaining)
                    sentence_index += 1
        
        # Signal no more sentences - will push sentinel after all TTS complete
        pipeline.mark_complete()
        
        # Wait for all results to be sent
        await sender_task
        
        # Send completion message
        elapsed_ms = int((time.time() - start_time) * 1000)
        await websocket.send_json({
            "type": "complete",
            "total_sentences": sentence_index,
            "total_duration_ms": total_duration_ms,
            "total_generation_ms": elapsed_ms
        })
        
        logger.info(f"Query complete: {sentence_index} sentences, {total_duration_ms}ms audio, {elapsed_ms}ms total")
        
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        await websocket.send_json({
            "type": "error",
            "message": str(e),
            "code": "PROCESSING_ERROR"
        })
    finally:
        await pipeline.close()


async def send_result(websocket: WebSocket, result):
    """Send a single TTS result to the client."""
    message = {
        "type": "sentence",
        "index": result.index,
        "caption": result.caption,
        "audio": result.audio,
        "format": result.format,
        "duration_ms": result.duration_ms
    }
    
    if result.error:
        message["error"] = result.error
        logger.warning(f"Sending sentence [{result.index}] with TTS error: {result.error}")
    else:
        logger.debug(f"Sending sentence [{result.index}]: {result.duration_ms}ms audio")
    
    await websocket.send_json(message)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

