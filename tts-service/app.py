"""
TTS Service - FastAPI Application

A platform-agnostic Text-to-Speech service with support for
multiple providers (Mock, MegaTTS3, etc.)
"""

import os
import io
import base64
import time
from typing import Optional, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from providers import TTSProvider, MockProvider


# =============================================================================
# Configuration
# =============================================================================

PROVIDER_TYPE = os.getenv("TTS_PROVIDER", "mock")  # mock, megatts3


def get_provider() -> TTSProvider:
    """Factory function to get the configured TTS provider."""
    if PROVIDER_TYPE == "mock":
        return MockProvider()
    elif PROVIDER_TYPE == "megatts3":
        # Import here to avoid loading model unless needed
        from providers.megatts3_provider import MegaTTS3Provider
        return MegaTTS3Provider()
    else:
        raise ValueError(f"Unknown provider: {PROVIDER_TYPE}")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="TTS Service",
    description="Text-to-Speech service with voice cloning support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded provider
_provider: Optional[TTSProvider] = None


def get_tts_provider() -> TTSProvider:
    """Get or initialize the TTS provider."""
    global _provider
    if _provider is None:
        print(f"🔄 Initializing TTS provider: {PROVIDER_TYPE}")
        _provider = get_provider()
        print(f"✅ Provider ready: {_provider.name}")
    return _provider


# =============================================================================
# Request/Response Models
# =============================================================================

class TTSRequest(BaseModel):
    """Request model for TTS generation."""
    text: str
    voice: str = "default"
    language: str = "en"


class TTSJsonResponse(BaseModel):
    """Response model with base64 encoded audio."""
    success: bool
    audio: str  # base64 encoded
    format: str
    duration_ms: int
    text: str
    voice: str
    language: str


class VoiceInfo(BaseModel):
    """Information about available voices."""
    voices: List[str]
    languages: List[str]


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    provider = get_tts_provider()
    return provider.health_check()


@app.get("/voices", response_model=VoiceInfo)
def get_voices():
    """Get available voices and languages."""
    provider = get_tts_provider()
    return VoiceInfo(
        voices=provider.get_voices(),
        languages=provider.get_languages()
    )


@app.post("/tts")
def generate_speech(request: TTSRequest):
    """
    Generate speech from text.
    
    Returns WAV audio as a streaming response.
    """
    try:
        provider = get_tts_provider()
        
        start_time = time.time()
        result = provider.generate(
            text=request.text,
            voice=request.voice,
            language=request.language
        )
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        print(f"🎤 Generated {result.duration_ms}ms audio in {elapsed_ms}ms")
        
        return StreamingResponse(
            io.BytesIO(result.audio),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "X-Duration-Ms": str(result.duration_ms),
                "X-Generation-Ms": str(elapsed_ms)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/json", response_model=TTSJsonResponse)
def generate_speech_json(request: TTSRequest):
    """
    Generate speech from text.
    
    Returns base64 encoded audio in JSON response.
    """
    try:
        provider = get_tts_provider()
        
        start_time = time.time()
        result = provider.generate(
            text=request.text,
            voice=request.voice,
            language=request.language
        )
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        print(f"🎤 Generated {result.duration_ms}ms audio in {elapsed_ms}ms")
        
        audio_base64 = base64.b64encode(result.audio).decode("utf-8")
        
        return TTSJsonResponse(
            success=True,
            audio=audio_base64,
            format=result.format,
            duration_ms=result.duration_ms,
            text=request.text,
            voice=request.voice,
            language=request.language
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time TTS.
    
    Protocol:
    - Client sends: {"text": "Hello", "voice": "osho", "language": "en"}
    - Server responds: {"type": "audio", "data": "<base64>", "duration_ms": 500}
    - Server sends: {"type": "done"}
    
    For streaming multiple sentences, client can send multiple messages.
    """
    await websocket.accept()
    provider = get_tts_provider()
    
    print("🔌 WebSocket client connected")
    
    try:
        while True:
            # Receive text request
            data = await websocket.receive_json()
            
            text = data.get("text", "")
            voice = data.get("voice", "default")
            language = data.get("language", "en")
            
            if not text:
                await websocket.send_json({
                    "type": "error",
                    "message": "No text provided"
                })
                continue
            
            print(f"📝 WS request: '{text[:50]}...' voice={voice} lang={language}")
            
            try:
                start_time = time.time()
                result = provider.generate(
                    text=text,
                    voice=voice,
                    language=language
                )
                elapsed_ms = int((time.time() - start_time) * 1000)
                
                # Send audio response
                audio_base64 = base64.b64encode(result.audio).decode("utf-8")
                await websocket.send_json({
                    "type": "audio",
                    "data": audio_base64,
                    "format": result.format,
                    "duration_ms": result.duration_ms,
                    "generation_ms": elapsed_ms,
                    "text": text
                })
                
                print(f"✅ WS response: {result.duration_ms}ms audio in {elapsed_ms}ms")
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

