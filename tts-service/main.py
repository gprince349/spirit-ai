"""
TTS Service - FastAPI Application

A platform-agnostic Text-to-Speech service with support for
multiple providers (Mock, MegaTTS3, etc.)
"""

import io
import base64
import time
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import PROVIDER_TYPE_EN, PROVIDER_TYPE_HI, HOST, PORT
from src.providers import TTSProvider, MockProvider


# =============================================================================
# Provider Factory
# =============================================================================

def _create_provider(provider_type: str) -> TTSProvider:
    """Create a provider by type."""
    if provider_type == "mock":
        return MockProvider()
    elif provider_type == "megatts3":
        from src.providers.megatts3_provider import MegaTTS3Provider
        return MegaTTS3Provider()
    else:
        raise ValueError(f"Unknown provider: {provider_type}")


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

# Lazy-loaded providers (one per language)
_provider_en: Optional[TTSProvider] = None
_provider_hi: Optional[TTSProvider] = None


def get_provider_for_language(language: str) -> TTSProvider:
    """Get or initialize the TTS provider for a specific language."""
    global _provider_en, _provider_hi
    
    if language == "hi":
        if _provider_hi is None:
            print(f"🔄 Initializing Hindi TTS provider: {PROVIDER_TYPE_HI}")
            _provider_hi = _create_provider(PROVIDER_TYPE_HI)
            print(f"✅ Hindi provider ready: {_provider_hi.name}")
        return _provider_hi
    else:
        # Default to English
        if _provider_en is None:
            print(f"🔄 Initializing English TTS provider: {PROVIDER_TYPE_EN}")
            _provider_en = _create_provider(PROVIDER_TYPE_EN)
            print(f"✅ English provider ready: {_provider_en.name}")
        return _provider_en


# =============================================================================
# Request/Response Models
# =============================================================================

class TTSRequest(BaseModel):
    """Request model for TTS generation."""
    text: str
    voice: str = "default"
    language: str = "en"
    format: str = "wav"


class TTSJsonResponse(BaseModel):
    """Response model with base64 encoded audio."""
    type: str = "audio"
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
    return {
        "status": "healthy",
        "providers": {
            "en": PROVIDER_TYPE_EN,
            "hi": PROVIDER_TYPE_HI
        }
    }


@app.get("/voices", response_model=VoiceInfo)
def get_voices():
    """Get available voices and languages."""
    provider_en = get_provider_for_language("en")
    return VoiceInfo(
        voices=provider_en.get_voices(),
        languages=["en", "hi"]
    )


@app.post("/tts")
def generate_speech(request: TTSRequest):
    """
    Generate speech from text.
    
    Returns WAV audio as a streaming response.
    """
    try:
        provider = get_provider_for_language(request.language)
        
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
        provider = get_provider_for_language(request.language)
        
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
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

