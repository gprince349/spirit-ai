"""TTS Service Client - REST client for TTS service."""

import asyncio
from typing import Optional
from dataclasses import dataclass

import httpx

from config import TTS_URL, TTS_TIMEOUT
from src.logging_config import get_logger

logger = get_logger("orchestrator.tts_client")


@dataclass
class TTSResponse:
    """Response from TTS service."""
    audio: Optional[str]  # base64 encoded
    format: str
    duration_ms: int
    text: str
    error: Optional[str] = None


class TTSClient:
    """
    Async client for TTS service.
    
    Handles HTTP calls to /tts/json endpoint.
    Manages connection pooling via shared httpx client.
    """
    
    def __init__(self, base_url: str = TTS_URL, timeout: int = TTS_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def generate(
        self,
        text: str,
        voice: str = "osho",
        language: str = "en"
    ) -> TTSResponse:
        """
        Generate audio from text.
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier
            language: Language code ("en" or "hi")
            
        Returns:
            TTSResponse with audio data or error
        """
        logger.debug(f"TTS request: voice={voice}, lang={language}, text='{text[:30]}...'")
        
        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/tts/json",
                json={
                    "text": text,
                    "voice": voice,
                    "language": language,
                    "format": "wav"
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"TTS service returned {response.status_code}")
                return TTSResponse(
                    audio=None,
                    format="wav",
                    duration_ms=0,
                    text=text,
                    error=f"TTS returned {response.status_code}"
                )
            
            data = response.json()
            logger.debug(f"TTS response: {data.get('duration_ms', 0)}ms audio")
            return TTSResponse(
                audio=data.get("audio"),
                format=data.get("format", "wav"),
                duration_ms=data.get("duration_ms", 0),
                text=text
            )
            
        except asyncio.TimeoutError:
            logger.error(f"TTS timeout after {self.timeout}s")
            return TTSResponse(
                audio=None,
                format="wav",
                duration_ms=0,
                text=text,
                error="TTS timeout"
            )
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return TTSResponse(
                audio=None,
                format="wav",
                duration_ms=0,
                text=text,
                error=str(e)
            )
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

