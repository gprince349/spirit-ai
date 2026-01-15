"""MegaTTS3 provider - calls Modal-deployed MegaTTS3 service."""

import os
import httpx
from typing import List

from .base import TTSProvider, TTSResult


class MegaTTS3Provider(TTSProvider):
    """
    MegaTTS3 provider that calls the Modal-deployed MegaTTS3 service.
    
    Provides high-quality voice cloning using MegaTTS3 with WavVAE encoder.
    """
    
    # Modal endpoint URLs
    DEFAULT_BASE_URL = "https://gashishk349--megatts3-tts-megatts3service"
    
    def __init__(
        self,
        base_url: str = None,
        default_voice: str = "osho",
        timeout: float = 120.0
    ):
        """
        Initialize MegaTTS3 provider.
        
        Args:
            base_url: Base URL for Modal endpoints (without method suffix)
            default_voice: Default voice to use if none specified
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv(
            "MEGATTS3_BASE_URL", 
            self.DEFAULT_BASE_URL
        )
        self.default_voice = default_voice
        self.timeout = timeout
        
        # Endpoint URLs
        self.generate_url = f"{self.base_url}-generate.modal.run"
        self.list_voices_url = f"{self.base_url}-list-voices.modal.run"
        self.health_url = f"{self.base_url}-health.modal.run"
    
    @property
    def name(self) -> str:
        return "megatts3"
    
    def generate(
        self, 
        text: str, 
        voice: str = "default",
        language: str = "en"
    ) -> TTSResult:
        """
        Generate speech using MegaTTS3 on Modal.
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier (must be uploaded to Modal first)
            language: Language code (en supported, hi experimental)
            
        Returns:
            TTSResult with WAV audio bytes
        """
        # Use default voice if "default" is passed
        if voice == "default":
            voice = self.default_voice
        
        # Make request to Modal
        payload = {
            "text": text,
            "voice": voice,
            "time_step": 32,  # Diffusion steps
            "p_w": 1.4,       # Intelligibility weight
            "t_w": 3.0        # Similarity weight
        }
        
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.generate_url, json=payload)
            
            if response.status_code != 200:
                error_msg = response.text[:200]
                raise RuntimeError(f"MegaTTS3 generation failed: {error_msg}")
            
            audio_bytes = response.content
        
        # Estimate duration from audio size (16-bit mono @ 24kHz)
        # WAV header is 44 bytes, then 2 bytes per sample at 24kHz
        sample_rate = 24000
        audio_data_size = len(audio_bytes) - 44
        num_samples = audio_data_size // 2
        duration_ms = int((num_samples / sample_rate) * 1000)
        
        return TTSResult(
            audio=audio_bytes,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            format="wav"
        )
    
    def get_voices(self) -> List[str]:
        """Get list of available voices from Modal."""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(self.list_voices_url)
                if response.status_code == 200:
                    data = response.json()
                    return [v["name"] for v in data.get("voices", [])]
        except Exception:
            pass
        
        # Fallback
        return [self.default_voice]
    
    def get_languages(self) -> List[str]:
        """MegaTTS3 supports English and Chinese natively."""
        return ["en", "zh"]
    
    def health_check(self) -> dict:
        """Check Modal service health."""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(self.health_url)
                if response.status_code == 200:
                    modal_health = response.json()
                    return {
                        "provider": self.name,
                        "status": "healthy",
                        "modal_status": modal_health,
                        "endpoint": self.base_url
                    }
        except Exception as e:
            return {
                "provider": self.name,
                "status": "unhealthy",
                "error": str(e),
                "endpoint": self.base_url
            }
        
        return {
            "provider": self.name,
            "status": "unhealthy",
            "endpoint": self.base_url
        }

