"""Abstract base class for TTS providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TTSConfig:
    """Configuration for TTS generation."""
    voice: str = "default"
    language: str = "en"
    sample_rate: int = 22050
    

@dataclass
class TTSResult:
    """Result from TTS generation."""
    audio: bytes
    sample_rate: int
    duration_ms: int
    format: str = "wav"


class TTSProvider(ABC):
    """
    Abstract base class for Text-to-Speech providers.
    
    This interface allows swapping between different TTS backends
    (Mock, MegaTTS3, Coqui, ElevenLabs, etc.) without changing the API.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    def generate(
        self, 
        text: str, 
        voice: str = "default",
        language: str = "en"
    ) -> TTSResult:
        """
        Generate audio from text.
        
        Args:
            text: The text to convert to speech
            voice: Voice identifier (e.g., "osho", "default")
            language: Language code (e.g., "en", "hi")
            
        Returns:
            TTSResult with audio bytes and metadata
        """
        pass
    
    @abstractmethod
    def get_voices(self) -> List[str]:
        """
        List available voices for this provider.
        
        Returns:
            List of voice identifiers
        """
        pass
    
    def get_languages(self) -> List[str]:
        """
        List supported languages.
        
        Returns:
            List of language codes
        """
        return ["en"]
    
    def health_check(self) -> dict:
        """
        Check provider health status.
        
        Returns:
            Dict with status information
        """
        return {
            "provider": self.name,
            "status": "healthy",
            "voices": self.get_voices(),
            "languages": self.get_languages()
        }

