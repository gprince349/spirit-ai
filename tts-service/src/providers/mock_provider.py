"""Mock TTS provider for testing - generates sine wave audio."""

import io
import os
import struct
import math
import time
from typing import List

from .base import TTSProvider, TTSResult


class MockProvider(TTSProvider):
    """
    Mock TTS provider that generates sine wave audio.
    
    Useful for testing the API without loading actual TTS models.
    Generates audio proportional to text length (100ms per word).
    
    Environment variables:
        MOCK_TTS_DELAY: Delay in seconds to simulate GPU processing (default: 2.0)
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._voices = ["default", "osho"]
        self._languages = ["en", "hi"]
        # Configurable delay to simulate GPU processing
        self.delay_seconds = float(os.getenv("MOCK_TTS_DELAY", "2.0"))
    
    @property
    def name(self) -> str:
        return "mock"
    
    def generate(
        self, 
        text: str, 
        voice: str = "default",
        language: str = "en"
    ) -> TTSResult:
        """
        Generate sine wave audio based on text length.
        
        Args:
            text: Input text
            voice: Voice identifier (ignored in mock)
            language: Language code (ignored in mock)
            
        Returns:
            TTSResult with WAV audio bytes
        """
        start_time = time.time()
        text_preview = text[:40] + "..." if len(text) > 40 else text
        print(f"🎤 [MOCK TTS] START: '{text_preview}'")
        
        # Simulate GPU processing delay
        if self.delay_seconds > 0:
            print(f"   ⏳ Simulating {self.delay_seconds}s GPU delay...")
            time.sleep(self.delay_seconds)
        
        # Calculate duration: 100ms per word, minimum 500ms
        word_count = len(text.split())
        duration_ms = max(500, word_count * 100)
        duration_seconds = duration_ms / 1000.0
        
        # Generate sine wave samples
        num_samples = int(self.sample_rate * duration_seconds)
        frequency = 440.0  # A4 note
        
        samples = []
        for i in range(num_samples):
            t = i / self.sample_rate
            # Create a simple envelope (fade in/out)
            envelope = 1.0
            fade_samples = int(0.05 * self.sample_rate)  # 50ms fade
            if i < fade_samples:
                envelope = i / fade_samples
            elif i > num_samples - fade_samples:
                envelope = (num_samples - i) / fade_samples
            
            # Generate sine wave with envelope
            sample = envelope * 0.3 * math.sin(2 * math.pi * frequency * t)
            samples.append(sample)
        
        # Convert to WAV bytes
        wav_bytes = self._samples_to_wav(samples)
        
        elapsed = time.time() - start_time
        print(f"🎤 [MOCK TTS] DONE: '{text_preview}' ({elapsed:.2f}s)")
        
        return TTSResult(
            audio=wav_bytes,
            sample_rate=self.sample_rate,
            duration_ms=duration_ms,
            format="wav"
        )
    
    def get_voices(self) -> List[str]:
        return self._voices
    
    def get_languages(self) -> List[str]:
        return self._languages
    
    def _samples_to_wav(self, samples: List[float]) -> bytes:
        """Convert float samples to WAV bytes."""
        buffer = io.BytesIO()
        
        num_samples = len(samples)
        num_channels = 1
        bits_per_sample = 16
        byte_rate = self.sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = num_samples * block_align
        
        # WAV header
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', 36 + data_size))
        buffer.write(b'WAVE')
        
        # fmt chunk
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<I', 16))  # chunk size
        buffer.write(struct.pack('<H', 1))   # audio format (PCM)
        buffer.write(struct.pack('<H', num_channels))
        buffer.write(struct.pack('<I', self.sample_rate))
        buffer.write(struct.pack('<I', byte_rate))
        buffer.write(struct.pack('<H', block_align))
        buffer.write(struct.pack('<H', bits_per_sample))
        
        # data chunk
        buffer.write(b'data')
        buffer.write(struct.pack('<I', data_size))
        
        # Write samples as 16-bit integers
        for sample in samples:
            # Clamp and convert to 16-bit int
            sample_int = int(max(-1.0, min(1.0, sample)) * 32767)
            buffer.write(struct.pack('<h', sample_int))
        
        return buffer.getvalue()

