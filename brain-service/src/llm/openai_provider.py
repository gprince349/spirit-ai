"""OpenAI LLM Provider - High quality, reliable."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import AsyncGenerator
from openai import AsyncOpenAI

from config import settings
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM Provider.
    
    Uses OpenAI's GPT models. Good balance of quality and speed.
    Typical latency: ~200-500ms for first token.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self._model = model or settings.OPENAI_MODEL
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Generate a complete response."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

