"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers must implement these methods to ensure
    a consistent interface across different backends.
    """
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a complete response.
        
        Args:
            prompt: User prompt/question
            system_prompt: System instruction prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens.
        
        Args:
            prompt: User prompt/question
            system_prompt: System instruction prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Yields:
            Individual tokens/chunks of the response
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name being used."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name (e.g., 'groq', 'openai')."""
        pass

