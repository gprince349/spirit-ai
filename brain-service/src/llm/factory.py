"""LLM Provider Factory - Creates provider based on configuration."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings
from .base import LLMProvider
from .groq_provider import GroqProvider
from .openai_provider import OpenAIProvider


def get_llm_provider(provider: str = None) -> LLMProvider:
    """
    Get an LLM provider instance based on configuration.
    
    Args:
        provider: Provider name ('groq' or 'openai'). Defaults to LLM_PROVIDER env var.
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If provider is unknown or API key is missing
    """
    provider = provider or settings.LLM_PROVIDER
    
    if provider == "groq":
        return GroqProvider()
    elif provider == "openai":
        return OpenAIProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'groq' or 'openai'.")


# Lazy-loaded global instance
_llm_provider: LLMProvider = None


def get_default_provider() -> LLMProvider:
    """
    Get the default LLM provider (singleton).
    
    Returns:
        LLMProvider instance
    """
    global _llm_provider
    
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    
    return _llm_provider
