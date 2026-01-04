"""LLM Provider Factory - Creates provider based on configuration."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import settings
from .base import LLMProvider
from .groq_provider import GroqProvider
from .openai_provider import OpenAIProvider


def get_llm_provider(
    provider: str = None,
    fallback: bool = True
) -> LLMProvider:
    """
    Get an LLM provider instance based on configuration.
    
    Args:
        provider: Provider name ('groq' or 'openai'). Defaults to settings.
        fallback: If True, try fallback provider if primary fails.
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If no valid provider can be created
    """
    provider = provider or settings.LLM_PROVIDER
    
    providers_to_try = [provider]
    if fallback:
        # Add fallback provider
        if provider == "groq":
            providers_to_try.append("openai")
        else:
            providers_to_try.append("groq")
    
    last_error = None
    
    for p in providers_to_try:
        try:
            if p == "groq":
                return GroqProvider()
            elif p == "openai":
                return OpenAIProvider()
            else:
                raise ValueError(f"Unknown provider: {p}")
        except ValueError as e:
            last_error = e
            if fallback:
                print(f"Provider '{p}' unavailable: {e}. Trying fallback...")
                continue
            raise
    
    raise ValueError(f"No LLM provider available. Last error: {last_error}")


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

