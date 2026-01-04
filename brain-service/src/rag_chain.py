"""RAG Chain - Orchestrates retrieval and LLM generation."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import AsyncGenerator, Optional

from .retriever import retriever
from .prompts import get_system_prompt, build_rag_prompt, detect_language
from .llm import get_llm_provider, LLMProvider
from config import settings


class RAGChain:
    """
    RAG Chain that orchestrates:
    1. Query embedding
    2. Document retrieval
    3. Context building
    4. LLM generation (with streaming support)
    """
    
    def __init__(self, llm_provider: LLMProvider = None):
        """
        Initialize RAG chain.
        
        Args:
            llm_provider: LLM provider instance (auto-created if None)
        """
        self.retriever = retriever
        self._llm_provider = llm_provider
    
    @property
    def llm(self) -> LLMProvider:
        """Get LLM provider (lazy initialization)."""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider()
        return self._llm_provider
    
    def load_index(self) -> bool:
        """
        Load the vector store index.
        
        Returns:
            True if loaded successfully
        """
        from .vector_store import vector_store
        return vector_store.load()
    
    def is_ready(self) -> bool:
        """Check if RAG chain is ready."""
        return self.retriever.is_ready()
    
    async def query(
        self,
        question: str,
        top_k: int = None,
        language: str = None,
        include_sources: bool = True
    ) -> dict:
        """
        Execute a RAG query (non-streaming).
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            language: Response language ('en', 'hi', 'bilingual', or None for auto)
            include_sources: Whether to include source documents
            
        Returns:
            Dict with 'answer', 'sources', and 'timings'
        """
        timings = {}
        
        # Auto-detect language if not specified
        if language is None:
            language = detect_language(question)
        
        # Step 1: Retrieve relevant documents
        start = time.time()
        context, sources = self.retriever.retrieve_as_context(
            question,
            top_k=top_k or settings.TOP_K
        )
        timings["retrieval_ms"] = (time.time() - start) * 1000
        
        # Step 2: Build prompts
        system_prompt = get_system_prompt(language)
        user_prompt = build_rag_prompt(question, context, language)
        
        # Step 3: Generate response
        start = time.time()
        answer = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS
        )
        timings["llm_ms"] = (time.time() - start) * 1000
        
        timings["total_ms"] = timings["retrieval_ms"] + timings["llm_ms"]
        
        result = {
            "answer": answer,
            "timings": timings,
            "language": language,
            "model": self.llm.model_name,
            "provider": self.llm.provider_name
        }
        
        if include_sources:
            result["sources"] = [
                {
                    "text": s["text"][:300] + "..." if len(s["text"]) > 300 else s["text"],
                    "score": round(s["score"], 4),
                    "metadata": s["metadata"]
                }
                for s in sources
            ]
        
        return result
    
    async def stream(
        self,
        question: str,
        top_k: int = None,
        language: str = None
    ) -> AsyncGenerator[dict, None]:
        """
        Execute a RAG query with streaming response.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            language: Response language ('en', 'hi', 'bilingual', or None for auto)
            
        Yields:
            Dicts with either 'token' (partial), 'sources', or 'done' (final)
        """
        timings = {}
        
        # Auto-detect language if not specified
        if language is None:
            language = detect_language(question)
        
        # Step 1: Retrieve relevant documents
        start = time.time()
        context, sources = self.retriever.retrieve_as_context(
            question,
            top_k=top_k or settings.TOP_K
        )
        timings["retrieval_ms"] = (time.time() - start) * 1000
        
        # Yield sources first
        yield {
            "type": "sources",
            "sources": [
                {
                    "text": s["text"][:300] + "..." if len(s["text"]) > 300 else s["text"],
                    "score": round(s["score"], 4),
                    "metadata": s["metadata"]
                }
                for s in sources
            ],
            "retrieval_ms": timings["retrieval_ms"]
        }
        
        # Step 2: Build prompts
        system_prompt = get_system_prompt(language)
        user_prompt = build_rag_prompt(question, context, language)
        
        # Step 3: Stream response
        start = time.time()
        full_response = ""
        
        async for token in self.llm.stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS
        ):
            full_response += token
            yield {
                "type": "token",
                "token": token
            }
        
        timings["llm_ms"] = (time.time() - start) * 1000
        timings["total_ms"] = timings["retrieval_ms"] + timings["llm_ms"]
        
        # Yield completion info
        yield {
            "type": "done",
            "full_response": full_response,
            "timings": timings,
            "language": language,
            "model": self.llm.model_name,
            "provider": self.llm.provider_name
        }
    
    @property
    def stats(self) -> dict:
        """Get RAG chain statistics."""
        return {
            "retriever": self.retriever.stats,
            "llm": {
                "provider": self.llm.provider_name,
                "model": self.llm.model_name
            },
            "ready": self.is_ready()
        }


# Global singleton instance (lazy loaded)
_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """Get the global RAG chain instance."""
    global _rag_chain
    
    if _rag_chain is None:
        _rag_chain = RAGChain()
    
    return _rag_chain

