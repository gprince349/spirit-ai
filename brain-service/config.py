"""Configuration for Brain Service."""

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==========================================================================
    # Embedding Model (Local)
    # ==========================================================================
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternatives:
    # - "BAAI/bge-small-en-v1.5" (better quality, slightly slower)
    # - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (multilingual)

    # ==========================================================================
    # Vector Store
    # ==========================================================================
    DATA_DIR: str = "data"
    DOCUMENTS_DIR: str = "data/documents"
    INDEX_DIR: str = "data/index"
    FAISS_INDEX_PATH: str = "data/index/faiss_index.bin"
    CHUNKS_PATH: str = "data/index/chunks.pkl"

    # ==========================================================================
    # Chunking
    # ==========================================================================
    CHUNK_SIZE: int = 512  # tokens/words per chunk
    CHUNK_OVERLAP: int = 50  # overlap between chunks

    # ==========================================================================
    # Retrieval
    # ==========================================================================
    TOP_K: int = 3  # number of chunks to retrieve (3 x 1024 words = ~3000 words context)

    # ==========================================================================
    # LLM Configuration
    # ==========================================================================
    LLM_PROVIDER: Literal["groq", "openai"] = "groq"

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Generation settings
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 100  # Shorter responses for voice (300 ~ 2-3 sentences)

    # ==========================================================================
    # Server
    # ==========================================================================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()

