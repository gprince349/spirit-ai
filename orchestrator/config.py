"""Orchestrator Service Configuration."""

import os


# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR

# Service URLs
BRAIN_URL = os.getenv("BRAIN_URL", "http://localhost:8000")
TTS_URL = os.getenv("TTS_URL", "http://localhost:8002")

# TTS Pipeline
MAX_PARALLEL_TTS = int(os.getenv("MAX_PARALLEL_TTS", "3"))
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "180"))  # MegaTTS3 can take 30-60s per sentence

# Sentence Buffer
MIN_SENTENCE_WORDS = int(os.getenv("MIN_SENTENCE_WORDS", "15"))
MAX_SENTENCE_WORDS = int(os.getenv("MAX_SENTENCE_WORDS", "30"))

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

