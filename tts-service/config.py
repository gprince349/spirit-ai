"""TTS Service Configuration."""

import os

# Provider type per language: mock, megatts3
PROVIDER_TYPE_EN = os.getenv("TTS_PROVIDER_EN", "megatts3")
PROVIDER_TYPE_HI = os.getenv("TTS_PROVIDER_HI", "mock")  # Hindi uses mock for now

# MegaTTS3 Modal endpoint
MEGATTS3_BASE_URL = os.getenv(
    "MEGATTS3_BASE_URL",
    "https://gashishk349--megatts3-tts-megatts3service"
)

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))

