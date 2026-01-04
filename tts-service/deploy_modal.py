"""
Modal Deployment Configuration for TTS Service

This file defines the Modal app, image, and web endpoint.
Deploy with: modal deploy deploy_modal.py
"""

import modal

# =============================================================================
# Modal App Definition
# =============================================================================

app = modal.App("spirit-tts-service")

# Image for the Mock provider (lightweight, no GPU needed)
mock_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.34.0",
        "pydantic>=2.9.0",
    )
    .copy_local_dir("providers", "/app/providers")
    .copy_local_file("app.py", "/app/app.py")
)

# Image for MegaTTS3 provider (requires GPU)
megatts3_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.34.0",
        "pydantic>=2.9.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pynini",  # For G2P
        "gradio_client",  # Fallback to HF API
    )
    .run_commands(
        # Clone MegaTTS3 from HuggingFace
        "git clone https://huggingface.co/spaces/mrfakename/MegaTTS3-Voice-Cloning /app/megatts3",
    )
    .copy_local_dir("providers", "/app/providers")
    .copy_local_file("app.py", "/app/app.py")
)

# Volume for voice references (persistent storage)
voices_volume = modal.Volume.from_name("tts-voices", create_if_missing=True)


# =============================================================================
# Mock Provider Deployment (CPU only, for testing)
# =============================================================================

@app.function(
    image=mock_image,
    cpu=1,
    memory=512,
    timeout=300,
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def mock_web():
    """Deploy the TTS service with Mock provider."""
    import sys
    sys.path.insert(0, "/app")
    
    import os
    os.environ["TTS_PROVIDER"] = "mock"
    
    from app import app as fastapi_app
    return fastapi_app


# =============================================================================
# MegaTTS3 Provider Deployment (GPU required)
# =============================================================================

@app.function(
    image=megatts3_image,
    gpu="A10G",  # Best cost/performance for TTS
    cpu=4,
    memory=16384,  # 16GB RAM
    timeout=600,
    volumes={"/voices": voices_volume},
    allow_concurrent_inputs=5,
)
@modal.asgi_app()
def megatts3_web():
    """Deploy the TTS service with MegaTTS3 provider."""
    import sys
    sys.path.insert(0, "/app")
    
    import os
    os.environ["TTS_PROVIDER"] = "megatts3"
    os.environ["VOICES_DIR"] = "/voices"
    
    from app import app as fastapi_app
    return fastapi_app


# =============================================================================
# CLI Commands
# =============================================================================

@app.local_entrypoint()
def main():
    """Print deployment info."""
    print("🚀 Spirit TTS Service")
    print("")
    print("Deployment options:")
    print("  modal deploy deploy_modal.py        # Deploy both endpoints")
    print("  modal serve deploy_modal.py         # Local dev with hot reload")
    print("")
    print("Endpoints after deployment:")
    print("  Mock:     https://<user>--spirit-tts-service-mock-web.modal.run")
    print("  MegaTTS3: https://<user>--spirit-tts-service-megatts3-web.modal.run")
    print("")
    print("Upload voices:")
    print("  modal volume put tts-voices ./voices/osho /osho")

