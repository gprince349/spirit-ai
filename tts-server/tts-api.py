from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from TTS.api import TTS
import io
import os
import torch
import base64
import tempfile
import time
import gc

app = FastAPI(title="Custom TTS API", version="1.0.0")

# Configuration
REFERENCE_VOICE = "ashish.wav"  # Your fixed reference
DEFAULT_LANGUAGE = "hi"
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global TTS instance - will be initialized lazily
tts = None

def get_tts():
    """Lazy initialization of TTS"""
    global tts
    if tts is None:
        # Init TTS with the target model
        tts = TTS(DEFAULT_MODEL).to(device)
    return tts

class TTSRequest(BaseModel):
    text: str
    language: str = DEFAULT_LANGUAGE
    use_reference: bool = True

class TTSResponse(BaseModel):
    success: bool
    audio: str  # base64
    format: str
    text: str
    language: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": DEFAULT_MODEL}

def safe_file_cleanup(file_path, max_retries=3, delay=0.1):
    """Safely delete a file with retries for Windows file locking issues"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
            return True
        except (OSError, PermissionError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                gc.collect()  # Force garbage collection
            else:
                print(f"Warning: Could not delete temp file {file_path}: {e}")
                return False
    return False

@app.post("/tts")
def generate_speech(request: TTSRequest):
    """Generate speech and return audio file"""
    temp_file_path = None
    try:
        tts_api = get_tts()
        
        # Create a temporary file with a unique name
        temp_fd, temp_file_path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(temp_fd)  # Close the file descriptor immediately
        
        # Generate speech directly to file
        tts_api.tts_to_file(
            text=request.text,
            speaker_wav=REFERENCE_VOICE if request.use_reference and os.path.exists(REFERENCE_VOICE) else None,
            language=request.language,
            file_path=temp_file_path
        )
        
        # Read the file into buffer
        audio_buffer = io.BytesIO()
        with open(temp_file_path, 'rb') as f:
            audio_buffer.write(f.read())
        
        audio_buffer.seek(0)
        
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=speech.wav"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file with retry logic
        if temp_file_path:
            safe_file_cleanup(temp_file_path)

@app.post("/tts-json", response_model=TTSResponse)
def generate_speech_json(request: TTSRequest):
    """Generate speech and return base64 JSON"""
    temp_file_path = None
    try:
        tts_api = get_tts()
        
        # Create a temporary file with a unique name
        temp_fd, temp_file_path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(temp_fd)  # Close the file descriptor immediately
        
        # Generate speech directly to file
        tts_api.tts_to_file(
            text=request.text,
            speaker_wav=REFERENCE_VOICE if request.use_reference and os.path.exists(REFERENCE_VOICE) else None,
            language=request.language,
            file_path=temp_file_path
        )
        
        # Read file and encode as base64
        with open(temp_file_path, 'rb') as f:
            audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode()
        
        return TTSResponse(
            success=True,
            audio=audio_base64,
            format="wav",
            text=request.text,
            language=request.language
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file with retry logic
        if temp_file_path:
            safe_file_cleanup(temp_file_path)

@app.get("/config")
def get_config():
    return {
        "reference_voice": REFERENCE_VOICE,
        "default_language": DEFAULT_LANGUAGE,
        "model": DEFAULT_MODEL,
        "device": str(device),
        "reference_exists": os.path.exists(REFERENCE_VOICE)
    }

# Run: uvicorn tts-api:app --host 0.0.0.0 --port 8000