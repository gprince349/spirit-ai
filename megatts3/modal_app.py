"""
MegaTTS3 Voice Cloning Service on Modal

Deploys MegaTTS3 with WavVAE encoder on Modal with GPU.
Provides REST API endpoints for text-to-speech with voice cloning.

Deploy:
    cd megatts3 && modal deploy modal_app.py

Download models first:
    cd megatts3 && modal run modal_app.py

Test health:
    curl https://your-workspace--megatts3-tts-megatts3service-health.modal.run

Test generate:
    curl -X POST "https://your-url/generate" \\
        -H "Content-Type: application/json" \\
        -d '{"text": "Hello world", "voice": "default"}' \\
        --output test.wav
"""

import io
import os
import sys
import pickle
from pathlib import Path

import modal
from pydantic import BaseModel


# =============================================================================
# Request Models (for JSON body parsing)
# =============================================================================

class GenerateRequest(BaseModel):
    text: str
    voice: str = "default"
    time_step: int = 32
    p_w: float = 1.4
    t_w: float = 3.0


class UploadVoiceRequest(BaseModel):
    name: str
    audio_base64: str

# =============================================================================
# Modal Configuration
# =============================================================================

# Path to the tts module (in same directory as this file)
MEGATTS3_DIR = Path(__file__).parent

# Image with all MegaTTS3 dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        # Web framework (required for fastapi_endpoint)
        "fastapi[standard]",
        # Core ML
        "torch==2.1.0",
        "torchaudio==2.1.0", 
        "numpy<2.0.0",
        # Audio processing
        "librosa",
        "soundfile",
        "pydub==0.25.1",
        "pyloudnorm==0.1.1",
        # NLP/ML
        "transformers>=4.41.2,<=4.49.0",
        "x-transformers==1.44.4",
        "torchdiffeq==0.2.5",
        "openai-whisper==20240930",
        # Text processing
        "langdetect==1.0.9",
        "WeTextProcessing==1.0.4.1",
        "modelscope==1.22.2",
        # Utils
        "attrdict",
        "setproctitle",
        "huggingface_hub",
        "hf-transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    # Copy the tts module (located in same directory)
    .add_local_dir(
        local_path=str(MEGATTS3_DIR / "tts"),
        remote_path="/app/tts"
    )
)

# Persistent volume for model weights and voice references
volume = modal.Volume.from_name("megatts3-models", create_if_missing=True)

# Modal app
app = modal.App("megatts3-tts", image=image)

# =============================================================================
# Model Download Function
# =============================================================================

@app.function(volumes={"/models": volume}, timeout=600)
def download_models():
    """Download MegaTTS3 weights to Modal Volume (run once)."""
    from huggingface_hub import snapshot_download
    
    checkpoints_dir = Path("/models/checkpoints")
    
    if checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
        print("✅ Checkpoints already exist, skipping download")
        # No commit needed - nothing changed
        return {"status": "exists", "path": str(checkpoints_dir)}
    
    print("📥 Downloading MegaTTS3 weights from HuggingFace...")
    print("   This may take 5-10 minutes on first run...")
    
    snapshot_download(
        repo_id="mrfakename/MegaTTS3-VoiceCloning",
        local_dir=str(checkpoints_dir),
        local_dir_use_symlinks=False
    )
    
    # Create voices directory
    voices_dir = Path("/models/voices")
    voices_dir.mkdir(parents=True, exist_ok=True)
    
    volume.commit()
    print("✅ Download complete!")
    
    return {"status": "downloaded", "path": str(checkpoints_dir)}


# =============================================================================
# MegaTTS3 Service Class
# =============================================================================

@app.cls(
    gpu="A10G",  # $1.10/h - good balance of speed and cost
    volumes={"/models": volume},
    timeout=300,
    scaledown_window=300,  # 5 min cooldown to reduce cold starts
)
@modal.concurrent(max_inputs=10)  # Handle up to 10 concurrent requests per container
class MegaTTS3Service:
    """MegaTTS3 voice cloning service with REST API."""
    
    @modal.enter()
    def load_model(self):
        """Initialize the model when container starts."""
        import torch
        
        print("🔄 Loading MegaTTS3 model...")
        print(f"   GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")
        
        # Add tts module to path
        sys.path.insert(0, "/app")
        
        # Reload volume to get latest data
        volume.reload()
        
        # Check if models are downloaded - FAIL FAST if not
        # (Don't call download_models.remote() here - that wastes GPU time!)
        checkpoints_dir = Path("/models/checkpoints")
        if not checkpoints_dir.exists() or not any(checkpoints_dir.iterdir()):
            raise RuntimeError(
                "❌ Model checkpoints not found! "
                "Please run 'modal run megatts3_modal.py' first to download models."
            )
        
        # Import and initialize the inference pipeline
        from tts.infer_cli import MegaTTS3DiTInfer
        
        self.infer_pipe = MegaTTS3DiTInfer(ckpt_root="/models/checkpoints")
        self.voices_dir = Path("/models/voices")
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for preprocessed voice contexts
        self._voice_cache = {}
        
        print("✅ MegaTTS3 model loaded!")
    
    def _cleanup_memory(self):
        """Clean up GPU and system memory."""
        import gc
        import torch
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _reset_model(self):
        """Reset the inference pipeline to recover from CUDA errors."""
        import torch
        
        try:
            self._cleanup_memory()
            print("🔄 Reinitializing MegaTTS3 model...")
            
            from tts.infer_cli import MegaTTS3DiTInfer
            self.infer_pipe = MegaTTS3DiTInfer(ckpt_root="/models/checkpoints")
            
            print("✅ Model reinitialized!")
            return True
        except Exception as e:
            print(f"❌ Failed to reinitialize model: {e}")
            return False
    
    def _preprocess_audio_robust(self, audio_bytes: bytes, target_sr: int = 22050, max_duration: int = 28) -> bytes:
        """
        Robustly preprocess audio to prevent CUDA errors.
        
        - Converts to mono
        - Normalizes volume
        - Validates for NaN/Inf values
        - Limits duration to max_duration seconds
        """
        import numpy as np
        import librosa
        import soundfile as sf
        from pydub import AudioSegment
        from pydub.effects import normalize
        
        try:
            # Load with pydub for robust format handling
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Limit duration to prevent memory issues
            if len(audio) > max_duration * 1000:
                audio = audio[:max_duration * 1000]
            
            # Normalize audio to prevent clipping
            audio = normalize(audio)
            
            # Convert to target sample rate
            audio = audio.set_frame_rate(target_sr)
            
            # Export to bytes
            buffer = io.BytesIO()
            audio.export(
                buffer,
                format="wav",
                parameters=["-acodec", "pcm_s16le", "-ac", "1", "-ar", str(target_sr)]
            )
            wav_bytes = buffer.getvalue()
            
            # Validate the audio with librosa
            buffer.seek(0)
            wav, sr = librosa.load(buffer, sr=target_sr, mono=True)
            
            # Check for invalid values
            if np.any(np.isnan(wav)) or np.any(np.isinf(wav)):
                raise ValueError("Audio contains NaN or infinite values")
            
            # Ensure reasonable amplitude range
            if np.max(np.abs(wav)) < 1e-6:
                raise ValueError("Audio signal is too quiet")
            
            # Re-encode validated audio
            output_buffer = io.BytesIO()
            sf.write(output_buffer, wav, sr, format='WAV')
            return output_buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Audio preprocessing failed: {e}")
            raise ValueError(f"Failed to process audio: {str(e)}")
    
    def _get_voice_context(self, voice: str):
        """Get or load preprocessed voice context."""
        # Check in-memory cache first
        if voice in self._voice_cache:
            cached = self._voice_cache[voice]
            # Verify cache has loudness_prompt (old caches may not)
            if 'loudness_prompt' in cached:
                return cached
            else:
                print(f"⚠️ In-memory cache missing loudness_prompt, reloading...")
                del self._voice_cache[voice]
        
        voice_dir = self.voices_dir / voice
        context_path = voice_dir / "context.pkl"
        audio_path = voice_dir / "reference.wav"
        
        # Reload volume to get latest files
        volume.reload()
        
        if context_path.exists():
            # Load cached context from disk
            print(f"📂 Loading cached voice context: {voice}")
            with open(context_path, "rb") as f:
                context = pickle.load(f)
            
            # Check if context has loudness_prompt (old pickles may not)
            if 'loudness_prompt' not in context:
                print(f"⚠️ Context missing loudness_prompt, re-preprocessing...")
                if audio_path.exists():
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                    context = self.infer_pipe.preprocess(audio_bytes)
                    context['loudness_prompt'] = self.infer_pipe.loudness_prompt
                    
                    # Update pickle
                    with open(context_path, "wb") as f:
                        pickle.dump(context, f)
                    volume.commit()
            
            self._voice_cache[voice] = context
            return context
        
        if audio_path.exists():
            # Preprocess and cache
            print(f"🔄 Preprocessing voice: {voice}")
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            context = self.infer_pipe.preprocess(audio_bytes)
            
            # Include loudness_prompt in context (set by preprocess on self.infer_pipe)
            context['loudness_prompt'] = self.infer_pipe.loudness_prompt
            
            # Cache to disk
            with open(context_path, "wb") as f:
                pickle.dump(context, f)
            volume.commit()
            
            self._voice_cache[voice] = context
            return context
        
        return None
    
    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: GenerateRequest):
        """
        Generate speech from text with voice cloning.
        
        Args:
            request.text: Text to synthesize
            request.voice: Voice name (must be uploaded first, or "default")
            request.time_step: Diffusion inference steps (higher = better quality, slower)
            request.p_w: Intelligibility weight (1.0-5.0)
            request.t_w: Similarity weight (0.0-10.0, higher = more similar to reference)
        
        Returns:
            WAV audio bytes
        """
        import torch
        from fastapi import Response
        from fastapi.responses import JSONResponse
        
        text = request.text
        voice = request.voice
        time_step = request.time_step
        p_w = request.p_w
        t_w = request.t_w
        
        if not text or not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Text is required"}
            )
        
        print(f"🎤 Generating: '{text[:50]}...' voice={voice}")
        
        # Get voice context
        voice_context = self._get_voice_context(voice)
        if voice_context is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Voice '{voice}' not found. Upload a reference first."}
            )
        
        try:
            # Restore loudness_prompt to model (needed by forward())
            if 'loudness_prompt' in voice_context:
                self.infer_pipe.loudness_prompt = voice_context['loudness_prompt']
            
            # Generate audio
            wav_bytes = self.infer_pipe.forward(
                voice_context,
                text,
                time_step=time_step,
                p_w=p_w,
                t_w=t_w
            )
            
            # Clean up memory after successful generation
            self._cleanup_memory()
            
            print(f"✅ Generated {len(wav_bytes)} bytes")
            
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=speech.wav"}
            )
            
        except RuntimeError as cuda_error:
            error_str = str(cuda_error)
            if "CUDA" in error_str or "out of memory" in error_str.lower():
                print(f"⚠️ CUDA error detected: {cuda_error}")
                # Try to reset the model to recover
                if self._reset_model():
                    return JSONResponse(
                        status_code=503,
                        content={"error": "CUDA error occurred. Model has been reset. Please retry."}
                    )
                else:
                    return JSONResponse(
                        status_code=500,
                        content={"error": "CUDA error occurred and model reset failed."}
                    )
            else:
                self._cleanup_memory()
                return JSONResponse(
                    status_code=500,
                    content={"error": str(cuda_error)}
                )
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            self._cleanup_memory()
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    @modal.fastapi_endpoint(method="POST")
    def upload_voice(self, request: UploadVoiceRequest):
        """
        Upload a reference audio for voice cloning.
        
        Args:
            request.name: Voice name (e.g., "osho", "custom1")
            request.audio_base64: Base64 encoded audio file (WAV/MP3)
        
        Returns:
            Status dict
        """
        import base64
        from pydub import AudioSegment
        from fastapi.responses import JSONResponse
        
        name = request.name
        audio_base64 = request.audio_base64
        
        if not name or not name.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Voice name is required"}
            )
        
        if not audio_base64:
            return JSONResponse(
                status_code=400,
                content={"error": "Audio data is required"}
            )
        
        # Sanitize name
        name = name.strip().lower().replace(" ", "_")
        print(f"📤 Uploading voice: {name}")
        
        try:
            # Decode audio
            audio_bytes = base64.b64decode(audio_base64)
            
            # Robustly preprocess audio (mono, normalize, validate)
            processed_audio = self._preprocess_audio_robust(audio_bytes)
            
            # Get duration from processed audio
            audio = AudioSegment.from_file(io.BytesIO(processed_audio))
            duration_ms = len(audio)
            
            # Create voice directory
            voice_dir = self.voices_dir / name
            voice_dir.mkdir(parents=True, exist_ok=True)
            
            # Save reference
            reference_path = voice_dir / "reference.wav"
            with open(reference_path, "wb") as f:
                f.write(processed_audio)
            
            # Preprocess and cache context
            context = self.infer_pipe.preprocess(processed_audio)
            
            # Include loudness_prompt in context (set by preprocess on self.infer_pipe)
            context['loudness_prompt'] = self.infer_pipe.loudness_prompt
            
            context_path = voice_dir / "context.pkl"
            with open(context_path, "wb") as f:
                pickle.dump(context, f)
            
            # Update caches
            self._voice_cache[name] = context
            volume.commit()
            
            print(f"✅ Voice '{name}' uploaded ({duration_ms}ms)")
            
            return {
                "status": "success",
                "voice": name,
                "duration_ms": duration_ms
            }
            
        except ValueError as ve:
            # Audio validation errors
            print(f"❌ Audio validation failed: {ve}")
            return JSONResponse(
                status_code=400,
                content={"error": str(ve)}
            )
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    @modal.fastapi_endpoint(method="GET")
    def list_voices(self):
        """List all available voices."""
        voices = []
        
        if self.voices_dir.exists():
            for voice_dir in self.voices_dir.iterdir():
                if voice_dir.is_dir():
                    has_context = (voice_dir / "context.pkl").exists()
                    has_audio = (voice_dir / "reference.wav").exists()
                    voices.append({
                        "name": voice_dir.name,
                        "ready": has_context,
                        "has_audio": has_audio
                    })
        
        return {"voices": voices}
    
    @modal.fastapi_endpoint(method="GET")
    def health(self):
        """Health check endpoint."""
        import torch
        
        gpu_name = None
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        
        voice_count = 0
        if self.voices_dir.exists():
            voice_count = len([d for d in self.voices_dir.iterdir() if d.is_dir()])
        
        return {
            "status": "healthy",
            "model_loaded": hasattr(self, "infer_pipe"),
            "gpu": gpu_name,
            "gpu_memory": gpu_memory,
            "voices": voice_count
        }


# =============================================================================
# CLI Entrypoint
# =============================================================================

@app.local_entrypoint()
def main():
    """Download models when running `modal run megatts3_modal.py`."""
    print("📥 Downloading MegaTTS3 models...")
    result = download_models.remote()
    print(f"✅ Result: {result}")
    print("\n🚀 Now deploy with: modal deploy megatts3_modal.py")
