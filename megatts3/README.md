# MegaTTS3 Voice Cloning Service

GPU-powered text-to-speech service with voice cloning, deployed on Modal.

## Structure

```
megatts3/
├── modal_app.py      # Modal deployment (GPU service)
├── requirements.txt  # Python dependencies
├── tts/              # MegaTTS3 inference code (from HuggingFace)
│   ├── infer_cli.py  # Main inference pipeline
│   └── modules/      # Model components
└── README.md
```

## Deployment

### First-time setup (download models)

```bash
cd megatts3
modal run modal_app.py
```

This downloads ~2.5GB of model weights to a Modal Volume.

### Deploy the service

```bash
cd megatts3
modal deploy modal_app.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate speech from text |
| `/upload_voice` | POST | Upload reference audio for voice cloning |
| `/list_voices` | GET | List available voices |
| `/health` | GET | Health check |

### Generate Speech

```bash
curl -X POST "https://your-workspace--megatts3-tts-megatts3service-generate.modal.run" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "osho"}' \
  --output output.wav
```

### Upload Voice

```python
import base64
import requests

with open("reference.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://your-workspace--megatts3-tts-megatts3service-upload-voice.modal.run",
    json={"name": "myvoice", "audio_base64": audio_base64}
)
```

## Configuration

- **GPU**: A10G (24GB VRAM, $1.10/hr)
- **Cooldown**: 5 minutes (container stays warm)
- **Timeout**: 300 seconds per request

## Cost Estimate

- Scale-to-zero: No idle costs
- Per request: ~$0.003-0.01 depending on text length
- Cold start: ~40-50 seconds
- Warm request: ~7-10 seconds for 20 words

