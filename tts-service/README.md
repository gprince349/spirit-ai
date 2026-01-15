# TTS Service

Text-to-Speech service with voice cloning support. Uses pluggable providers (Mock for testing, MegaTTS3 for production).

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run Server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_PROVIDER` | `mock` | Default provider for all languages |
| `TTS_PROVIDER_EN` | (uses TTS_PROVIDER) | Provider for English |
| `TTS_PROVIDER_HI` | (uses TTS_PROVIDER) | Provider for Hindi |

**Single provider for all languages:**
```bash
TTS_PROVIDER=mock python app.py
```

**Different providers per language:**
```bash
TTS_PROVIDER_EN=mock TTS_PROVIDER_HI=megatts3 python app.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/voices` | GET | List available voices |
| `/tts` | POST | Generate audio (binary WAV) |
| `/tts/json` | POST | Generate audio (JSON with base64) |

## Request Format

```json
{
  "text": "Hello world",
  "voice": "osho",
  "language": "en",
  "format": "wav"
}
```

## Response Format (`/tts/json`)

```json
{
  "type": "audio",
  "audio": "base64-encoded-wav...",
  "format": "wav",
  "duration_ms": 1500,
  "text": "Hello world",
  "voice": "osho",
  "language": "en"
}
```

## File Structure

```
tts-service/
├── main.py              # FastAPI application
├── config.py            # Configuration settings
├── src/
│   ├── __init__.py
│   └── providers/
│       ├── __init__.py
│       ├── base.py          # TTSProvider abstract class
│       └── mock_provider.py # Mock implementation
├── voices/              # Voice reference files
├── infra/               # Deployment configs
│   └── deploy_modal.py
├── requirements.txt
└── README.md
```

## Providers

- **mock**: Generates sine wave audio for testing
- **megatts3**: MegaTTS3 model with voice cloning (requires GPU)

