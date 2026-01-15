# Spirit AI

An AI-powered spiritual guide that channels the wisdom of Osho through voice conversations. Ask questions about life, love, meditation, and receive spoken responses in Osho's distinctive voice.

## Architecture

```
┌─────────────┐     WebSocket      ┌──────────────┐
│  UI Client  │◄──────────────────►│ Orchestrator │
└─────────────┘                    └──────┬───────┘
                                          │
                          ┌───────────────┴───────────────┐
                          │ SSE                      HTTP │
                          ▼                               ▼
                  ┌──────────────┐                ┌──────────────┐
                  │Brain Service │                │ TTS Service  │
                  │  (RAG+LLM)   │                │  (Wrapper)   │
                  └──────┬───────┘                └──────┬───────┘
                         │                               │ HTTP
                         ▼                               ▼
                  ┌──────────────┐                ┌──────────────┐
                  │  Groq API    │                │  MegaTTS3    │
                  │              │                │   (Modal)    │
                  └──────────────┘                └──────┬───────┘
                                                         │
                                                         ▼
                                                  ┌──────────────┐
                                                  │   A10G GPU   │
                                                  └──────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `brain-service` | 8000 | RAG + LLM - Retrieves Osho's teachings and generates responses |
| `orchestrator` | 8001 | WebSocket coordinator - Buffers sentences, manages TTS pipeline |
| `tts-service` | 8002 | TTS wrapper - Routes to appropriate voice provider |
| `megatts3` | Modal | GPU-powered voice cloning on Modal (scale-to-zero) |

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- Groq API key ([get one free](https://console.groq.com))
- Modal account ([sign up](https://modal.com)) for voice cloning

### 1. Clone & Setup

```bash
git clone https://github.com/your-repo/spirit-ai.git
cd spirit-ai

# Copy environment template
cp env.example .env
# Edit .env with your API keys
```

### 2. Deploy MegaTTS3 (Voice Cloning)

```bash
# Install Modal CLI
pip install modal
modal setup

# Download models (first time only)
cd megatts3
modal run modal_app.py

# Deploy the service
modal deploy modal_app.py
```

### 3. Run Services

**Option A: Docker Compose (Recommended)**
```bash
docker-compose up --build
```

**Option B: Local Development**
```bash
# Terminal 1 - Brain Service
cd brain-service
pip install -r requirements.txt
python main.py

# Terminal 2 - TTS Service
cd tts-service
pip install -r requirements.txt
python main.py

# Terminal 3 - Orchestrator
cd orchestrator
pip install -r requirements.txt
python main.py
```

### 4. Test

```bash
# Health checks
curl http://localhost:8000/health  # Brain
curl http://localhost:8002/health  # TTS

# Test Brain Service
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is love?", "language": "en"}'

# Test TTS Service
curl -X POST http://localhost:8002/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "osho"}' \
  --output test.wav

# Test Orchestrator (WebSocket)
npm install -g wscat
wscat -c ws://localhost:8001/ws/conversation
# Send: {"query": "What is meditation?", "language": "en"}
```

## Project Structure

```
spirit-ai/
├── brain-service/          # RAG + LLM service
│   ├── src/                # Application code
│   ├── data/               # Documents & vector index
│   ├── main.py             # Entry point
│   ├── Dockerfile
│   └── README.md
│
├── tts-service/            # TTS wrapper service
│   ├── src/providers/      # TTS provider implementations
│   ├── voices/             # Reference audio files
│   ├── main.py
│   ├── Dockerfile
│   └── README.md
│
├── megatts3/               # GPU TTS on Modal
│   ├── tts/                # MegaTTS3 inference code
│   ├── modal_app.py        # Modal deployment
│   └── README.md
│
├── orchestrator/           # WebSocket coordinator
│   ├── src/                # Clients, pipeline, buffer
│   ├── main.py
│   ├── Dockerfile
│   └── README.md
│
├── ui-client/              # Frontend (Next.js)
├── osho-scraper/           # Data collection scripts
├── docs/                   # Documentation
│
├── docker-compose.yml      # Run all services
├── env.example             # Environment template
└── README.md               # This file
```

## Configuration

All configuration is done through environment variables. See `env.example` for all options.

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (required) | - |
| `LLM_PROVIDER` | LLM provider | `groq` |
| `TTS_PROVIDER_EN` | English TTS provider | `megatts3` |
| `TTS_PROVIDER_HI` | Hindi TTS provider | `mock` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Development

### Ingest Documents

```bash
cd brain-service
python ingest.py
```

### Upload Voice Reference

```python
import base64
import requests

with open("reference.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

requests.post(
    "https://your-modal-url/upload_voice",
    json={"name": "osho", "audio_base64": audio_b64}
)
```

### Run Individual Services

```bash
# Docker
docker-compose up brain-service

# Local
cd brain-service && python main.py
```

## Cost Estimates

| Component | Cost |
|-----------|------|
| Groq API | Free tier: 30 req/min |
| Modal MegaTTS3 | ~$1.10/hr (A10G), scale-to-zero |
| Docker hosting | Depends on provider |

## License

MIT
