# Orchestrator Service

Connects UI clients to Brain and TTS services. Handles token streaming, sentence buffering, parallel TTS calls, and ordered audio delivery.

## Architecture

```
UI Client (WebSocket)
       ↓
   Orchestrator
       ├── Brain Client (SSE) → tokens
       ├── Sentence Buffer → sentences
       └── TTS Pipeline (REST) → audio
       ↓
UI Client (audio + captions)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `BRAIN_URL` | `http://localhost:8000` | Brain service URL |
| `TTS_URL` | `http://localhost:8002` | TTS service URL |
| `MAX_PARALLEL_TTS` | `3` | Concurrent TTS calls |
| `MIN_SENTENCE_WORDS` | `15` | Min words before sentence split |
| `MAX_SENTENCE_WORDS` | `30` | Force split at this word count |
| `TTS_TIMEOUT` | `30` | TTS call timeout (seconds) |
| `PORT` | `8003` | Server port |

## Logging

The service logs key events at various stages:

- **INFO**: Client connections, query start/complete, sentence ready, TTS calls
- **DEBUG**: Token streaming progress, buffer state, detailed TTS responses
- **WARNING**: TTS failures, invalid messages
- **ERROR**: Service errors, processing failures

Set `LOG_LEVEL=DEBUG` for detailed logs during development.

## WebSocket Protocol

**Connect:** `ws://localhost:8003/conversation`

**Client → Server:**
```json
{
  "type": "query",
  "query": "What is love?",
  "language": "en"
}
```

**Server → Client (sentence):**
```json
{
  "type": "sentence",
  "index": 0,
  "caption": "Love is a state of being...",
  "audio": "base64...",
  "format": "wav",
  "duration_ms": 3500
}
```

**Server → Client (complete):**
```json
{
  "type": "complete",
  "total_sentences": 5,
  "total_duration_ms": 18500,
  "total_generation_ms": 4200
}
```

## File Structure

```
orchestrator/
├── main.py              # FastAPI + WebSocket handler
├── config.py            # Configuration settings
├── src/
│   ├── __init__.py
│   ├── logging_config.py    # Logging setup
│   ├── sentence_buffer.py   # Token → Sentence logic
│   ├── tts_pipeline.py      # Parallel TTS with ordering
│   └── clients/
│       ├── __init__.py
│       ├── brain_client.py  # Brain service SSE client
│       └── tts_client.py    # TTS service REST client
├── requirements.txt
└── README.md
```

