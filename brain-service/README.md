# Brain Service

RAG + LLM service for Spirit AI. Retrieves relevant context from Osho's teachings and generates responses using Groq/OpenAI.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys (GROQ_API_KEY or OPENAI_API_KEY)
```

## Ingest Documents

```bash
# Place documents in data/ folder (supports .txt and .pdf)
python ingest.py
```

## Run Server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Service statistics |
| `/query` | POST | Non-streaming query |
| `/query/stream` | POST | SSE streaming query |
| `/ws/chat` | WS | WebSocket streaming |

## Request Format

```json
{
  "query": "What is love?",
  "language": "en",
  "session_id": "optional"
}
```

## Response Format

```json
{
  "type": "response",
  "answer": "...",
  "language": "en"
}
```

