# 🌐 FastAPI Backend Documentation

## Overview

The RAG Chatbot API provides a comprehensive RESTful interface for document-based question answering using Retrieval-Augmented Generation (RAG). Built with FastAPI, it offers high performance, automatic documentation, and robust error handling.

## 🚀 Quick Start

### 1. Start the API Server

```bash
# Method 1: Using main_app.py
python main_app.py api

# Method 2: Direct uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Method 3: Custom configuration
python main_app.py api --host 127.0.0.1 --port 8080 --reload
```

### 2. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 📋 API Endpoints

### Core Chat Functionality

#### `POST /chat` - Main Query Endpoint
Process natural language queries using RAG.

**Request Example:**
```json
{
  "query": "What is artificial intelligence?",
  "max_sources": 3,
  "template_type": "qa",
  "include_sources": true
}
```

**Response Example:**
```json
{
  "query": "What is artificial intelligence?",
  "response": "Artificial intelligence (AI) is a branch of computer science...",
  "sources": [
    {
      "filename": "ai-introduction.pdf",
      "page_number": 2,
      "chunk_text": "AI systems are designed to perform tasks...",
      "similarity_score": 0.92
    }
  ],
  "model_used": "gpt-3.5-turbo",
  "timestamp": "2024-01-15T10:30:00",
  "response_time": 2.45,
  "confidence_score": 0.85
}
```

**Template Types:**
- `default`: General-purpose responses
- `qa`: Question-answering format
- `summary`: Summary-focused responses  
- `explanation`: Detailed explanations
- `comparison`: Comparative analysis

#### `GET /examples` - Query Examples
Get example queries to understand API capabilities.

### Document Management

#### `POST /upload` - Upload Documents
Upload PDF documents to expand the knowledge base.

**Form Data:**
- `file`: PDF file (max 50MB)
- `process_immediately`: Boolean (default: true)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf" \
     -F "process_immediately=true"
```

#### `GET /documents` - List Documents
Get list of all uploaded documents with metadata.

**Response Example:**
```json
{
  "documents": [
    {
      "filename": "ai-healthcare-report.pdf",
      "upload_date": "2024-01-15T10:30:00",
      "file_size": 2048576,
      "status": "processed",
      "chunks_count": 45
    }
  ],
  "total_documents": 1,
  "total_size_mb": 2.0
}
```

#### `DELETE /documents/{filename}` - Delete Document
Remove a document from the system.

### System Management

#### `GET /health` - Health Check
Basic service health verification.

#### `GET /status` - System Status
Detailed system status including component health.

#### `GET /config` - Configuration
Current system configuration and settings.

#### `POST /index/rebuild` - Rebuild Index
Rebuild the vector search index from all documents.

## 🧪 Testing the API

### Using the Test Script

```bash
# Run comprehensive API tests
python test_api.py
```

### Manual Testing with curl

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Upload a document
curl -X POST "http://localhost:8000/upload" \
     -F "file=@your-document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?", "max_sources": 3}'

# List documents
curl -X GET "http://localhost:8000/documents"
```

### Using Python Requests

```python
import requests

# Chat query
response = requests.post("http://localhost:8000/chat", json={
    "query": "How does AI work in healthcare?",
    "max_sources": 3,
    "include_sources": True
})

result = response.json()
print(f"Response: {result['response']}")
print(f"Sources: {len(result['sources'])}")
```

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=1000

# Vector Database
VECTOR_STORE_TYPE=faiss
INDEX_PATH=data/vector_index

# Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_K=5
```

### API Server Options

```python
# Start with custom settings
from src.api import run_api

run_api(
    host="0.0.0.0",      # Bind to all interfaces
    port=8000,           # Port number
    reload=True          # Auto-reload on changes (development)
)
```

## 📚 Response Models

### ChatResponse
- `query`: Original user query
- `response`: AI-generated answer
- `sources`: List of source documents used
- `model_used`: AI model identifier
- `timestamp`: Response generation time
- `response_time`: Processing time in seconds
- `confidence_score`: Response confidence (0-1)

### DocumentUploadResponse
- `filename`: Uploaded file name
- `status`: Processing status
- `message`: Status message
- `processing_time`: Time taken to process
- `chunks_created`: Number of text chunks

### SystemStatus
- `status`: Overall system health
- `components`: Individual component status
- `index_stats`: Vector database statistics
- `configuration`: Current settings

## 🛡️ Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found (document/endpoint)
- `413`: Payload Too Large (file size)
- `422`: Validation Error (request format)
- `500`: Internal Server Error
- `503`: Service Unavailable (not initialized)

### Error Response Format

```json
{
  "detail": "Error description",
  "status_code": 400,
  "error_type": "ValidationError"
}
```

## 🔍 Query Types & Examples

### Direct Questions
```json
{"query": "What is artificial intelligence?"}
```

### Summarization
```json
{
  "query": "Summarize the main AI applications in healthcare",
  "template_type": "summary"
}
```

### Explanations
```json
{
  "query": "Explain how neural networks learn",
  "template_type": "explanation"
}
```

### Comparisons
```json
{
  "query": "Compare machine learning vs deep learning",
  "template_type": "comparison"
}
```

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY=your_key_here
export VECTOR_STORE_TYPE=faiss

# Run in production mode
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📊 Monitoring & Logging

### Health Monitoring

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/status
```

### Log Levels
- `INFO`: General operation logs
- `WARNING`: Non-critical issues
- `ERROR`: Error conditions
- `DEBUG`: Detailed debugging (development)

## 🔗 Integration Examples

### JavaScript/Node.js

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'What is machine learning?',
    max_sources: 3
  })
});

const result = await response.json();
console.log(result.response);
```

### Python Client

```python
import requests

class RAGChatbot:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def ask(self, question, max_sources=3):
        response = requests.post(f"{self.base_url}/chat", json={
            "query": question,
            "max_sources": max_sources,
            "include_sources": True
        })
        return response.json()
    
    def upload_document(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/upload", files=files)
        return response.json()

# Usage
bot = RAGChatbot()
result = bot.ask("How does AI work?")
print(result['response'])
```

## 📈 Performance Tips

1. **Batch Uploads**: Upload multiple documents before rebuilding index
2. **Source Limiting**: Use appropriate `max_sources` values (3-5 recommended)
3. **Template Selection**: Choose specific templates for better responses
4. **Caching**: Implement response caching for frequently asked questions
5. **Rate Limiting**: Consider implementing rate limiting for production

## 🔧 Troubleshooting

### Common Issues

**503 Service Unavailable**
- Check if OpenAI API key is set
- Verify documents are uploaded and processed
- Ensure vector index is built

**422 Validation Error**
- Check request format matches API schema
- Verify required fields are provided
- Ensure data types are correct

**500 Internal Server Error**
- Check server logs for detailed error
- Verify OpenAI API key is valid
- Ensure sufficient disk space for processing

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with reload for development
python main_app.py api --reload
```

## 📝 API Schema

The complete API schema is available at:
- **Interactive**: http://localhost:8000/docs
- **JSON**: http://localhost:8000/openapi.json

This provides detailed information about all endpoints, request/response models, and validation rules.
