# Orchestrator Service - WebSocket coordinator
# Connects Brain, TTS, and UI clients

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY config.py .
COPY src/ ./src/

# Runtime defaults (can be overridden)
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["python", "main.py"]
