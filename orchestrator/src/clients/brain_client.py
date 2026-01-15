"""Brain Service Client - SSE streaming client."""

import json
from typing import AsyncGenerator

import httpx

from config import BRAIN_URL
from src.logging_config import get_logger

logger = get_logger("orchestrator.brain_client")


async def stream_brain_response(
    query: str,
    language: str = "en",
    session_id: str = None
) -> AsyncGenerator[dict, None]:
    """
    Stream tokens from Brain Service.
    
    Args:
        query: User's question
        language: Response language ("en" or "hi")
        session_id: Optional session ID for conversation history
        
    Yields:
        Dict with token data: {"type": "token", "token": "...", "index": 0}
        Or done event: {"type": "done", "total_tokens": N, "finish_reason": "stop"}
        Or error: {"type": "error", "message": "...", "code": "..."}
    """
    url = f"{BRAIN_URL}/query/stream"
    
    payload = {
        "query": query,
        "language": language
    }
    if session_id:
        payload["session_id"] = session_id
    
    logger.info(f"Connecting to Brain service: {url}")
    logger.debug(f"Payload: {payload}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            url,
            json=payload,
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status_code != 200:
                logger.error(f"Brain service returned {response.status_code}")
                yield {
                    "type": "error",
                    "message": f"Brain service returned {response.status_code}",
                    "code": "BRAIN_ERROR"
                }
                return
            
            logger.info("Brain service stream connected")
            token_count = 0
            
            # Parse SSE stream
            async for line in response.aiter_lines():
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Parse data lines
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                        
                        if data.get("type") == "token":
                            token_count += 1
                            if token_count % 50 == 0:
                                logger.debug(f"Received {token_count} tokens...")
                        
                        yield data
                        
                        # Stop on done or error
                        if data.get("type") in ("done", "error"):
                            logger.info(f"Brain stream ended: {data.get('type')} (total tokens: {token_count})")
                            return
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse SSE data: {data_str[:50]}")
                        continue

