"""TTS Pipeline - Parallel TTS calls with push-based ordered results."""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass

from src.clients import TTSClient
from src.logging_config import get_logger
from config import MAX_PARALLEL_TTS

logger = get_logger("orchestrator.tts_pipeline")

# =============================================================================
# Global Semaphore - Shared across ALL queries to limit total TTS concurrency
# =============================================================================
_global_semaphore: Optional[asyncio.Semaphore] = None


def _get_global_semaphore() -> asyncio.Semaphore:
    """Get or create the global semaphore (must be called within event loop)."""
    global _global_semaphore
    if _global_semaphore is None:
        _global_semaphore = asyncio.Semaphore(MAX_PARALLEL_TTS)
        logger.info(f"Global TTS semaphore created: max_parallel={MAX_PARALLEL_TTS}")
    return _global_semaphore


@dataclass
class SentenceResult:
    """Result for a sentence with TTS audio."""
    index: int
    caption: str
    audio: Optional[str]  # base64 encoded
    format: str
    duration_ms: int
    error: Optional[str] = None


class TTSPipeline:
    """
    Manages parallel TTS calls with push-based ordered results.
    
    Features:
    - Limits concurrent calls via GLOBAL semaphore
    - Push-based: Results are pushed to a queue as they complete (in order)
    - Consumer just awaits on the queue - no polling needed
    """
    
    def __init__(
        self,
        voice: str = "osho",
        language: str = "en"
    ):
        self.voice = voice
        self.language = language
        self._client = TTSClient()
        
        # Results queue - consumer reads from here
        self.results_queue: asyncio.Queue[SentenceResult] = asyncio.Queue()
        
        # Internal state for ordering
        self._pending_results: Dict[int, SentenceResult] = {}
        self._next_to_push = 0
        self._tasks: Dict[int, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    async def _call_tts(self, index: int, text: str) -> SentenceResult:
        """Make a single TTS call."""
        semaphore = _get_global_semaphore()
        active = MAX_PARALLEL_TTS - semaphore._value
        text_preview = text[:30] + "..." if len(text) > 30 else text
        
        logger.info(f"🔒 TTS [{index}] QUEUED (active: {active}/{MAX_PARALLEL_TTS}): '{text_preview}'")
        queue_time = time.time()
        
        async with semaphore:
            wait_time = time.time() - queue_time
            active_now = MAX_PARALLEL_TTS - semaphore._value + 1
            logger.info(f"🚀 TTS [{index}] STARTED (waited {wait_time:.2f}s, active: {active_now}/{MAX_PARALLEL_TTS}): '{text_preview}'")
            
            start_time = time.time()
            response = await self._client.generate(
                text=text,
                voice=self.voice,
                language=self.language
            )
            elapsed = time.time() - start_time
            
            if response.error:
                logger.warning(f"❌ TTS [{index}] FAILED ({elapsed:.2f}s): {response.error}")
            else:
                logger.info(f"✅ TTS [{index}] COMPLETE ({elapsed:.2f}s, {response.duration_ms}ms audio)")
            
            return SentenceResult(
                index=index,
                caption=text,
                audio=response.audio,
                format=response.format,
                duration_ms=response.duration_ms,
                error=response.error
            )
    
    async def _on_result_ready(self, index: int, result: SentenceResult):
        """
        Called when a TTS result is ready.
        Pushes results to queue in order (waits for earlier indices first).
        """
        async with self._lock:
            self._pending_results[index] = result
            
            # Push all consecutive ready results to queue
            while self._next_to_push in self._pending_results:
                ready_result = self._pending_results.pop(self._next_to_push)
                await self.results_queue.put(ready_result)
                logger.info(f"📤 TTS [{self._next_to_push}] PUSHED to queue")
                self._next_to_push += 1
    
    def submit(self, index: int, text: str):
        """
        Submit a sentence for TTS processing.
        Results will be pushed to results_queue in order as they complete.
        """
        logger.debug(f"TTS submit [{index}]: '{text[:30]}...'")
        
        async def _task():
            result = await self._call_tts(index, text)
            await self._on_result_ready(index, result)
        
        task = asyncio.create_task(_task())
        self._tasks[index] = task
    
    async def get_result(self) -> SentenceResult:
        """
        Get the next result from the queue (blocks until available).
        Results are guaranteed to be in order.
        """
        return await self.results_queue.get()
    
    def mark_complete(self):
        """
        Signal that no more sentences will be submitted.
        Pushes a None sentinel to the queue after all results are sent.
        """
        async def _push_sentinel():
            # Wait for all tasks to complete
            if self._tasks:
                await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            # Push sentinel to signal end
            await self.results_queue.put(None)
            logger.debug("📤 Pushed sentinel to queue (no more results)")
        
        asyncio.create_task(_push_sentinel())
    
    async def close(self):
        """Close the TTS client."""
        await self._client.close()
