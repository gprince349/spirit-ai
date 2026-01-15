"""Sentence Buffer - Accumulates tokens into sentences for TTS."""

from typing import Optional
from config import MIN_SENTENCE_WORDS, MAX_SENTENCE_WORDS
from src.logging_config import get_logger

logger = get_logger("orchestrator.sentence_buffer")


class SentenceBuffer:
    """
    Buffer that accumulates tokens into complete sentences.
    
    Rules:
    - Minimum words: 15 (don't split too early)
    - Maximum words: 30 (force split if too long)
    - Valid endings: . ! ? । ...
    """
    
    # Valid sentence endings
    ENDINGS_EN = {'.', '!', '?'}
    ENDINGS_HI = {'।', '॥'}
    PAUSE_MARKERS = {'...', '—', '–'}
    
    def __init__(
        self,
        min_words: int = MIN_SENTENCE_WORDS,
        max_words: int = MAX_SENTENCE_WORDS
    ):
        self.buffer = ""
        self.min_words = min_words
        self.max_words = max_words
        self.all_endings = self.ENDINGS_EN | self.ENDINGS_HI
    
    def _count_words(self) -> int:
        """Count words in buffer."""
        return len(self.buffer.split())
    
    def _ends_with_punctuation(self) -> bool:
        """Check if buffer ends with valid sentence punctuation."""
        text = self.buffer.rstrip()
        if not text:
            return False
        
        # Check for pause markers like "..."
        for marker in self.PAUSE_MARKERS:
            if text.endswith(marker):
                return True
        
        # Check for single character endings
        return text[-1] in self.all_endings
    
    def _find_split_point(self) -> int:
        """Find best point to split when max words exceeded."""
        text = self.buffer
        
        # Look for last sentence ending
        for i in range(len(text) - 1, -1, -1):
            if text[i] in self.all_endings:
                return i + 1
        
        # Fallback: split at last space
        last_space = text.rfind(' ')
        if last_space > 0:
            return last_space
        
        # No good split point, return all
        return len(text)
    
    def add(self, token: str) -> Optional[str]:
        """
        Add a token to the buffer.
        
        Returns:
            Complete sentence if ready, None otherwise.
        """
        self.buffer += token
        word_count = self._count_words()
        
        # Check if we have enough words and proper ending
        if word_count >= self.min_words and self._ends_with_punctuation():
            sentence = self.buffer.strip()
            self.buffer = ""
            logger.debug(f"Sentence complete (punctuation): {word_count} words")
            return sentence
        
        # Force split if too long
        if word_count >= self.max_words:
            split_point = self._find_split_point()
            sentence = self.buffer[:split_point].strip()
            self.buffer = self.buffer[split_point:].lstrip()
            logger.debug(f"Sentence force-split (max words): {word_count} words")
            return sentence
        
        return None
    
    def flush(self) -> Optional[str]:
        """
        Flush remaining buffer content.
        
        Call this when the token stream ends.
        
        Returns:
            Remaining text if any, None otherwise.
        """
        if self.buffer.strip():
            sentence = self.buffer.strip()
            word_count = len(sentence.split())
            self.buffer = ""
            logger.debug(f"Sentence flushed (stream end): {word_count} words")
            return sentence
        return None
    
    def reset(self):
        """Clear the buffer."""
        self.buffer = ""

