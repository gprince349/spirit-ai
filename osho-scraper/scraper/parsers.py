"""Content parsers for extracting transcription text from oshoworld.com API responses."""

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config import clean_html_to_text, clean_slug_for_filename


@dataclass
class ParsedTalk:
    """Represents parsed transcription from a talk."""
    title: str
    series_name: str
    series_slug: str
    talk_number: int
    content: str
    url: str
    scraped_at: str
    duration: Optional[str] = None
    author: Optional[str] = None
    language: str = "English"
    
    def to_text_file(self) -> str:
        """Convert to text file format with YAML frontmatter."""
        frontmatter = [
            "---",
            f"title: {self._escape_yaml(self.title)}",
            f"series: {self._escape_yaml(self.series_name)}",
            f"talk_number: {self.talk_number}",
            f"url: {self.url}",
            f"language: {self.language}",
        ]
        
        if self.duration:
            frontmatter.append(f"duration: {self.duration}")
        
        if self.author:
            frontmatter.append(f"author: {self._escape_yaml(self.author)}")
        
        frontmatter.extend([
            f"scraped_at: {self.scraped_at}",
            "---",
            "",
        ])
        
        return "\n".join(frontmatter) + self.content
    
    def _escape_yaml(self, text: str) -> str:
        """Escape special YAML characters in string values."""
        if not text:
            return '""'
        # If contains special chars, wrap in quotes and escape internal quotes
        if any(c in text for c in [':', '#', '"', "'", '\n', '[', ']', '{', '}']):
            escaped = text.replace('"', '\\"')
            return f'"{escaped}"'
        return text
    
    def get_filename(self) -> str:
        """Generate a clean filename for this talk."""
        clean_series = clean_slug_for_filename(self.series_slug)
        return f"{clean_series}-{self.talk_number:02d}.txt"


class OshoWorldParser:
    """Parse transcription content from oshoworld.com API responses."""
    
    def parse_talk_json(self, json_data: dict, talk_slug: str) -> Optional[ParsedTalk]:
        """
        Extract transcription from Next.js data API response.
        
        Args:
            json_data: JSON response from /_next/data/{build_id}/{talk_slug}.json
            talk_slug: URL slug of the talk
        
        Returns:
            ParsedTalk object or None if parsing failed
        """
        try:
            page_props = json_data.get("pageProps", {})
            data = page_props.get("data", {})
            page_data = data.get("pageData", {})
            audio_data = page_data.get("audioData", {})
            
            if not audio_data:
                return None
            
            # Extract transcription from description field
            description_html = audio_data.get("description", "")
            content = clean_html_to_text(description_html)
            
            if not content or len(content) < 100:
                return None
            
            # Extract metadata
            title = audio_data.get("title", talk_slug)
            series_name = audio_data.get("series_name", "")
            series_slug = audio_data.get("series_slug", "")
            duration = audio_data.get("duration")
            author = audio_data.get("author")
            
            # Extract talk number from slug or audio_index
            talk_number = self._extract_talk_number(talk_slug, audio_data)
            
            return ParsedTalk(
                title=title,
                series_name=series_name,
                series_slug=series_slug,
                talk_number=talk_number,
                content=content,
                url=f"https://oshoworld.com/{talk_slug}",
                scraped_at=datetime.now(timezone.utc).isoformat(),
                duration=duration,
                author=author,
            )
        
        except Exception as e:
            print(f"Error parsing talk {talk_slug}: {e}")
            return None
    
    def _extract_talk_number(self, slug: str, audio_data: dict) -> int:
        """Extract talk number from slug or audio data."""
        # Try audio_index first
        if "audio_index" in audio_data:
            return int(audio_data["audio_index"])
        
        # Try to extract from slug (e.g., "a-bird-on-the-wing-01" -> 1)
        match = re.search(r'-(\d+)$', slug)
        if match:
            return int(match.group(1))
        
        return 0


# For backwards compatibility
ContentParser = OshoWorldParser
