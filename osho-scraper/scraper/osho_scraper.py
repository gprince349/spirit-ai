"""Main Osho World scraper using APIs with rate limiting and resume support."""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    BASE_URL,
    NEXTJS_DATA_API,
    DEFAULT_DELAY,
    DEFAULT_OUTPUT_DIR,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF,
    STATE_FILE,
    USER_AGENT,
)
from .parsers import OshoWorldParser, ParsedTalk
from .sitemap import OshoWorldDiscovery, TalkInfo, DiscourseSeriesInfo


@dataclass
class ScraperState:
    """Tracks scraping progress for resume capability."""
    scraped_slugs: set = field(default_factory=set)
    failed_slugs: dict = field(default_factory=dict)  # slug -> error message
    total_scraped: int = 0
    
    def save(self, path: Path = STATE_FILE):
        """Save state to JSON file."""
        data = {
            "scraped_slugs": list(self.scraped_slugs),
            "failed_slugs": self.failed_slugs,
            "total_scraped": self.total_scraped,
        }
        path.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, path: Path = STATE_FILE) -> "ScraperState":
        """Load state from JSON file."""
        if not path.exists():
            return cls()
        
        try:
            data = json.loads(path.read_text())
            return cls(
                scraped_slugs=set(data.get("scraped_slugs", [])),
                failed_slugs=data.get("failed_slugs", {}),
                total_scraped=data.get("total_scraped", 0),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()
    
    def clear(self, path: Path = STATE_FILE):
        """Clear saved state."""
        if path.exists():
            path.unlink()
        self.scraped_slugs.clear()
        self.failed_slugs.clear()
        self.total_scraped = 0


class OshoScraper:
    """
    Scrape transcriptions from oshoworld.com English discourse library.
    
    Directory Structure Output:
        documents/
        └── discourses/
            └── {series-slug}/
                ├── {series-slug}-01.txt
                ├── {series-slug}-02.txt
                └── ...
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        delay: float = DEFAULT_DELAY,
        resume: bool = True,
    ):
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.delay = delay
        self.parser = OshoWorldParser()
        self.discovery = OshoWorldDiscovery()
        
        # Load or create state
        self.state = ScraperState.load() if resume else ScraperState()
        
        # Setup session with retry logic
        self.session = self._create_session()
        self._build_id = None
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.5",
        })
        
        return session
    
    def _get_build_id(self) -> str:
        """Get Next.js build ID from discovery or fetch it."""
        if self._build_id:
            return self._build_id
        
        # Try to get from discovery
        self._build_id = self.discovery._get_build_id()
        return self._build_id
    
    def _fetch_talk_json(self, talk_slug: str) -> Optional[dict]:
        """Fetch talk data from Next.js data API."""
        try:
            build_id = self._get_build_id()
            url = f"{NEXTJS_DATA_API}/{build_id}/{talk_slug}.json?index={talk_slug}"
            
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        
        except requests.RequestException as e:
            print(f"  ✗ Error fetching {talk_slug}: {e}")
            return None
    
    def _get_series_dir(self, series_slug: str) -> Path:
        """Get the output directory for a series."""
        from config import clean_slug_for_filename
        clean_name = clean_slug_for_filename(series_slug)
        
        series_dir = self.output_dir / "discourses" / clean_name
        series_dir.mkdir(parents=True, exist_ok=True)
        return series_dir
    
    def scrape_talk(self, talk: TalkInfo) -> Optional[ParsedTalk]:
        """
        Scrape a single talk.
        
        Args:
            talk: TalkInfo object with slug and metadata
        
        Returns:
            ParsedTalk object or None if failed
        """
        # Skip if already scraped
        if talk.slug in self.state.scraped_slugs:
            return None
        
        # Rate limiting
        time.sleep(self.delay)
        
        # Fetch JSON data
        json_data = self._fetch_talk_json(talk.slug)
        if not json_data:
            self.state.failed_slugs[talk.slug] = "Failed to fetch"
            return None
        
        # Parse content
        parsed = self.parser.parse_talk_json(json_data, talk.slug)
        if not parsed:
            self.state.failed_slugs[talk.slug] = "Failed to parse or no content"
            return None
        
        # Update metadata from talk info if missing
        if not parsed.series_slug:
            parsed.series_slug = talk.series_slug
        if not parsed.series_name:
            parsed.series_name = talk.series_name
        if not parsed.talk_number:
            parsed.talk_number = talk.audio_index
        
        return parsed
    
    def save_talk(self, talk: ParsedTalk) -> Path:
        """Save parsed talk to file."""
        series_dir = self._get_series_dir(talk.series_slug)
        filename = talk.get_filename()
        filepath = series_dir / filename
        
        filepath.write_text(talk.to_text_file(), encoding='utf-8')
        return filepath
    
    def scrape_series(self, series: DiscourseSeriesInfo) -> int:
        """
        Scrape all talks in a series.
        
        Args:
            series: DiscourseSeriesInfo object
        
        Returns:
            Number of talks scraped
        """
        print(f"\n📚 Scraping series: {series.title}")
        
        # Discover all talks in this series
        talks = self.discovery.discover_talks_in_series(series.slug)
        
        if not talks:
            print(f"  No talks found in {series.title}")
            return 0
        
        print(f"  Found {len(talks)} talks")
        
        scraped_count = 0
        for i, talk in enumerate(talks):
            print(f"  [{i+1}/{len(talks)}] {talk.title}...", end=" ", flush=True)
            
            if talk.slug in self.state.scraped_slugs:
                print("(already scraped)")
                continue
            
            parsed = self.scrape_talk(talk)
            if parsed:
                filepath = self.save_talk(parsed)
                self.state.scraped_slugs.add(talk.slug)
                self.state.total_scraped += 1
                scraped_count += 1
                print(f"✓ saved")
            else:
                print("✗ failed")
        
        # Save state after each series
        self.state.save()
        
        return scraped_count
    
    def scrape_all(self, letters: list[str] = None, limit: int = None) -> int:
        """
        Scrape all talks from the English discourse library.
        
        Args:
            letters: Optional list of letters to filter series (A-Z)
            limit: Optional limit on number of talks to scrape
        
        Returns:
            Total number of talks scraped
        """
        print("=" * 60)
        print("OSHO World Discourse Scraper (API-based)")
        print("=" * 60)
        
        if letters:
            print(f"Filtering by letters: {', '.join(letters)}")
        
        # Discover all series
        series_list = self.discovery.discover_all_series(letters=letters)
        
        if not series_list:
            print("No series found!")
            return 0
        
        print(f"\nFound {len(series_list)} discourse series")
        
        total_scraped = 0
        for i, series in enumerate(series_list):
            print(f"\n[{i+1}/{len(series_list)}] {series.title} ({series.talk_count} talks)")
            
            count = self.scrape_series(series)
            total_scraped += count
            
            if limit and total_scraped >= limit:
                print(f"\nReached limit of {limit} talks")
                break
        
        print("\n" + "=" * 60)
        print(f"✓ Scraping complete!")
        print(f"  Total talks scraped this session: {total_scraped}")
        print(f"  Total talks in state: {len(self.state.scraped_slugs)}")
        print(f"  Failed: {len(self.state.failed_slugs)}")
        print(f"  Output directory: {self.output_dir}")
        print("=" * 60)
        
        return total_scraped
    
    def scrape_single_series(self, series_slug: str) -> int:
        """
        Scrape a specific series by slug.
        
        Args:
            series_slug: URL slug of the series
        
        Returns:
            Number of talks scraped
        """
        # Create a minimal series info
        series = DiscourseSeriesInfo(
            slug=series_slug,
            title=series_slug.replace('-', ' ').title(),
            talk_count=0,
            letter=series_slug[0].upper() if series_slug else 'A'
        )
        
        return self.scrape_series(series)
    
    def retry_failed(self) -> int:
        """Retry scraping failed slugs."""
        if not self.state.failed_slugs:
            print("No failed slugs to retry")
            return 0
        
        print(f"Retrying {len(self.state.failed_slugs)} failed talks...")
        
        failed_list = list(self.state.failed_slugs.keys())
        scraped_count = 0
        
        for slug in failed_list:
            print(f"  Retrying {slug}...", end=" ", flush=True)
            
            # Create minimal talk info
            talk = TalkInfo(
                slug=slug,
                title=slug,
                series_slug="",
                series_name="",
                audio_index=0,
            )
            
            time.sleep(self.delay)
            
            parsed = self.scrape_talk(talk)
            if parsed:
                self.save_talk(parsed)
                self.state.scraped_slugs.add(slug)
                del self.state.failed_slugs[slug]
                scraped_count += 1
                print("✓")
            else:
                print("✗")
        
        self.state.save()
        print(f"Successfully retried {scraped_count} talks")
        return scraped_count
    
    def get_stats(self) -> dict:
        """Get scraping statistics."""
        return {
            "total_scraped": len(self.state.scraped_slugs),
            "failed": len(self.state.failed_slugs),
            "output_dir": str(self.output_dir),
        }
    
    def clear_state(self):
        """Clear all scraping state."""
        self.state.clear()
        print("State cleared")
