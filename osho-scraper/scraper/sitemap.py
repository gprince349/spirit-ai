"""API-based URL discovery for oshoworld.com English discourse library."""

import math
import re
from dataclasses import dataclass
from typing import Optional

import requests

from config import (
    BASE_URL,
    SERIES_SEARCH_API,
    NEXTJS_DATA_API,
    USER_AGENT,
    REQUEST_TIMEOUT,
    ALPHABET,
    ITEMS_PER_PAGE,
)


@dataclass
class DiscourseSeriesInfo:
    """Represents a discovered discourse series."""
    slug: str
    title: str
    talk_count: int
    letter: str


@dataclass
class TalkInfo:
    """Represents an individual talk."""
    slug: str
    title: str
    series_slug: str
    series_name: str
    audio_index: int
    duration: Optional[str] = None


class OshoWorldDiscovery:
    """Discover discourse URLs from oshoworld.com using APIs."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Origin": BASE_URL,
            "Referer": f"{BASE_URL}/audio-series-home-english",
        })
        self._build_id = None
    
    def _get_build_id(self) -> str:
        """
        Fetch the Next.js build ID from the website.
        This is needed to access the _next/data API.
        """
        if self._build_id:
            return self._build_id
        
        try:
            # Fetch the main page and extract build ID
            response = self.session.get(
                f"{BASE_URL}/audio-english-home",
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            # Look for build ID in the page
            match = re.search(r'/_next/data/([^/]+)/', response.text)
            if match:
                self._build_id = match.group(1)
                return self._build_id
            
            # Alternative: look in script tags
            match = re.search(r'"buildId":"([^"]+)"', response.text)
            if match:
                self._build_id = match.group(1)
                return self._build_id
            
            raise ValueError("Could not find Next.js build ID")
        
        except requests.RequestException as e:
            print(f"Error fetching build ID: {e}")
            raise
    
    def discover_series_by_letter(self, letter: str, page: int = 1) -> tuple[list[DiscourseSeriesInfo], int]:
        """
        Discover discourse series starting with a specific letter.
        
        Args:
            letter: Single uppercase letter (A-Z)
            page: Page number (1-indexed)
        
        Returns:
            Tuple of (list of series, total count)
        """
        try:
            response = self.session.post(
                SERIES_SEARCH_API,
                json={
                    "letter": letter.upper(),
                    "page": page,
                    "search": "",
                    "sortBy": "name",
                    "searchQuery": {},
                    "language": "english"
                },
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            
            total_list = data.get("total", [])
            total = total_list[0].get("total", 0) if total_list else 0
            items = data.get("items", [])
            
            series_list = []
            for item in items:
                series = DiscourseSeriesInfo(
                    slug=item.get("slug", ""),
                    title=item.get("title", ""),
                    talk_count=item.get("count", 0),
                    letter=letter.upper()
                )
                series_list.append(series)
            
            return series_list, total
        
        except requests.RequestException as e:
            print(f"Error fetching series for letter {letter}: {e}")
            return [], 0
    
    def discover_all_series(self, letters: list[str] = None) -> list[DiscourseSeriesInfo]:
        """
        Discover all discourse series.
        
        Args:
            letters: Optional list of letters to filter by (A-Z).
                    If None, discovers all series.
        
        Returns:
            List of DiscourseSeriesInfo objects
        """
        all_series = []
        letters_to_search = [l.upper() for l in (letters or ALPHABET)]
        
        print(f"Discovering series for letters: {', '.join(letters_to_search)}")
        
        for letter in letters_to_search:
            # First page to get total
            series, total = self.discover_series_by_letter(letter, page=1)
            all_series.extend(series)
            
            if total == 0:
                continue
            
            # Calculate remaining pages
            total_pages = math.ceil(total / ITEMS_PER_PAGE)
            
            print(f"  {letter}: {total} series ({total_pages} pages)")
            
            # Fetch remaining pages
            for page in range(2, total_pages + 1):
                more_series, _ = self.discover_series_by_letter(letter, page=page)
                all_series.extend(more_series)
        
        print(f"\nTotal series discovered: {len(all_series)}")
        return all_series
    
    def discover_talks_in_series(self, series_slug: str) -> list[TalkInfo]:
        """
        Discover all talks within a discourse series using Next.js data API.
        
        Args:
            series_slug: URL slug of the series
        
        Returns:
            List of TalkInfo objects
        """
        try:
            build_id = self._get_build_id()
            url = f"{NEXTJS_DATA_API}/{build_id}/{series_slug}.json?index={series_slug}"
            
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            page_data = data.get("pageProps", {}).get("data", {}).get("pageData", {})
            list_data = page_data.get("listData", [])
            category_data = page_data.get("categoryData", {})
            
            series_name = category_data.get("title", series_slug)
            
            talks = []
            for item in list_data:
                talk = TalkInfo(
                    slug=item.get("slug", ""),
                    title=item.get("title", ""),
                    series_slug=series_slug,
                    series_name=series_name,
                    audio_index=item.get("audio_index") or 0,
                    duration=item.get("duration"),
                )
                talks.append(talk)
            
            return sorted(talks, key=lambda t: t.audio_index or 0)
        
        except requests.RequestException as e:
            print(f"Error fetching talks for {series_slug}: {e}")
            return []
    
    def discover_all_talks(self, letters: list[str] = None) -> list[TalkInfo]:
        """
        Discover all individual talks from all series.
        
        Args:
            letters: Optional list of letters to filter series by (A-Z)
        
        Returns:
            List of all TalkInfo objects
        """
        all_talks = []
        
        # First get all series
        series_list = self.discover_all_series(letters=letters)
        
        print(f"\nDiscovering talks from {len(series_list)} series...")
        
        for i, series in enumerate(series_list):
            print(f"  [{i+1}/{len(series_list)}] {series.title}...", end=" ", flush=True)
            talks = self.discover_talks_in_series(series.slug)
            all_talks.extend(talks)
            print(f"found {len(talks)} talks")
        
        print(f"\nTotal talks discovered: {len(all_talks)}")
        return all_talks


# For backwards compatibility
SitemapParser = OshoWorldDiscovery
