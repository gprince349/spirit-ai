"""Configuration settings for the Osho World website scraper."""

from pathlib import Path


# Base URLs - Using oshoworld.com
BASE_URL = "https://oshoworld.com"

# API Endpoints
SERIES_SEARCH_API = f"{BASE_URL}/api/server/audio/search-series-home"
NEXTJS_DATA_API = f"{BASE_URL}/_next/data"

# Rate limiting
DEFAULT_DELAY = 1.0  # seconds between requests
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # exponential backoff multiplier

# Request settings
REQUEST_TIMEOUT = 30  # seconds
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Output settings
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "brain-service" / "data" / "documents"
STATE_FILE = Path(__file__).parent / "state.json"

# Alphabet for filtering discourse series
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Items per page in series search API
ITEMS_PER_PAGE = 16


def clean_slug_for_filename(slug: str) -> str:
    """Convert URL slug to clean filename."""
    import re
    # Remove any file extension or query params
    clean = slug.split('?')[0].rstrip('/')
    # Replace non-alphanumeric with hyphen
    clean = re.sub(r'[^\w-]', '-', clean)
    return clean.lower()


def clean_html_to_text(html: str) -> str:
    """Convert HTML content to plain text."""
    import re
    
    if not html:
        return ""
    
    # Replace <br> and <br/> with newlines
    text = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
    
    # Replace </p> and </div> with double newlines
    text = re.sub(r'</(?:p|div)>', '\n\n', text, flags=re.IGNORECASE)
    
    # Remove <strong> and </strong> but keep content
    text = re.sub(r'</?strong>', '', text, flags=re.IGNORECASE)
    
    # Remove all other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    import html
    text = html.unescape(text)
    
    # Clean up whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Clean up each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()
