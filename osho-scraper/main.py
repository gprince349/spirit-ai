#!/usr/bin/env python3
"""
Osho World Discourse Scraper CLI

Scrapes transcription text from oshoworld.com English discourse library using APIs.

Usage:
    python main.py discover                    # List all discourse series
    python main.py discover --letter A         # List series starting with A
    python main.py talks SERIES_SLUG           # List talks in a series
    python main.py scrape --all                # Scrape all discourses
    python main.py scrape --letter A           # Scrape series starting with A
    python main.py scrape --letter A,B,C       # Scrape multiple letters
    python main.py scrape --series SLUG        # Scrape specific series
    python main.py scrape --limit 10           # Scrape only 10 talks
    python main.py retry                       # Retry failed URLs
    python main.py stats                       # Show scraping statistics
    python main.py clear                       # Clear scraping state

Output Structure:
    brain-service/data/documents/discourses/
    └── {series-slug}/
        ├── {series-slug}-01.txt
        ├── {series-slug}-02.txt
        └── ...
"""

import argparse
import sys
from pathlib import Path

# Add the scraper module to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DEFAULT_DELAY, DEFAULT_OUTPUT_DIR
from scraper import OshoScraper, OshoWorldDiscovery


def cmd_discover(args):
    """Discover and list all discourse series."""
    discovery = OshoWorldDiscovery()
    
    # Parse letters if provided
    letters = None
    if args.letter:
        letters = [l.strip().upper() for l in args.letter.split(',')]
    
    series_list = discovery.discover_all_series(letters=letters)
    
    print("\n" + "=" * 60)
    print("OSHO World English Discourse Library")
    print("=" * 60)
    
    if letters:
        print(f"Filtered by letters: {', '.join(letters)}")
    
    print(f"\nFound {len(series_list)} discourse series:\n")
    
    current_letter = None
    total_talks = 0
    for series in series_list:
        if series.letter != current_letter:
            current_letter = series.letter
            print(f"\n--- {current_letter} ---")
        print(f"  {series.title} ({series.talk_count} talks)")
        print(f"    slug: {series.slug}")
        total_talks += series.talk_count
    
    print(f"\n\nTotal: {len(series_list)} series, {total_talks} talks")


def cmd_discover_talks(args):
    """Discover talks in a specific series."""
    discovery = OshoWorldDiscovery()
    
    series_slug = args.series
    talks = discovery.discover_talks_in_series(series_slug)
    
    if not talks:
        print(f"No talks found for series: {series_slug}")
        return
    
    print(f"\nTalks in '{series_slug}':")
    print("-" * 50)
    
    for talk in talks:
        duration = f" ({talk.duration})" if talk.duration else ""
        print(f"  {talk.audio_index:02d}. {talk.title}{duration}")
        print(f"      slug: {talk.slug}")
    
    print(f"\nTotal: {len(talks)} talks")


def cmd_scrape(args):
    """Scrape discourse transcriptions."""
    # Parse output directory
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    
    # Create scraper
    scraper = OshoScraper(
        output_dir=output_dir,
        delay=args.delay,
        resume=not args.fresh,
    )
    
    # Parse letters if provided
    letters = None
    if args.letter:
        letters = [l.strip().upper() for l in args.letter.split(',')]
    
    # Parse limit
    limit = args.limit if args.limit and args.limit > 0 else None
    
    # Specific series scraping
    if args.series:
        count = scraper.scrape_single_series(args.series)
        print(f"\nScraped {count} talks from '{args.series}'")
        return
    
    # All or filtered scraping
    if args.all or letters:
        total = scraper.scrape_all(letters=letters, limit=limit)
        print(f"\nTotal scraped: {total} talks")
    else:
        print("Please specify --all, --letter, or --series")
        print("Run 'python main.py scrape --help' for options")


def cmd_retry(args):
    """Retry failed scraping attempts."""
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    
    scraper = OshoScraper(
        output_dir=output_dir,
        delay=args.delay,
        resume=True,
    )
    
    count = scraper.retry_failed()
    print(f"\nRetried {count} talks successfully")


def cmd_stats(args):
    """Show scraping statistics."""
    scraper = OshoScraper(resume=True)
    stats = scraper.get_stats()
    
    print("\n" + "=" * 50)
    print("Scraping Statistics")
    print("=" * 50)
    print(f"Total scraped:  {stats['total_scraped']}")
    print(f"Failed:         {stats['failed']}")
    print(f"Output dir:     {stats['output_dir']}")
    print("=" * 50)
    
    # Show directory structure if it exists
    output_path = Path(stats['output_dir']) / "discourses"
    if output_path.exists():
        print("\nSeries directories:")
        for series_dir in sorted(output_path.iterdir()):
            if series_dir.is_dir():
                file_count = len(list(series_dir.glob("*.txt")))
                print(f"  {series_dir.name}/: {file_count} talks")


def cmd_clear(args):
    """Clear scraping state."""
    scraper = OshoScraper(resume=True)
    
    if not args.yes:
        confirm = input("Are you sure you want to clear all state? [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled")
            return
    
    scraper.clear_state()
    print("State cleared successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape OSHO discourse transcriptions from oshoworld.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py discover                    # List all series
    python main.py discover --letter A         # List series starting with A
    python main.py talks a-bird-on-the-wing-01-11
    python main.py scrape --all                # Scrape everything
    python main.py scrape --letter A           # Scrape A series only
    python main.py scrape --series a-bird-on-the-wing-01-11
    python main.py stats                       # Show statistics
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover discourse series from oshoworld.com"
    )
    discover_parser.add_argument(
        "--letter", "-l",
        type=str,
        help="Filter by letter(s), comma-separated (e.g., A or A,B,C)"
    )
    discover_parser.set_defaults(func=cmd_discover)
    
    # Discover talks in series
    talks_parser = subparsers.add_parser(
        "talks",
        help="List talks in a specific series"
    )
    talks_parser.add_argument(
        "series",
        type=str,
        help="Series slug (e.g., a-bird-on-the-wing-01-11)"
    )
    talks_parser.set_defaults(func=cmd_discover_talks)
    
    # Scrape command
    scrape_parser = subparsers.add_parser(
        "scrape",
        help="Scrape discourse transcriptions"
    )
    scrape_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Scrape all discourses"
    )
    scrape_parser.add_argument(
        "--letter", "-l",
        type=str,
        help="Filter by letter(s), comma-separated (e.g., A or A,B,C)"
    )
    scrape_parser.add_argument(
        "--series", "-s",
        type=str,
        help="Scrape specific series by slug"
    )
    scrape_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of talks to scrape"
    )
    scrape_parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})"
    )
    scrape_parser.add_argument(
        "--output", "-o",
        type=str,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    scrape_parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring previous state"
    )
    scrape_parser.set_defaults(func=cmd_scrape)
    
    # Retry command
    retry_parser = subparsers.add_parser(
        "retry",
        help="Retry failed scraping attempts"
    )
    retry_parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests (default: {DEFAULT_DELAY})"
    )
    retry_parser.add_argument(
        "--output", "-o",
        type=str,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    retry_parser.set_defaults(func=cmd_retry)
    
    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show scraping statistics"
    )
    stats_parser.set_defaults(func=cmd_stats)
    
    # Clear command
    clear_parser = subparsers.add_parser(
        "clear",
        help="Clear scraping state"
    )
    clear_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    clear_parser.set_defaults(func=cmd_clear)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
