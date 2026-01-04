#!/usr/bin/env python3
"""
Merge all discourse talks into single files per series.

Takes all individual talk .txt files from discourses/{series}/ folders
and merges them into single files in the english/ folder.
"""

import re
from pathlib import Path


def extract_series_name(folder_name: str) -> str:
    """Extract clean series name from folder name (remove number suffix)."""
    # Remove trailing number patterns like -01-11, -01-10, -by-osho-01-09, etc.
    clean = re.sub(r'-\d{2}(-\d{2,3})?$', '', folder_name)
    clean = re.sub(r'-by-osho$', '', clean)
    return clean


def get_talk_number(filename: str) -> int:
    """Extract talk number from filename for sorting."""
    match = re.search(r'-(\d+)\.txt$', filename)
    if match:
        return int(match.group(1))
    return 0


def merge_series(series_dir: Path, output_dir: Path) -> Path:
    """Merge all talks in a series folder into a single file."""
    # Get all txt files sorted by talk number
    txt_files = sorted(
        series_dir.glob("*.txt"),
        key=lambda f: get_talk_number(f.name)
    )
    
    if not txt_files:
        return None
    
    # Extract series name for output filename
    series_name = extract_series_name(series_dir.name)
    output_file = output_dir / f"{series_name}.txt"
    
    # Merge content
    merged_content = []
    
    for i, txt_file in enumerate(txt_files):
        content = txt_file.read_text(encoding='utf-8')
        
        # Skip YAML frontmatter for all but first file
        if i > 0:
            # Remove frontmatter (everything between --- and ---)
            if content.startswith('---'):
                end_marker = content.find('---', 3)
                if end_marker != -1:
                    content = content[end_marker + 3:].strip()
        
        merged_content.append(content)
        merged_content.append("\n\n" + "=" * 60 + "\n\n")  # Separator between talks
    
    # Remove last separator
    if merged_content and merged_content[-1].strip() == "=" * 60:
        merged_content.pop()
    
    # Write merged file
    output_file.write_text("".join(merged_content), encoding='utf-8')
    return output_file


def main():
    # Paths
    base_dir = Path(__file__).parent.parent / "brain-service" / "data" / "documents"
    discourses_dir = base_dir / "discourses"
    english_dir = base_dir / "english"
    
    # Create english folder if it doesn't exist
    english_dir.mkdir(parents=True, exist_ok=True)
    
    if not discourses_dir.exists():
        print(f"Discourses directory not found: {discourses_dir}")
        return
    
    # Get all series folders
    series_folders = [d for d in discourses_dir.iterdir() if d.is_dir()]
    
    if not series_folders:
        print("No series folders found")
        return
    
    print(f"Found {len(series_folders)} series to merge")
    print(f"Output directory: {english_dir}")
    print("=" * 50)
    
    merged_count = 0
    for series_dir in sorted(series_folders):
        txt_count = len(list(series_dir.glob("*.txt")))
        if txt_count == 0:
            continue
        
        print(f"  {series_dir.name} ({txt_count} talks)...", end=" ")
        
        output_file = merge_series(series_dir, english_dir)
        if output_file:
            print(f"✓ -> {output_file.name}")
            merged_count += 1
        else:
            print("✗ no files")
    
    print("=" * 50)
    print(f"Merged {merged_count} series into {english_dir}")


if __name__ == "__main__":
    main()

