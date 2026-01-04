"""Document Ingestion Script - Process PDFs and text files to build vector index."""

import argparse
import re
import time
from pathlib import Path

from pypdf import PdfReader

from config import settings
from src.embeddings import embedder
from src.vector_store import vector_store


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None
) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Words per chunk (default from settings)
        overlap: Overlap between chunks (default from settings)
        
    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"  ⚠ Error reading {pdf_path}: {e}")
        return ""


def extract_txt_text(txt_path: str) -> tuple[str, dict]:
    """
    Extract text and metadata from a text file with optional YAML frontmatter.
    
    Args:
        txt_path: Path to text file
        
    Returns:
        Tuple of (text content, metadata dict)
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {}
        text = content
        
        # Check for YAML frontmatter (between --- markers)
        frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', content, re.DOTALL)
        
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            text = frontmatter_match.group(2).strip()
            
            # Parse simple YAML key-value pairs
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    metadata[key] = value
        
        return text.strip(), metadata
        
    except Exception as e:
        print(f"  ⚠ Error reading {txt_path}: {e}")
        return "", {}


def find_documents(docs_path: Path) -> tuple[list[Path], list[Path]]:
    """
    Find all PDF and text files in directory (including subdirectories).
    
    Args:
        docs_path: Base directory to search
        
    Returns:
        Tuple of (pdf_files, txt_files)
    """
    pdf_files = list(docs_path.rglob("*.pdf"))
    txt_files = list(docs_path.rglob("*.txt"))
    return pdf_files, txt_files


def ingest_documents(
    docs_dir: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    batch_size: int = 100
):
    """
    Ingest all PDF and text documents from directory.
    
    Args:
        docs_dir: Directory containing documents (PDFs and .txt files)
        chunk_size: Words per chunk
        chunk_overlap: Overlap between chunks
        batch_size: Embedding batch size
    """
    docs_dir = docs_dir or settings.DOCUMENTS_DIR
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        print(f"❌ Documents directory not found: {docs_dir}")
        print(f"   Please copy your documents to: {docs_path.absolute()}")
        return
    
    # Find all documents (PDFs and text files)
    pdf_files, txt_files = find_documents(docs_path)
    total_files = len(pdf_files) + len(txt_files)
    
    if total_files == 0:
        print(f"❌ No PDF or text files found in: {docs_dir}")
        return
    
    print("=" * 60)
    print(f"Ingesting documents from {docs_dir}")
    print(f"  PDF files: {len(pdf_files)}")
    print(f"  Text files: {len(txt_files)}")
    print("=" * 60)
    
    all_chunks = []
    all_metadata = []
    
    start_time = time.time()
    file_index = 0
    
    # Process PDF files
    for pdf_file in pdf_files:
        file_index += 1
        print(f"[{file_index}/{total_files}] Processing: {pdf_file.name}")
        
        # Extract text
        text = extract_pdf_text(str(pdf_file))
        
        if not text:
            print(f"  ⚠ No text extracted, skipping")
            continue
        
        # Chunk text
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        print(f"  ✓ Extracted {len(chunks)} chunks")
        
        # Add chunks and metadata
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": pdf_file.name,
                "file_type": "pdf",
                "chunk_index": j,
                "total_chunks": len(chunks)
            })
    
    # Process text files
    for txt_file in txt_files:
        file_index += 1
        # Show relative path for nested files
        rel_path = txt_file.relative_to(docs_path)
        print(f"[{file_index}/{total_files}] Processing: {rel_path}")
        
        # Extract text and metadata
        text, file_metadata = extract_txt_text(str(txt_file))
        
        if not text:
            print(f"  ⚠ No text extracted, skipping")
            continue
        
        # Chunk text
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        print(f"  ✓ Extracted {len(chunks)} chunks")
        
        # Build metadata from frontmatter
        content_type = file_metadata.get("type", "article")
        source = file_metadata.get("title", txt_file.stem)
        
        # Add chunks and metadata
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": source,
                "file_type": "txt",
                "content_type": content_type,
                "url": file_metadata.get("url", ""),
                "chunk_index": j,
                "total_chunks": len(chunks)
            })
    
    if not all_chunks:
        print("❌ No chunks extracted from any documents")
        return
    
    print("-" * 60)
    print(f"Total chunks: {len(all_chunks)}")
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    embedding_start = time.time()
    
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.append(batch_embeddings)
        print(f"  Embedded {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} chunks")
    
    import numpy as np
    embeddings = np.vstack(all_embeddings)
    
    embedding_time = time.time() - embedding_start
    print(f"  ✓ Embeddings generated in {embedding_time:.2f}s")
    
    # Build index
    print("Building FAISS index...")
    vector_store.add(all_chunks, embeddings, all_metadata)
    
    # Save index
    print("Saving index...")
    vector_store.save()
    
    total_time = time.time() - start_time
    
    print("=" * 60)
    print("✓ Ingestion complete!")
    print(f"  Documents: {total_files} ({len(pdf_files)} PDFs, {len(txt_files)} text files)")
    print(f"  Chunks: {len(all_chunks)}")
    print(f"  Vectors: {vector_store.total_vectors}")
    print(f"  Dimension: {embedder.dimension}")
    print(f"  Time: {total_time:.2f}s")
    print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Brain Service vector store"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=settings.DOCUMENTS_DIR,
        help=f"Directory containing documents - PDFs and .txt files (default: {settings.DOCUMENTS_DIR})"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.CHUNK_SIZE,
        help=f"Words per chunk (default: {settings.CHUNK_SIZE})"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {settings.CHUNK_OVERLAP})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Embedding batch size (default: 100)"
    )
    
    args = parser.parse_args()
    
    ingest_documents(
        docs_dir=args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

