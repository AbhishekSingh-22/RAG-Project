"""
Runner script for document chunking pipeline.

This script processes extracted markdown documents and creates chunks
suitable for RAG (Retrieval-Augmented Generation) applications.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

from text_chunker import (
    DocumentChunker,
    chunk_documents,
    save_chunks_to_json
)


def main():
    """Main function to run the document chunking pipeline."""
    
    # Configuration
    CHUNK_SIZE = 1000  # Characters per chunk
    CHUNK_OVERLAP = 50  # Character overlap between chunks
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "temp_extraction" / "markdown"
    output_dir = project_root / "temp_extraction" / "chunks"
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory not found at {input_dir}")
        print("Please run the PDF extraction pipeline first.")
        sys.exit(1)
    
    # Count input files
    md_files = list(input_dir.glob("*.md"))
    if not md_files:
        print(f"Error: No markdown files found in {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"\n{'='*60}")
    print("DOCUMENT CHUNKING PIPELINE")
    print(f"{'='*60}")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Input Files: {len(md_files)}")
    print(f"Chunk Size: {CHUNK_SIZE} characters")
    print(f"Chunk Overlap: {CHUNK_OVERLAP} characters")
    print(f"{'='*60}\n")
    
    # Process documents
    print("Processing documents...")
    
    docs, stats = chunk_documents(
        input_dir=input_dir,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        file_pattern="*.md"
    )
    
    # Save chunks to JSON for inspection
    chunks_json_path = output_dir / "chunks.json"
    save_chunks_to_json(docs, chunks_json_path, include_content=True)
    
    # Save metadata-only version (smaller file for quick inspection)
    chunks_metadata_path = output_dir / "chunks_metadata.json"
    save_chunks_to_json(docs, chunks_metadata_path, include_content=False)
    
    # Save statistics
    stats_path = output_dir / "chunking_stats.json"
    full_stats = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "input_directory": str(input_dir),
            "output_directory": str(output_dir)
        },
        "statistics": stats,
        "sample_metadata": [doc.metadata for doc in docs[:5]] if docs else []
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(full_stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("CHUNKING COMPLETE")
    print(f"{'='*60}")
    print(f"Documents Processed: {stats['total_documents']}")
    print(f"Total Chunks Created: {stats['total_chunks']}")
    print(f"Average Chunk Size: {stats['avg_chunk_size']:.0f} characters")
    print(f"Min Chunk Size: {stats['min_chunk_size']} characters")
    print(f"Max Chunk Size: {stats['max_chunk_size']} characters")
    print(f"\nOutput Files:")
    print(f"  Chunks (full): {chunks_json_path}")
    print(f"  Chunks (metadata only): {chunks_metadata_path}")
    print(f"  Statistics: {stats_path}")
    
    # Show sample chunks
    if docs:
        print(f"\n{'='*60}")
        print("SAMPLE CHUNKS")
        print(f"{'='*60}")
        
        for i, doc in enumerate(docs[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"File: {doc.metadata['fileName']}")
            print(f"Page: {doc.metadata['Page No']}")
            print(f"Heading: {doc.metadata['Heading']}")
            print(f"SubHeading: {doc.metadata['SubHeading']}")
            print(f"Chunk: {doc.metadata['chunk_index']}/{doc.metadata['total_chunks']}")
            print(f"Content ({len(doc.page_content)} chars):")
            print(f"  {doc.page_content[:150]}...")
    
    print(f"\n{'='*60}")
    print("✓ Chunking pipeline completed successfully!")
    print(f"{'='*60}\n")
    
    return docs, stats


def chunk_single_file(filepath: Path, chunk_size: int = 1000, chunk_overlap: int = 50):
    """
    Utility function to chunk a single file (for testing/debugging).
    
    Args:
        filepath: Path to the markdown file
        chunk_size: Characters per chunk
        chunk_overlap: Character overlap
        
    Returns:
        List of LangChain Documents
    """
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return []
    
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    content = filepath.read_text(encoding='utf-8')
    chunked_docs = chunker.chunk_document(content, filepath)
    
    # Convert to LangChain Documents
    docs = [doc.to_langchain_document() for doc in chunked_docs]
    
    print(f"\nChunked '{filepath.name}' into {len(docs)} chunks")
    for i, doc in enumerate(docs):
        print(f"\n  Chunk {i+1}:")
        print(f"    Metadata: {doc.metadata}")
        print(f"    Preview: {doc.page_content[:100]}...")
    
    return docs


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Single file mode for testing
        file_path = Path(sys.argv[1])
        chunk_single_file(file_path)
    else:
        # Full pipeline mode
        main()

