"""
Text Chunking Module for RAG Pipeline

Implements document chunking using LangChain's RecursiveCharacterTextSplitter
with rich metadata extraction for improved retrieval.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


@dataclass
class ChunkMetadata:
    """Metadata structure for each document chunk."""
    file_name: str
    heading: str
    sub_heading: str
    page_no: int
    chunk_index: int = 0
    total_chunks: int = 0
    source_path: str = ""
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary format."""
        return {
            "fileName": self.file_name,
            "Heading": self.heading,
            "SubHeading": self.sub_heading,
            "Page No": self.page_no,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "source_path": self.source_path
        }


@dataclass
class ChunkedDocument:
    """Represents a chunked document with content and metadata."""
    content: str
    metadata: ChunkMetadata
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format."""
        return Document(
            page_content=self.content,
            metadata=self.metadata.to_dict()
        )


class DocumentChunker:
    """
    Document chunker using LangChain's RecursiveCharacterTextSplitter.
    
    Handles:
    - Recursive text splitting with configurable chunk size and overlap
    - Metadata extraction (headings, subheadings, page numbers)
    - Markdown-aware splitting preserving document structure
    """
    
    # Default configuration
    DEFAULT_CHUNK_SIZE = 1000  # Characters per chunk
    DEFAULT_CHUNK_OVERLAP = 50  # Character overlap between chunks
    
    # Markdown-specific separators (in order of priority)
    MARKDOWN_SEPARATORS = [
        "\n\n---\n\n",  # Horizontal rule with spacing
        "\n---\n",      # Horizontal rule
        "\n\n",         # Paragraph break
        "\n",           # Line break
        " ",            # Word boundary
        ""              # Character level (last resort)
    ]
    
    # Regex patterns for metadata extraction
    PAGE_COMMENT_PATTERN = re.compile(r'<!--\s*Page\s*(\d+)\s*-->')
    H1_PATTERN = re.compile(r'^#\s+(.+)$', re.MULTILINE)
    H2_PATTERN = re.compile(r'^##\s+(.+)$', re.MULTILINE)
    H3_PATTERN = re.compile(r'^###\s+(.+)$', re.MULTILINE)
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        length_function: callable = len,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            length_function: Function to measure chunk length
            separators: Custom separators for splitting (defaults to markdown-aware)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or self.MARKDOWN_SEPARATORS
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=self.separators,
            keep_separator=True,
            is_separator_regex=False
        )
        
        # Statistics tracking
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": float('inf'),
            "max_chunk_size": 0
        }
    
    def extract_page_number(self, content: str, filename: str) -> int:
        """
        Extract page number from content or filename.
        
        Args:
            content: Document content
            filename: Source filename
            
        Returns:
            Page number (1-indexed), defaults to 0 if not found
        """
        # Try extracting from page comment
        match = self.PAGE_COMMENT_PATTERN.search(content)
        if match:
            return int(match.group(1))
        
        # Try extracting from filename (e.g., "document_page_0001.md")
        page_match = re.search(r'page[_-]?(\d+)', filename, re.IGNORECASE)
        if page_match:
            return int(page_match.group(1))
        
        return 0
    
    def extract_headings(self, content: str) -> Tuple[str, str]:
        """
        Extract primary heading and subheading from markdown content.
        
        The heading hierarchy is determined by:
        1. H1 (#) - Primary heading
        2. H2 (##) - Subheading (if H1 not present, H2 becomes primary)
        3. H3 (###) - Used as subheading when H2 is primary
        
        Args:
            content: Markdown content
            
        Returns:
            Tuple of (heading, sub_heading)
        """
        heading = ""
        sub_heading = ""
        
        # Find all headings
        h1_matches = self.H1_PATTERN.findall(content)
        h2_matches = self.H2_PATTERN.findall(content)
        h3_matches = self.H3_PATTERN.findall(content)
        
        # Determine heading hierarchy
        if h1_matches:
            heading = h1_matches[0].strip()
            if h2_matches:
                sub_heading = h2_matches[0].strip()
            elif h3_matches:
                sub_heading = h3_matches[0].strip()
        elif h2_matches:
            heading = h2_matches[0].strip()
            if h3_matches:
                sub_heading = h3_matches[0].strip()
        elif h3_matches:
            heading = h3_matches[0].strip()
        
        return heading, sub_heading
    
    def extract_filename(self, filepath: Path) -> str:
        """
        Extract clean filename from path.
        
        Args:
            filepath: Path to the source file
            
        Returns:
            Clean filename without extension
        """
        return filepath.stem
    
    def extract_source_pdf_name(self, filename: str) -> str:
        """
        Extract the original PDF name from the markdown filename.
        
        Handles patterns like: "PDFName_page_0001"
        
        Args:
            filename: Markdown filename without extension
            
        Returns:
            Original PDF name
        """
        # Remove page suffix if present
        match = re.match(r'^(.+?)_page[_-]?\d+$', filename, re.IGNORECASE)
        if match:
            return match.group(1)
        return filename
    
    def clean_content_for_chunking(self, content: str) -> str:
        """
        Clean content before chunking by removing metadata comments.
        
        Args:
            content: Raw document content
            
        Returns:
            Cleaned content ready for chunking
        """
        # Remove page comment
        content = self.PAGE_COMMENT_PATTERN.sub('', content)
        # Strip leading/trailing whitespace
        content = content.strip()
        return content
    
    def get_heading_context_for_chunk(
        self,
        chunk_content: str,
        full_content: str,
        chunk_start_idx: int
    ) -> Tuple[str, str]:
        """
        Get the heading context for a specific chunk based on its position.
        
        This method finds the most relevant heading and subheading that
        apply to the chunk based on its position in the document.
        
        Args:
            chunk_content: The chunk's text content
            full_content: The full document content
            chunk_start_idx: Approximate start index of chunk in full content
            
        Returns:
            Tuple of (heading, sub_heading) applicable to this chunk
        """
        # Get content before this chunk
        content_before = full_content[:chunk_start_idx] if chunk_start_idx > 0 else ""
        
        # Find headings in content before the chunk
        heading = ""
        sub_heading = ""
        
        # Search backwards for the most recent H1
        h1_matches = list(self.H1_PATTERN.finditer(content_before))
        if h1_matches:
            heading = h1_matches[-1].group(1).strip()
        
        # Search backwards for the most recent H2 or H3
        h2_matches = list(self.H2_PATTERN.finditer(content_before))
        h3_matches = list(self.H3_PATTERN.finditer(content_before))
        
        # Determine subheading based on what's most recent
        if h2_matches:
            sub_heading = h2_matches[-1].group(1).strip()
        elif h3_matches:
            sub_heading = h3_matches[-1].group(1).strip()
        
        # Also check the chunk itself for headings
        chunk_h1 = self.H1_PATTERN.search(chunk_content)
        chunk_h2 = self.H2_PATTERN.search(chunk_content)
        chunk_h3 = self.H3_PATTERN.search(chunk_content)
        
        # If chunk starts with a heading, use it
        if chunk_h1:
            heading = chunk_h1.group(1).strip()
            sub_heading = ""  # Reset subheading for new section
            if chunk_h2:
                sub_heading = chunk_h2.group(1).strip()
        elif chunk_h2 and not heading:
            heading = chunk_h2.group(1).strip()
            if chunk_h3:
                sub_heading = chunk_h3.group(1).strip()
        elif chunk_h3 and not sub_heading:
            sub_heading = chunk_h3.group(1).strip()
        
        return heading, sub_heading
    
    def chunk_document(
        self,
        content: str,
        filepath: Path
    ) -> List[ChunkedDocument]:
        """
        Chunk a single document with metadata.
        
        Args:
            content: Document content
            filepath: Path to the source document
            
        Returns:
            List of ChunkedDocument objects
        """
        # Extract file-level metadata
        filename = self.extract_filename(filepath)
        pdf_name = self.extract_source_pdf_name(filename)
        page_no = self.extract_page_number(content, filename)
        
        # Get document-level headings (fallback)
        doc_heading, doc_sub_heading = self.extract_headings(content)
        
        # Clean content for chunking
        clean_content = self.clean_content_for_chunking(content)
        
        if not clean_content:
            return []
        
        # Split the content
        chunks = self.text_splitter.split_text(clean_content)
        
        if not chunks:
            return []
        
        # Create ChunkedDocument objects with metadata
        chunked_docs = []
        total_chunks = len(chunks)
        
        # Track position for heading context
        current_position = 0
        
        for idx, chunk_text in enumerate(chunks):
            # Get heading context for this specific chunk
            heading, sub_heading = self.get_heading_context_for_chunk(
                chunk_text,
                clean_content,
                current_position
            )
            
            # Fall back to document-level headings if none found
            if not heading:
                heading = doc_heading
            if not sub_heading:
                sub_heading = doc_sub_heading
            
            # Create metadata
            metadata = ChunkMetadata(
                file_name=pdf_name,
                heading=heading,
                sub_heading=sub_heading,
                page_no=page_no,
                chunk_index=idx + 1,
                total_chunks=total_chunks,
                source_path=str(filepath)
            )
            
            chunked_doc = ChunkedDocument(
                content=chunk_text,
                metadata=metadata
            )
            chunked_docs.append(chunked_doc)
            
            # Update position tracking
            current_position += len(chunk_text) - self.chunk_overlap
            
            # Update statistics
            chunk_len = len(chunk_text)
            self.stats["min_chunk_size"] = min(self.stats["min_chunk_size"], chunk_len)
            self.stats["max_chunk_size"] = max(self.stats["max_chunk_size"], chunk_len)
        
        self.stats["total_documents"] += 1
        self.stats["total_chunks"] += total_chunks
        
        return chunked_docs
    
    def chunk_documents_from_directory(
        self,
        input_dir: Path,
        file_pattern: str = "*.md"
    ) -> List[ChunkedDocument]:
        """
        Process all documents in a directory.
        
        Args:
            input_dir: Directory containing documents
            file_pattern: Glob pattern for file selection
            
        Returns:
            List of all chunked documents
        """
        all_chunks = []
        
        # Get all matching files
        files = sorted(input_dir.glob(file_pattern))
        
        for filepath in files:
            try:
                content = filepath.read_text(encoding='utf-8')
                chunks = self.chunk_document(content, filepath)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
        
        # Calculate average chunk size
        if self.stats["total_chunks"] > 0:
            total_chars = sum(len(chunk.content) for chunk in all_chunks)
            self.stats["avg_chunk_size"] = total_chars / self.stats["total_chunks"]
        
        return all_chunks
    
    def get_statistics(self) -> Dict:
        """Get chunking statistics."""
        stats = self.stats.copy()
        if stats["min_chunk_size"] == float('inf'):
            stats["min_chunk_size"] = 0
        return stats
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": float('inf'),
            "max_chunk_size": 0
        }


def chunk_documents(
    input_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    file_pattern: str = "*.md"
) -> Tuple[List[Document], Dict]:
    """
    Main function to chunk documents from a directory.
    
    Args:
        input_dir: Directory containing markdown documents
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        file_pattern: Glob pattern for file selection
        
    Returns:
        Tuple of (list of LangChain Documents, statistics dict)
    """
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunked_docs = chunker.chunk_documents_from_directory(input_dir, file_pattern)
    
    # Convert to LangChain Documents
    langchain_docs = [doc.to_langchain_document() for doc in chunked_docs]
    
    return langchain_docs, chunker.get_statistics()


def save_chunks_to_json(
    chunks: List[Document],
    output_path: Path,
    include_content: bool = True
) -> None:
    """
    Save chunked documents to a JSON file for inspection.
    
    Args:
        chunks: List of LangChain Documents
        output_path: Path to save JSON file
        include_content: Whether to include full content in output
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "chunks": []
    }
    
    for idx, chunk in enumerate(chunks):
        chunk_data = {
            "index": idx,
            "metadata": chunk.metadata
        }
        if include_content:
            chunk_data["content"] = chunk.page_content
            chunk_data["content_length"] = len(chunk.page_content)
        else:
            chunk_data["content_preview"] = chunk.page_content[:200] + "..."
            chunk_data["content_length"] = len(chunk.page_content)
        
        output_data["chunks"].append(chunk_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "temp_extraction" / "markdown"
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
    else:
        # Chunk documents
        docs, stats = chunk_documents(
            input_dir=input_dir,
            chunk_size=1000,
            chunk_overlap=50
        )
        
        print(f"\nChunking Statistics:")
        print(f"  Total Documents: {stats['total_documents']}")
        print(f"  Total Chunks: {stats['total_chunks']}")
        print(f"  Avg Chunk Size: {stats['avg_chunk_size']:.0f} chars")
        print(f"  Min Chunk Size: {stats['min_chunk_size']} chars")
        print(f"  Max Chunk Size: {stats['max_chunk_size']} chars")
        
        # Show sample chunks
        if docs:
            print(f"\n--- Sample Chunk (1 of {len(docs)}) ---")
            print(f"Metadata: {docs[0].metadata}")
            print(f"Content preview: {docs[0].page_content[:200]}...")

