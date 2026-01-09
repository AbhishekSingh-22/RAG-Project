"""
Image Description Chunking Module for RAG Pipeline

Chunks markdown image descriptions using LangChain's RecursiveCharacterTextSplitter,
with metadata extraction for improved retrieval.
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
class ImageChunkMetadata:
    """Metadata structure for each image description chunk."""
    file_name: str
    heading: str
    sub_heading: str
    page_no: int
    chunk_index: int = 0
    total_chunks: int = 0
    source_path: str = ""
    
    def to_dict(self) -> Dict:
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
class ChunkedImageDescription:
    """Represents a chunked image description with content and metadata."""
    content: str
    metadata: ImageChunkMetadata
    
    def to_langchain_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata=self.metadata.to_dict()
        )

class ImageDescriptionChunker:
    """
    Chunks markdown image descriptions using LangChain's RecursiveCharacterTextSplitter.
    Extracts metadata from markdown header and file name.
    """
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 50
    MARKDOWN_SEPARATORS = [
        "\n\n---\n\n",
        "\n---\n",
        "\n\n",
        "\n",
        " ",
        ""
    ]
    EXTRACTION_DATE_PATTERN = re.compile(r'\*\*Extraction Date\*\*: ([0-9.]+)')
    IMAGE_NAME_PATTERN = re.compile(r'# Description for: (.+)')
    HEADING_PATTERN = re.compile(r'##\s+Summary\s*\n(.+?)(?:\n##|\Z)', re.DOTALL)
    SUBHEADING_PATTERN = re.compile(r'##\s+Image Type\s*\n(.+?)(?:\n##|\Z)', re.DOTALL)
    PAGE_NO_PATTERN = re.compile(r'Page\s*No\.*\s*(\d+)', re.IGNORECASE)
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = self.MARKDOWN_SEPARATORS
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            keep_separator=True,
            is_separator_regex=False
        )
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": float('inf'),
            "max_chunk_size": 0
        }
    
    def extract_metadata(self, content: str, filepath: Path) -> Dict:
        """
        Extracts required metadata fields from markdown content and filename.
        """
        file_name = filepath.stem.split('_page')[0]
        heading = ""
        sub_heading = ""
        page_no = 0
        # Try to extract heading and subheading from markdown sections
        heading_match = self.HEADING_PATTERN.search(content)
        if heading_match:
            heading = heading_match.group(1).strip().split('\n')[0]
        subheading_match = self.SUBHEADING_PATTERN.search(content)
        if subheading_match:
            sub_heading = subheading_match.group(1).strip().split('\n')[0]
        # Try to extract page number from filename
        page_match = re.search(r'page[_-]?(\d+)', filepath.stem, re.IGNORECASE)
        if page_match:
            page_no = int(page_match.group(1))
        return {
            "file_name": file_name,
            "heading": heading,
            "sub_heading": sub_heading,
            "page_no": page_no
        }
    
    def extract_filename(self, filepath: Path) -> str:
        return filepath.stem
    
    def clean_content_for_chunking(self, content: str) -> str:
        return content.strip()
    
    def chunk_image_description(self, content: str, filepath: Path) -> List[ChunkedImageDescription]:
        meta = self.extract_metadata(content, filepath)
        clean_content = self.clean_content_for_chunking(content)
        if not clean_content:
            return []
        chunks = self.text_splitter.split_text(clean_content)
        if not chunks:
            return []
        chunked_docs = []
        total_chunks = len(chunks)
        for idx, chunk_text in enumerate(chunks):
            metadata = ImageChunkMetadata(
                file_name=meta["file_name"],
                heading=meta["heading"],
                sub_heading=meta["sub_heading"],
                page_no=meta["page_no"],
                chunk_index=idx + 1,
                total_chunks=total_chunks,
                source_path=str(filepath)
            )
            chunked_doc = ChunkedImageDescription(
                content=chunk_text,
                metadata=metadata
            )
            chunked_docs.append(chunked_doc)
            chunk_len = len(chunk_text)
            self.stats["min_chunk_size"] = min(self.stats["min_chunk_size"], chunk_len)
            self.stats["max_chunk_size"] = max(self.stats["max_chunk_size"], chunk_len)
        self.stats["total_documents"] += 1
        self.stats["total_chunks"] += total_chunks
        return chunked_docs
    
    def chunk_descriptions_from_directory(self, input_dir: Path, file_pattern: str = "*_description.md") -> List[ChunkedImageDescription]:
        all_chunks = []
        files = sorted(input_dir.glob(file_pattern))
        for filepath in files:
            try:
                content = filepath.read_text(encoding='utf-8')
                chunks = self.chunk_image_description(content, filepath)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
        if self.stats["total_chunks"] > 0:
            total_chars = sum(len(chunk.content) for chunk in all_chunks)
            self.stats["avg_chunk_size"] = total_chars / self.stats["total_chunks"]
        return all_chunks
    
    def get_statistics(self) -> Dict:
        stats = self.stats.copy()
        if stats["min_chunk_size"] == float('inf'):
            stats["min_chunk_size"] = 0
        return stats
    
    def reset_statistics(self):
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": float('inf'),
            "max_chunk_size": 0
        }

def chunk_image_descriptions(
    input_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 50,
    file_pattern: str = "*_description.md"
) -> Tuple[List[Document], Dict]:
    chunker = ImageDescriptionChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = chunker.chunk_descriptions_from_directory(input_dir, file_pattern)
    langchain_docs = [doc.to_langchain_document() for doc in chunked_docs]
    return langchain_docs, chunker.get_statistics()

def save_image_chunks_to_json(
    chunks: List[Document],
    output_dir: Path,
    include_content: bool = True
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        chunk_data = {
            "metadata": chunk.metadata,
            "content": chunk.page_content if include_content else chunk.page_content[:200] + "..."
        }
        # Compose filename: fileName_pageNo_chunkIndex.json
        meta = chunk.metadata
        file_name = meta.get("fileName") or meta.get("file_name")
        page_no = meta.get("Page No") or meta.get("page_no")
        chunk_index = meta.get("chunk_index")
        out_name = f"{file_name}_page_{page_no:04d}_chunk_{chunk_index}.json"
        out_path = output_dir / out_name
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "temp_extraction" / "image_descriptions"
    review_dir = project_root / "temp_extraction" / "image_chunks_review"
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
    else:
        docs, stats = chunk_image_descriptions(
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
        # Save each chunk as a separate JSON file for review
        save_image_chunks_to_json(docs, review_dir, include_content=True)
        print(f"\nChunks saved for review in: {review_dir}")
