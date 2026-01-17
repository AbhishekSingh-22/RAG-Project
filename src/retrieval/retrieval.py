"""
RAG Retrieval Pipeline

Fetches top-k chunks from the vector store, reranks them, and returns the best results.
"""
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ingestion')))

from langchain_core.documents import Document
from embeddings import LangChainJinaEmbeddings, EmbeddingConfig
from vector_store import create_vector_store, ChromaVectorStore


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval pipeline."""
    persist_dir: Optional[Path] = None
    collection_name: str = "rag_documents"
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    default_k: int = 10
    rerank_top_n: int = 3
    
    def __post_init__(self):
        if self.persist_dir is None:
            project_root = Path(__file__).parent.parent.parent
            self.persist_dir = project_root / "data" / "vectorstore"


class Retriever:
    """
    Retriever class that manages vector store connection and provides search functionality.
    
    Initializes resources once and reuses them for efficient querying.
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        Initialize the retriever.
        
        Args:
            config: Retrieval configuration
        """
        self.config = config or RetrievalConfig()
        self._store: Optional[ChromaVectorStore] = None
        self._embeddings: Optional[LangChainJinaEmbeddings] = None
        self._initialized = False
    
    def _initialize(self) -> None:
        """Lazily initialize embeddings and vector store."""
        if self._initialized:
            return
        
        # Validate vector store exists
        persist_dir = self.config.persist_dir
        if persist_dir is None or not persist_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.config.persist_dir}. "
                "Please run the embedding pipeline first."
            )
        
        # Initialize embeddings model
        embed_config = EmbeddingConfig(model_name=self.config.model_name)
        self._embeddings = LangChainJinaEmbeddings(embed_config)
        
        # Connect to vector store
        self._store = create_vector_store(
            persist_directory=str(self.config.persist_dir),
            collection_name=self.config.collection_name,
            embeddings=self._embeddings
        )
        
        # Verify collection has documents
        assert self._store is not None, "Store initialization failed"
        stats = self._store.get_collection_stats()
        if stats['document_count'] == 0:
            raise ValueError(
                f"Collection '{self.config.collection_name}' is empty. "
                "Please run the embedding pipeline to add documents."
            )
        
        self._initialized = True
    
    def search(
        self, 
        query: str, 
        k: Optional[int] = None,
        filter_metadata: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for relevant chunks given a query.
        
        Args:
            query: Search query string
            k: Number of results to return (default from config)
            filter_metadata: Optional metadata filter for ChromaDB
            
        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        self._initialize()
        
        k = k or self.config.default_k
        
        assert self._store is not None, "Store not initialized"
        results = self._store.similarity_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )
        
        return results
    
    def search_with_rerank(
        self,
        query: str,
        initial_k: Optional[int] = None,
        final_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search with two-stage retrieval: fetch top-k then rerank to top-n.
        
        Args:
            query: Search query string
            initial_k: Number of candidates to fetch (default from config)
            final_k: Number of final results after reranking (default from config)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples after reranking
        """
        initial_k = initial_k or self.config.default_k
        final_k = final_k or self.config.rerank_top_n
        
        # Stage 1: Fetch candidates
        candidates = self.search(query, k=initial_k, filter_metadata=filter_metadata)
        
        if not candidates:
            return []
        
        # Stage 2: Rerank (currently score-based, can be extended to cross-encoder)
        reranked = self._rerank(candidates, query, top_n=final_k)
        
        return reranked
    
    def _rerank(
        self, 
        chunks: List[Tuple[Document, float]], 
        query: str, 
        top_n: int
    ) -> List[Tuple[Document, float]]:
        """
        Rerank chunks using score-based sorting.
        
        Note: This is a simple implementation. For better results, consider:
        - Cross-encoder reranking (e.g., ms-marco-MiniLM-L-6-v2)
        - LLM-based reranking
        - Reciprocal Rank Fusion for hybrid search
        
        Args:
            chunks: List of (Document, score) tuples
            query: Original query (for future cross-encoder use)
            top_n: Number of top results to return
            
        Returns:
            Top-n reranked chunks
        """
        # Sort by score descending (higher = more relevant for cosine similarity)
        sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
        return sorted_chunks[:top_n]
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        self._initialize()
        assert self._store is not None, "Store not initialized"
        return self._store.get_collection_stats()


def format_result(doc: Document, score: float, index: int) -> str:
    """Format a single result for display."""
    meta = doc.metadata
    content_preview = doc.page_content[:500].replace('\n', ' ').strip()
    if len(doc.page_content) > 500:
        content_preview += "..."
    
    return f"""
--- Result {index} (Score: {score:.4f}) ---
File: {meta.get('fileName', 'N/A')}
Page: {meta.get('Page No', 'N/A')}
Heading: {meta.get('Heading', 'N/A')}
SubHeading: {meta.get('SubHeading', 'N/A')}
Content: {content_preview}
"""


def print_results(results: List[Tuple[Document, float]]) -> None:
    """Print results in a readable format."""
    if not results:
        print("No results found.")
        return
    
    for i, (doc, score) in enumerate(results, 1):
        print(format_result(doc, score, i))


# Convenience functions for backward compatibility
def fetch_top_k_chunks(query: str, k: int = 10) -> List[Tuple[Document, float]]:
    """
    Fetch top-k chunks from the vector store for the query.
    
    Args:
        query: Search query
        k: Number of results
        
    Returns:
        List of (Document, score) tuples
    """
    retriever = Retriever()
    return retriever.search(query, k=k)


def rerank_chunks(
    chunks: List[Tuple[Document, float]], 
    query: str, 
    top_n: int = 3
) -> List[Tuple[Document, float]]:
    """
    Rerank chunks and return top-n results.
    
    Args:
        chunks: Candidate chunks
        query: Original query
        top_n: Number of results to return
        
    Returns:
        Top-n reranked chunks
    """
    sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
    return sorted_chunks[:top_n]


def main():
    """CLI interface for retrieval."""
    if len(sys.argv) < 2:
        print("Usage: python retrieval.py <query> [k]")
        print("  query: Search query string")
        print("  k: Number of results (default: 10)")
        sys.exit(1)
    
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    try:
        retriever = Retriever()
        
        # Show collection stats
        stats = retriever.get_stats()
        print(f"Collection: {stats['collection_name']} ({stats['document_count']} documents)")
        
        # Fetch and display results
        print(f"\nFetching top {k} chunks...")
        results = retriever.search(query, k=k)
        print_results(results)
        
        # Show reranked results
        if len(results) > 3:
            print(f"\n{'='*60}")
            print("Top 3 after reranking:")
            print(f"{'='*60}")
            reranked = retriever.search_with_rerank(query, initial_k=k, final_k=3)
            print_results(reranked)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
