"""
RAG Retrieval Pipeline

Fetches top-k chunks from the vector store, reranks them, and returns the best results.
"""
import sys
import os
from pathlib import Path
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ingestion')))

from embeddings import LangChainJinaEmbeddings, EmbeddingConfig
from vector_store import (create_vector_store)

def encode_query(query: str, embeddings) -> List[float]:
    """Encode the query using the embedding model."""
    return embeddings.embed_query(query)

def fetch_top_k_chunks(query: str, k: int = 10) -> List[Tuple]:
    """
    Fetch top-k chunks from the vector store for the query.
    Returns a list of (Document, score) tuples.
    """
    project_root = Path(__file__).parent.parent.parent
    persist_dir = project_root / "data" / "vectorstore"
    embed_config = EmbeddingConfig(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = LangChainJinaEmbeddings(embed_config)
    store = create_vector_store(
        persist_directory=str(persist_dir),
        collection_name="rag_documents",
        embeddings=embeddings
    )
    results = store.similarity_search(query, k=k)
    return results

def rerank_chunks(chunks: List[Tuple], query: str, top_n: int = 3) -> List[Tuple]:
    """
    Rerank the top-k chunks using a simple score-based reranking (can be replaced with LLM reranker).
    Returns the top_n reranked chunks.
    """
    # For now, just sort by score descending and take top_n
    reranked = sorted(chunks, key=lambda x: x[1], reverse=True)[:top_n]
    return reranked

def print_results(results: List[Tuple]):
    """
    Print the results in a readable format.
    """
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        print(f"\n--- Result {i} (Score: {score:.4f}) ---")
        print(f"File: {meta.get('fileName', 'N/A')}")
        print(f"Page: {meta.get('Page No', 'N/A')}")
        print(f"Heading: {meta.get('Heading', 'N/A')}")
        print(f"SubHeading: {meta.get('SubHeading', 'N/A')}")
        print(f"Content: {doc.page_content[:300]}...")

def main():
    if len(sys.argv) < 2:
        print("Usage: python retrieval.py <query>")
        sys.exit(1)
    query = sys.argv[1]
    print(f"\nQuery: {query}")
    print("Fetching top 10 chunks...")
    top_chunks = fetch_top_k_chunks(query, k=10)
    print_results(top_chunks)
    print("\nReranking and fetching top 3 chunks...")
    top_reranked = rerank_chunks(top_chunks, query, top_n=3)
    print_results(top_reranked)

if __name__ == "__main__":
    main()
