"""
Runner script for embedding pipeline.

Loads chunked documents, generates embeddings using Jina v4,
and stores them in ChromaDB vector store.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

from text_chunker import chunk_documents
from embeddings import LangChainJinaEmbeddings, EmbeddingConfig
from vector_store import (
    ChromaVectorStore,
    VectorStoreConfig,
    load_chunks_from_json,
    create_vector_store
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings and store in ChromaDB"
    )
    parser.add_argument(
        "--chunks-path",
        type=str,
        help="Path to chunks.json file (default: temp_extraction/chunks/chunks.json)"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        help="Directory for persistent ChromaDB storage (default: data/vectorstore)"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="rag_documents",
        help="ChromaDB collection name (default: rag_documents)"
    )
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=None,
        help="Truncate embeddings to dimension (Matryoshka: 128, 256, 512, 1024, 2048)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding generation (default: 8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device for model inference (default: auto)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the collection before adding documents"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the embedding pipeline."""
    args = parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    
    chunks_path = Path(args.chunks_path) if args.chunks_path else \
                  project_root / "temp_extraction" / "chunks" / "chunks.json"
    
    persist_dir = Path(args.persist_dir) if args.persist_dir else \
                  project_root / "data" / "vectorstore"
    
    # Validate chunks file exists
    if not chunks_path.exists():
        print(f"Error: Chunks file not found at {chunks_path}")
        print("Please run the chunking pipeline first: python run_chunking.py")
        sys.exit(1)
    
    # Print configuration
    print(f"\n{'='*60}")
    print("EMBEDDING PIPELINE")
    print(f"{'='*60}")
    print(f"Chunks File: {chunks_path}")
    print(f"Persist Directory: {persist_dir}")
    print(f"Collection Name: {args.collection_name}")
    print(f"Embedding Dimension: {args.truncate_dim or 2048}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Reset Collection: {args.reset}")
    print(f"{'='*60}\n")
    
    # Load chunks
    print("Loading chunks from JSON...")
    documents = load_chunks_from_json(chunks_path)
    print(f"Loaded {len(documents)} document chunks")
    
    # Create embedding model
    print("\nInitializing Jina Embeddings v4 model...")
    embed_config = EmbeddingConfig(
        device=args.device,
        truncate_dim=args.truncate_dim,
        batch_size=args.batch_size
    )
    embeddings = LangChainJinaEmbeddings(embed_config)
    
    # Create vector store
    print("\nInitializing ChromaDB vector store...")
    store_config = VectorStoreConfig(
        collection_name=args.collection_name,
        persist_directory=str(persist_dir),
        embedding_dimension=args.truncate_dim or 2048
    )
    
    vector_store = ChromaVectorStore(config=store_config, embeddings=embeddings)
    vector_store.initialize()
    
    # Reset collection if requested
    if args.reset:
        print("Resetting collection...")
        vector_store.delete_collection()
    
    # Get initial stats
    initial_stats = vector_store.get_collection_stats()
    print(f"Initial collection size: {initial_stats['document_count']}")
    
    # Add documents
    print(f"\nGenerating embeddings and storing {len(documents)} documents...")
    print("This may take a while depending on your hardware...\n")
    
    start_time = datetime.now()
    
    stats = vector_store.add_documents(
        documents=documents,
        batch_size=args.batch_size,
        show_progress=True
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save statistics
    stats_path = persist_dir / "embedding_stats.json"
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    full_stats = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "chunks_file": str(chunks_path),
            "persist_directory": str(persist_dir),
            "collection_name": args.collection_name,
            "embedding_model": "jinaai/jina-embeddings-v4",
            "embedding_dimension": args.truncate_dim or 2048,
            "batch_size": args.batch_size,
            "device": args.device
        },
        "results": stats,
        "performance": {
            "total_duration_seconds": duration,
            "documents_per_second": len(documents) / duration if duration > 0 else 0
        }
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(full_stats, f, indent=2, ensure_ascii=False)
    
    # Final stats
    final_stats = vector_store.get_collection_stats()
    
    # Print summary
    print(f"\n{'='*60}")
    print("EMBEDDING COMPLETE")
    print(f"{'='*60}")
    print(f"Documents Processed: {stats['total_documents']}")
    print(f"Documents Added: {stats['documents_added']}")
    print(f"Collection Size: {final_stats['document_count']}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Speed: {len(documents)/duration:.2f} docs/second")
    
    if stats.get('errors'):
        print(f"\nErrors: {len(stats['errors'])}")
        for error in stats['errors'][:5]:
            print(f"  - {error}")
    
    print(f"\nOutput:")
    print(f"  Vector Store: {persist_dir}")
    print(f"  Statistics: {stats_path}")
    print(f"{'='*60}")
    print("✓ Embedding pipeline completed successfully!")
    print(f"{'='*60}\n")
    
    return full_stats


def test_search(query: str, k: int = 5):
    """
    Utility function to test search on the vector store.
    
    Args:
        query: Search query
        k: Number of results
    """
    project_root = Path(__file__).parent.parent.parent
    persist_dir = project_root / "data" / "vectorstore"
    
    if not persist_dir.exists():
        print(f"Error: Vector store not found at {persist_dir}")
        print("Please run the embedding pipeline first.")
        return
    
    # Create embeddings
    embeddings = LangChainJinaEmbeddings()
    
    # Create vector store
    store = create_vector_store(
        persist_directory=str(persist_dir),
        collection_name="rag_documents",
        embeddings=embeddings
    )
    
    print(f"\nSearching for: '{query}'")
    print(f"{'='*60}")
    
    results = store.similarity_search(query, k=k)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
        print(f"File: {doc.metadata.get('fileName', 'N/A')}")
        print(f"Page: {doc.metadata.get('Page No', 'N/A')}")
        print(f"Heading: {doc.metadata.get('Heading', 'N/A')}")
        print(f"SubHeading: {doc.metadata.get('SubHeading', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--search":
        # Search mode
        query = sys.argv[2] if len(sys.argv) > 2 else "camera setup"
        k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        test_search(query, k)
    else:
        # Main pipeline mode
        main()

