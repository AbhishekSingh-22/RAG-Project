"""
Vector Store Module for RAG Pipeline

Manages ChromaDB vector store for storing and retrieving document embeddings.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from embeddings import LangChainJinaEmbeddings, EmbeddingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store."""
    collection_name: str = "rag_documents"
    persist_directory: Optional[str] = None
    embedding_dimension: int = 2048
    distance_metric: str = "cosine"  # "cosine", "l2", "ip"


class ChromaVectorStore:
    """
    ChromaDB vector store manager for document embeddings.
    
    Features:
    - Persistent storage with automatic save
    - Metadata filtering
    - Batch operations
    - Collection management
    """
    
    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        embeddings: Optional[LangChainJinaEmbeddings] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            config: Vector store configuration
            embeddings: Embedding model instance
        """
        self.config = config or VectorStoreConfig()
        self.embeddings = embeddings
        self.client = None
        self.collection = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the ChromaDB client and collection."""
        if self._initialized:
            return
        
        logger.info("Initializing ChromaDB vector store...")
        
        # Create client with persistence if directory specified
        if self.config.persist_directory:
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Using persistent storage at: {persist_path}")
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info("Using in-memory storage")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric}
        )
        
        self._initialized = True
        logger.info(f"Collection '{self.config.collection_name}' initialized with {self.collection.count()} documents")
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Add documents with their embeddings to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to process per batch
            show_progress: Whether to show progress
            
        Returns:
            Statistics about the operation
        """
        if not self._initialized:
            self.initialize()
        
        if self.embeddings is None:
            raise ValueError("Embeddings model not provided")
        
        # Initialize embeddings model
        self.embeddings.jina.initialize()
        
        stats = {
            "total_documents": len(documents),
            "documents_added": 0,
            "errors": [],
            "start_time": datetime.now().isoformat()
        }
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        # Process in batches
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(documents), batch_size):
            batch_docs = documents[batch_idx:batch_idx + batch_size]
            current_batch = batch_idx // batch_size + 1
            
            if show_progress:
                logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch_docs)} documents)")
            
            try:
                # Extract texts and metadata
                texts = [doc.page_content for doc in batch_docs]
                metadatas = [doc.metadata for doc in batch_docs]
                
                # Generate unique IDs
                ids = [
                    f"{meta.get('fileName', 'doc')}_{meta.get('Page No', 0)}_{meta.get('chunk_index', batch_idx + i)}"
                    for i, meta in enumerate(metadatas)
                ]
                
                # Generate embeddings
                embeddings = self.embeddings.embed_documents(texts)
                
                # Sanitize metadata for ChromaDB (only str, int, float, bool allowed)
                sanitized_metadatas = []
                for meta in metadatas:
                    sanitized = {}
                    for key, value in meta.items():
                        if isinstance(value, (str, int, float, bool)):
                            sanitized[key] = value
                        else:
                            sanitized[key] = str(value)
                    sanitized_metadatas.append(sanitized)
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=sanitized_metadatas
                )
                
                stats["documents_added"] += len(batch_docs)
                
            except Exception as e:
                error_msg = f"Error in batch {current_batch}: {str(e)}"
                logger.error(error_msg)
                stats["errors"].append(error_msg)
        
        stats["end_time"] = datetime.now().isoformat()
        stats["collection_size"] = self.collection.count()
        
        logger.info(f"Added {stats['documents_added']}/{stats['total_documents']} documents")
        
        return stats
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        if not self._initialized:
            self.initialize()
        
        if self.embeddings is None:
            raise ValueError("Embeddings model not provided")
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Build where clause for filtering
        where = None
        if filter_metadata:
            where = filter_metadata
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to LangChain Documents with scores
        documents_with_scores = []
        
        if results and results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0
                
                # Convert distance to similarity score (for cosine: 1 - distance)
                if self.config.distance_metric == "cosine":
                    score = 1 - distance
                else:
                    score = distance
                
                doc = Document(page_content=doc_text, metadata=metadata)
                documents_with_scores.append((doc, score))
        
        return documents_with_scores
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        if not self._initialized:
            self.initialize()
        
        return {
            "collection_name": self.config.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.config.persist_directory,
            "distance_metric": self.config.distance_metric
        }
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        if not self._initialized:
            self.initialize()
        
        self.client.delete_collection(self.config.collection_name)
        logger.info(f"Collection '{self.config.collection_name}' deleted")
        
        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric}
        )
    
    def persist(self) -> None:
        """
        Explicitly persist the database.
        
        Note: ChromaDB PersistentClient auto-persists, but this can be called
        for explicit saves.
        """
        if self.config.persist_directory and self.client:
            logger.info("Database persisted automatically by ChromaDB")


def create_vector_store(
    persist_directory: Optional[str] = None,
    collection_name: str = "rag_documents",
    embeddings: Optional[LangChainJinaEmbeddings] = None
) -> ChromaVectorStore:
    """
    Factory function to create a vector store.
    
    Args:
        persist_directory: Path for persistent storage
        collection_name: Name of the collection
        embeddings: Embedding model instance
        
    Returns:
        Configured ChromaVectorStore instance
    """
    config = VectorStoreConfig(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    store = ChromaVectorStore(config=config, embeddings=embeddings)
    store.initialize()
    
    return store


def load_chunks_from_json(chunks_path: Path) -> List[Document]:
    """
    Load chunks from the JSON file produced by the chunking pipeline.
    
    Args:
        chunks_path: Path to chunks.json file
        
    Returns:
        List of LangChain Document objects
    """
    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for chunk in data.get("chunks", []):
        doc = Document(
            page_content=chunk.get("content", ""),
            metadata=chunk.get("metadata", {})
        )
        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} chunks from {chunks_path}")
    return documents


if __name__ == "__main__":
    # Test the vector store
    print("Testing ChromaDB Vector Store...")
    
    project_root = Path(__file__).parent.parent.parent
    persist_dir = project_root / "data" / "vectorstore"
    
    # Create embeddings
    from embeddings import EmbeddingConfig
    
    embed_config = EmbeddingConfig(truncate_dim=512)
    embeddings = LangChainJinaEmbeddings(embed_config)
    
    # Create vector store
    store = create_vector_store(
        persist_directory=str(persist_dir),
        collection_name="test_collection",
        embeddings=embeddings
    )
    
    print(f"\nCollection stats: {store.get_collection_stats()}")
    
    # Test with sample documents
    sample_docs = [
        Document(
            page_content="How to set up the HomeHawk camera system",
            metadata={"fileName": "test", "Page No": 1, "Heading": "Setup"}
        ),
        Document(
            page_content="The LED indicator shows the camera status",
            metadata={"fileName": "test", "Page No": 2, "Heading": "Status"}
        )
    ]
    
    print(f"\nAdding {len(sample_docs)} test documents...")
    stats = store.add_documents(sample_docs)
    print(f"Stats: {stats}")
    
    # Test search
    query = "camera setup"
    print(f"\nSearching for: '{query}'")
    results = store.similarity_search(query, k=2)
    
    for doc, score in results:
        print(f"  Score: {score:.4f} - {doc.page_content[:50]}...")
    
    print("\n✓ Vector store test completed!")

