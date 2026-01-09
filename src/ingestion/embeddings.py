"""
Embedding Module for RAG Pipeline

Uses sentence-transformers all-mpnet-base-v2 model for generating
high-quality embeddings optimized for retrieval tasks.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass
import logging

import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model."""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "cuda"  # "cuda", "cpu", "mps"
    task: str = "retrieval"  # "retrieval" or other task (for compatibility)
    max_length: int = 384  # Max tokens per text for all-mpnet-base-v2
    batch_size: int = 32


class JinaEmbeddings:
    """
    Sentence Transformers all-mpnet-base-v2 wrapper for generating text embeddings.
    
    Features:
    - Optimized for semantic search and retrieval tasks
    - Lightweight model (~438M params) that fits on GPU
    - Mean pooling for symmetric embeddings
    - Multilingual support
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding model.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.device = None
        self._initialized = False
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _convert_embeddings_to_list(self, embeddings) -> List[List[float]]:
        """
        Convert embeddings to a list of lists of floats.
        
        Handles various return types from the model.
        
        Args:
            embeddings: Raw embeddings from the model
            
        Returns:
            List of embedding vectors as Python lists
        """
        # Case 1: Single PyTorch tensor (batch of embeddings)
        if hasattr(embeddings, 'cpu') and hasattr(embeddings, 'numpy'):
            return embeddings.cpu().numpy().tolist()
        
        # Case 2: Single NumPy array
        if hasattr(embeddings, 'tolist') and not isinstance(embeddings, list):
            return embeddings.tolist()
        
        # Case 3: Already a list
        if isinstance(embeddings, list):
            return embeddings
        
        # Fallback
        return embeddings
    
    def initialize(self) -> None:
        """Load the embedding model."""
        if self._initialized:
            return
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.device = self._get_device()
        
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load SentenceTransformer model
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.device,
                trust_remote_code=True
            )
            
            self._initialized = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def embed_documents(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for documents/passages.
        
        Args:
            texts: List of document texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            self.initialize()
        
        batch_size = batch_size or self.config.batch_size
        
        try:
            # SentenceTransformer handles batching automatically
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=False,
                show_progress_bar=True
            )
            
            # Convert to list of lists
            embeddings = self._convert_embeddings_to_list(embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        if not self._initialized:
            self.initialize()
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_tensor=False
            )
            
            # Convert to list
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            elif isinstance(embedding, list):
                return embedding
            else:
                return list(embedding)
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def embed_texts(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generic embedding function for texts.
        
        Args:
            texts: List of texts to embed
            is_query: If True, use query encoding; else document encoding
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if is_query:
            return [self.embed_query(text) for text in texts]
        else:
            return self.embed_documents(texts, batch_size)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return 768  # all-mpnet-base-v2 outputs 768-dim embeddings
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class LangChainJinaEmbeddings:
    """
    LangChain-compatible wrapper for Sentence Transformers embeddings.
    
    Implements the interface expected by LangChain vector stores.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize with optional config."""
        self.jina = JinaEmbeddings(config)
        self._config = config or EmbeddingConfig()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs - LangChain interface."""
        return self.jina.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text - LangChain interface."""
        return self.jina.embed_query(text)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.jina.embedding_dimension


def create_embeddings(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "cuda",
) -> LangChainJinaEmbeddings:
    """
    Factory function to create embeddings instance.
    
    Args:
        model_name: Hugging Face model name
        device: Device to use ("cuda" or "cpu")
        
    Returns:
        LangChain-compatible embeddings instance
    """
    config = EmbeddingConfig(
        model_name=model_name,
        device=device
    )
    return LangChainJinaEmbeddings(config)


if __name__ == "__main__":
    # Test the embeddings
    print("Testing Sentence Transformers Embeddings...")
    
    config = EmbeddingConfig()
    
    embeddings = LangChainJinaEmbeddings(config)
    
    # Test documents
    docs = [
        "How do I set up the HomeHawk camera?",
        "The LED indicator shows the camera status."
    ]
    
    print(f"\nEmbedding {len(docs)} documents...")
    doc_embeddings = embeddings.embed_documents(docs)
    print(f"Document embeddings shape: {len(doc_embeddings)} x {len(doc_embeddings[0])}")
    
    # Test query
    query = "camera setup instructions"
    print(f"\nEmbedding query: '{query}'")
    query_embedding = embeddings.embed_query(query)
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    print("\n✓ Embeddings test completed successfully!")

