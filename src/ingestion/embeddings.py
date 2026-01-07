"""
Embedding Module for RAG Pipeline

Uses Jina Embeddings v4 from Hugging Face for generating high-quality
embeddings optimized for retrieval tasks.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass
import logging

import torch
from transformers import AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model."""
    model_name: str = "jinaai/jina-embeddings-v4"
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    torch_dtype: str = "float16"  # "float16", "float32", "bfloat16"
    task: str = "retrieval"  # "retrieval", "text-matching", "code"
    truncate_dim: Optional[int] = None  # Matryoshka: 128, 256, 512, 1024, 2048
    max_length: int = 8192  # Max tokens per text
    batch_size: int = 8


class JinaEmbeddings:
    """
    Jina Embeddings v4 wrapper for generating text embeddings.
    
    Features:
    - Task-specific adapters (retrieval, text-matching, code)
    - Matryoshka dimension reduction (2048 → 128)
    - Asymmetric query/passage encoding for retrieval
    - Multilingual support (30+ languages)
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the Jina Embeddings model.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.device = None
        self._initialized = False
    
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from config string."""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.torch_dtype, torch.float16)
    
    def _convert_embeddings_to_list(self, embeddings) -> List[List[float]]:
        """
        Convert embeddings to a list of lists of floats.
        
        Handles various return types from the model:
        - Single PyTorch tensor (batch)
        - List of individual PyTorch tensors
        - NumPy array
        - List of NumPy arrays
        - Already a list of lists
        
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
        
        # Case 3: List (could be list of tensors, arrays, or already lists)
        if isinstance(embeddings, list) and len(embeddings) > 0:
            converted = []
            for emb in embeddings:
                if hasattr(emb, 'cpu') and hasattr(emb, 'numpy'):
                    # PyTorch tensor
                    converted.append(emb.cpu().numpy().tolist())
                elif hasattr(emb, 'tolist'):
                    # NumPy array
                    converted.append(emb.tolist())
                else:
                    # Already a list
                    converted.append(emb)
            return converted
        
        # Fallback: return as-is (empty list or already correct format)
        return embeddings
    
    def initialize(self) -> None:
        """Load the embedding model."""
        if self._initialized:
            return
        
        logger.info(f"Loading Jina Embeddings v4 model: {self.config.model_name}")
        
        self.device = self._get_device()
        torch_dtype = self._get_torch_dtype()
        
        logger.info(f"Using device: {self.device}, dtype: {torch_dtype}")
        
        try:
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
            
            # Move to device
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            
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
        
        For retrieval tasks, documents are encoded as "passages".
        
        Args:
            texts: List of document texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            self.initialize()
        
        batch_size = batch_size or self.config.batch_size
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                embeddings = self.model.encode_text(
                    texts=batch_texts,
                    task=self.config.task,
                    prompt_name="passage",  # Documents are passages
                    truncate_dim=self.config.truncate_dim,
                    max_length=self.config.max_length
                )
                
                # Convert to list of lists - handle various return types
                embeddings = self._convert_embeddings_to_list(embeddings)
                
                all_embeddings.extend(embeddings)
                
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size}: {e}")
                raise
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.
        
        For retrieval tasks, queries are encoded differently than documents
        to optimize asymmetric retrieval.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        if not self._initialized:
            self.initialize()
        
        try:
            embedding = self.model.encode_text(
                texts=[text],
                task=self.config.task,
                prompt_name="query",  # Queries use query prompt
                truncate_dim=self.config.truncate_dim,
                max_length=self.config.max_length
            )
            
            # Convert to list - handle various return types
            embeddings_list = self._convert_embeddings_to_list(embedding)
            
            return embeddings_list[0]
            
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
        if self.config.truncate_dim:
            return self.config.truncate_dim
        return 2048  # Default Jina v4 dimension
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class LangChainJinaEmbeddings:
    """
    LangChain-compatible wrapper for Jina Embeddings.
    
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
    model_name: str = "jinaai/jina-embeddings-v4",
    device: str = "auto",
    truncate_dim: Optional[int] = None
) -> LangChainJinaEmbeddings:
    """
    Factory function to create embeddings instance.
    
    Args:
        model_name: Hugging Face model name
        device: Device to use ("auto", "cuda", "cpu", "mps")
        truncate_dim: Dimension to truncate embeddings to (Matryoshka)
        
    Returns:
        LangChain-compatible embeddings instance
    """
    config = EmbeddingConfig(
        model_name=model_name,
        device=device,
        truncate_dim=truncate_dim
    )
    return LangChainJinaEmbeddings(config)


if __name__ == "__main__":
    # Test the embeddings
    print("Testing Jina Embeddings v4...")
    
    config = EmbeddingConfig(
        truncate_dim=512  # Use smaller dimension for testing
    )
    
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

