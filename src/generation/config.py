"""
Configuration Management for Generation Module

Centralized configuration with environment variable support and validation.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers."""
    GOOGLE_GEMINI = "google_gemini"
    # Add more providers as needed: OPENAI, ANTHROPIC, etc.


class ResponseStyle(Enum):
    """Response style options."""
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"


@dataclass
class GenerationConfig:
    """
    Configuration for the generation pipeline.
    
    Supports loading from environment variables with sensible defaults.
    
    Attributes:
        provider: LLM provider to use
        model_name: Specific model name
        api_key: API key (loaded from environment)
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        response_style: Style of generated responses
        include_sources: Whether to include source citations
        max_context_chunks: Maximum number of context chunks to include
        stream: Whether to stream responses
    """
    # Provider settings
    provider: LLMProvider = LLMProvider.GOOGLE_GEMINI
    model_name: str = "gemini-2.5-flash-lite-preview-09-2025"
    api_key: Optional[str] = field(default=None, repr=False)
    
    # Generation parameters
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    
    # Response settings
    response_style: ResponseStyle = ResponseStyle.DETAILED
    include_sources: bool = True
    max_context_chunks: int = 5
    
    # Streaming
    stream: bool = False
    
    # Timeouts
    timeout_seconds: int = 60
    max_retries: int = 3
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "API key not found. Set GOOGLE_API_KEY environment variable "
                "or pass api_key to GenerationConfig."
            )
        
        # Validate parameters
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.max_context_chunks < 1:
            raise ValueError("max_context_chunks must be at least 1")
    
    @classmethod
    def from_env(cls) -> "GenerationConfig":
        """
        Create configuration from environment variables.
        
        Environment variables:
            GOOGLE_API_KEY: API key for Google Gemini
            LLM_MODEL: Model name (default: gemini-2.0-flash)
            LLM_TEMPERATURE: Temperature (default: 0.3)
            LLM_MAX_TOKENS: Max tokens (default: 2048)
            LLM_RESPONSE_STYLE: Response style (default: detailed)
        """
        return cls(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model_name=os.getenv("LLM_MODEL", "gemini-2.5-flash-lite-preview-09-2025"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            response_style=ResponseStyle(
                os.getenv("LLM_RESPONSE_STYLE", "detailed")
            ),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding sensitive data)."""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "response_style": self.response_style.value,
            "include_sources": self.include_sources,
            "max_context_chunks": self.max_context_chunks,
            "stream": self.stream,
        }


@dataclass  
class RAGConfig:
    """
    Complete RAG pipeline configuration.
    
    Combines retrieval and generation settings.
    """
    # Retrieval settings
    retrieval_k: int = 10  # Number of chunks to retrieve
    rerank_top_n: int = 5  # Number of chunks after reranking
    
    # Generation settings
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Vector store settings
    vectorstore_path: Optional[Path] = None
    collection_name: str = "rag_documents"
    
    def __post_init__(self):
        if self.vectorstore_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.vectorstore_path = project_root / "data" / "vectorstore"
