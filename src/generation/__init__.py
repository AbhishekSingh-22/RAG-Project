"""
Generation Module for RAG Pipeline

This module handles answer generation using LLMs with retrieved context.

Components:
    - config: Configuration management for generation pipeline
    - llm_client: Unified LLM client with provider abstraction
    - prompts: Prompt templates for different response styles
    - rag_chain: Complete RAG pipeline combining retrieval + generation
    - cli: Command-line interface for querying

Usage:
    from generation import RAGChain, create_rag_chain
    
    # Quick start
    chain = create_rag_chain(response_style="detailed")
    response = chain.query("How do I set up the camera?")
    print(response.answer)
    
    # With custom config
    from generation import RAGConfig, GenerationConfig, ResponseStyle
    
    config = RAGConfig(
        retrieval_k=10,
        generation=GenerationConfig(
            response_style=ResponseStyle.TECHNICAL,
            temperature=0.2,
        )
    )
    chain = RAGChain(config)
"""

from .config import GenerationConfig, RAGConfig, ResponseStyle
from .llm_client import LLMClient, LLMResponse, LLMError
from .prompts import PromptTemplate, RAGPromptBuilder, ContextChunk
from .rag_chain import RAGChain, RAGResponse, Source, create_rag_chain

__all__ = [
    # Config
    "GenerationConfig",
    "RAGConfig", 
    "ResponseStyle",
    # LLM
    "LLMClient",
    "LLMResponse",
    "LLMError",
    # Prompts
    "PromptTemplate",
    "RAGPromptBuilder",
    "ContextChunk",
    # RAG Chain
    "RAGChain",
    "RAGResponse",
    "Source",
    "create_rag_chain",
]
