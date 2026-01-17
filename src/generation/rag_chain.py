"""
RAG Chain Module

Combines retrieval and generation into a unified pipeline.
Handles the complete flow from query to answer.
"""

import sys
import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

# Add ingestion and retrieval modules to path
_current_dir = Path(__file__).parent
sys.path.insert(0, str(_current_dir))
sys.path.insert(0, str(_current_dir.parent / "ingestion"))
sys.path.insert(0, str(_current_dir.parent / "retrieval"))

from langchain_core.documents import Document

# Handle both relative and absolute imports
try:
    from .config import GenerationConfig, RAGConfig, ResponseStyle
    from .llm_client import LLMClient, LLMResponse, LLMError
    from .prompts import RAGPromptBuilder, ContextChunk, FollowUpPromptBuilder
except ImportError:
    from config import GenerationConfig, RAGConfig, ResponseStyle
    from llm_client import LLMClient, LLMResponse, LLMError
    from prompts import RAGPromptBuilder, ContextChunk, FollowUpPromptBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Represents a source citation."""
    file_name: str
    page: int
    heading: str
    sub_heading: str
    relevance_score: float
    content_preview: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_name": self.file_name,
            "page": self.page,
            "heading": self.heading,
            "sub_heading": self.sub_heading,
            "relevance_score": round(self.relevance_score, 4),
            "content_preview": self.content_preview,
        }


@dataclass
class RAGResponse:
    """
    Complete RAG response with answer and metadata.
    
    Attributes:
        query: Original user query
        answer: Generated answer
        sources: List of source citations
        context_chunks: Number of context chunks used
        retrieval_time_ms: Time spent on retrieval
        generation_time_ms: Time spent on generation
        total_time_ms: Total processing time
        model: Model used for generation
        tokens_used: Token usage statistics
    """
    query: str
    answer: str
    sources: List[Source] = field(default_factory=list)
    context_chunks: int = 0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    model: str = ""
    tokens_used: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return self.error is None and bool(self.answer)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "context_chunks": self.context_chunks,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "generation_time_ms": round(self.generation_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "model": self.model,
            "tokens_used": self.tokens_used,
            "success": self.success,
            "error": self.error,
        }
    
    def format_answer_with_sources(self) -> str:
        """Format answer with source citations."""
        result = self.answer
        
        if self.sources:
            result += "\n\n---\n**Sources:**\n"
            for i, source in enumerate(self.sources, 1):
                source_info = f"{source.file_name}"
                if source.page:
                    source_info += f", Page {source.page}"
                if source.heading:
                    source_info += f" - {source.heading}"
                result += f"{i}. {source_info}\n"
        
        return result


class RAGChain:
    """
    Complete RAG pipeline combining retrieval and generation.
    
    Features:
    - Lazy initialization of components
    - Configurable retrieval and generation parameters
    - Error handling and fallback responses
    - Performance tracking
    - Multi-turn conversation support
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the RAG chain.
        
        Args:
            config: RAG configuration (uses defaults if not provided)
        """
        self.config = config or RAGConfig()
        self._retriever = None
        self._llm_client = None
        self._prompt_builder = None
        self._initialized = False
        
        # Conversation history for multi-turn
        self._conversation_history: List[Dict[str, str]] = []
    
    def _initialize(self) -> None:
        """Lazily initialize all components."""
        if self._initialized:
            return
        
        logger.info("Initializing RAG chain components...")
        
        # Import retriever (lazy import to avoid circular dependencies)
        from retrieval import Retriever, RetrievalConfig
        
        # Initialize retriever
        retriever_config = RetrievalConfig(
            persist_dir=self.config.vectorstore_path,
            collection_name=self.config.collection_name,
            default_k=self.config.retrieval_k,
            rerank_top_n=self.config.rerank_top_n,
        )
        self._retriever = Retriever(retriever_config)
        
        # Initialize LLM client
        self._llm_client = LLMClient(self.config.generation)
        
        # Initialize prompt builder
        self._prompt_builder = RAGPromptBuilder(
            style=self.config.generation.response_style,
            include_sources=self.config.generation.include_sources,
        )
        
        self._initialized = True
        logger.info("RAG chain initialized successfully")
    
    def _retrieve_context(
        self, 
        query: str
    ) -> Tuple[List[ContextChunk], float]:
        """
        Retrieve relevant context chunks.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (context chunks, retrieval time in ms)
        """
        start_time = time.time()
        
        # Get results from retriever
        results = self._retriever.search_with_rerank(
            query=query,
            initial_k=self.config.retrieval_k,
            final_k=self.config.rerank_top_n,
        )
        
        retrieval_time = (time.time() - start_time) * 1000
        
        # Convert to ContextChunks
        chunks = []
        for doc, score in results:
            chunk = ContextChunk.from_document(doc, score)
            chunks.append(chunk)
        
        return chunks, retrieval_time
    
    def _extract_sources(self, chunks: List[ContextChunk]) -> List[Source]:
        """Extract source citations from context chunks."""
        sources = []
        for chunk in chunks:
            source = Source(
                file_name=chunk.source,
                page=chunk.page,
                heading=chunk.heading,
                sub_heading=chunk.sub_heading,
                relevance_score=chunk.score,
                content_preview=chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content,
            )
            sources.append(source)
        return sources
    
    def query(
        self, 
        query: str,
        include_history: bool = False,
    ) -> RAGResponse:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: User's question
            include_history: Whether to include conversation history
            
        Returns:
            RAGResponse with answer and metadata
        """
        total_start = time.time()
        
        try:
            self._initialize()
            
            # Step 1: Retrieve relevant context
            logger.info(f"Processing query: {query[:50]}...")
            chunks, retrieval_time = self._retrieve_context(query)
            
            if not chunks:
                logger.warning("No relevant context found")
                return RAGResponse(
                    query=query,
                    answer="I couldn't find any relevant information in the documentation to answer your question. "
                           "Please try rephrasing your question or ask about a different topic.",
                    context_chunks=0,
                    retrieval_time_ms=retrieval_time,
                    total_time_ms=(time.time() - total_start) * 1000,
                )
            
            # Step 2: Build prompt
            if include_history and self._conversation_history:
                prompt = self._prompt_builder.build_with_history(
                    query=query,
                    context=chunks,
                    chat_history=self._conversation_history,
                )
            else:
                prompt = self._prompt_builder.build(query=query, context=chunks)
            
            system_prompt = self._prompt_builder.get_system_prompt()
            
            # Step 3: Generate answer
            gen_start = time.time()
            llm_response = self._llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
            )
            generation_time = (time.time() - gen_start) * 1000
            
            # Step 4: Build response
            sources = self._extract_sources(chunks)
            total_time = (time.time() - total_start) * 1000
            
            response = RAGResponse(
                query=query,
                answer=llm_response.content,
                sources=sources,
                context_chunks=len(chunks),
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=total_time,
                model=llm_response.model,
                tokens_used=llm_response.usage,
            )
            
            # Update conversation history
            self._conversation_history.append({"role": "user", "content": query})
            self._conversation_history.append({"role": "assistant", "content": llm_response.content})
            
            # Keep history bounded
            if len(self._conversation_history) > 10:
                self._conversation_history = self._conversation_history[-10:]
            
            logger.info(
                f"Query completed: {len(chunks)} chunks, "
                f"retrieval={retrieval_time:.0f}ms, generation={generation_time:.0f}ms"
            )
            
            return response
            
        except LLMError as e:
            logger.error(f"LLM error: {e}")
            return RAGResponse(
                query=query,
                answer="",
                error=f"Generation error: {str(e)}",
                total_time_ms=(time.time() - total_start) * 1000,
            )
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return RAGResponse(
                query=query,
                answer="",
                error=f"An unexpected error occurred: {str(e)}",
                total_time_ms=(time.time() - total_start) * 1000,
            )
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        self._initialize()
        
        retriever_stats = self._retriever.get_stats()
        
        return {
            "retriever": retriever_stats,
            "generation": self.config.generation.to_dict(),
            "conversation_history_length": len(self._conversation_history),
        }


def create_rag_chain(
    response_style: str = "detailed",
    retrieval_k: int = 10,
    rerank_top_n: int = 5,
) -> RAGChain:
    """
    Factory function to create a RAG chain with common configurations.
    
    Args:
        response_style: One of "concise", "detailed", "technical", "conversational"
        retrieval_k: Number of chunks to retrieve
        rerank_top_n: Number of chunks after reranking
        
    Returns:
        Configured RAGChain instance
    """
    style = ResponseStyle(response_style)
    
    gen_config = GenerationConfig(response_style=style)
    rag_config = RAGConfig(
        retrieval_k=retrieval_k,
        rerank_top_n=rerank_top_n,
        generation=gen_config,
    )
    
    return RAGChain(rag_config)
