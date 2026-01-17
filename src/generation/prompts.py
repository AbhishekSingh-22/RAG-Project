"""
Prompt Templates Module

Provides structured prompt templates for RAG generation.
Supports different response styles and customization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum
from string import Template

# Handle both relative and absolute imports
try:
    from .config import ResponseStyle
except ImportError:
    from config import ResponseStyle


@dataclass
class ContextChunk:
    """
    Represents a context chunk for prompt building.
    
    Attributes:
        content: Text content of the chunk
        source: Source file name
        page: Page number
        heading: Section heading
        sub_heading: Sub-section heading
        score: Relevance score
    """
    content: str
    source: str = ""
    page: int = 0
    heading: str = ""
    sub_heading: str = ""
    score: float = 0.0
    
    def format_source(self) -> str:
        """Format source information for citation."""
        parts = []
        if self.source:
            parts.append(self.source)
        if self.page:
            parts.append(f"Page {self.page}")
        if self.heading:
            parts.append(f"§ {self.heading}")
        return " | ".join(parts) if parts else "Unknown source"
    
    @classmethod
    def from_document(cls, doc: Any, score: float = 0.0) -> "ContextChunk":
        """Create from a LangChain Document."""
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        return cls(
            content=doc.page_content if hasattr(doc, 'page_content') else str(doc),
            source=metadata.get('fileName', ''),
            page=metadata.get('Page No', 0),
            heading=metadata.get('Heading', ''),
            sub_heading=metadata.get('SubHeading', ''),
            score=score,
        )


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""
    
    @abstractmethod
    def build(self, query: str, context: List[ContextChunk]) -> str:
        """Build the complete prompt."""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        pass


class RAGPromptBuilder:
    """
    Builder for RAG prompts with different styles and configurations.
    
    Supports:
    - Multiple response styles (concise, detailed, technical, conversational)
    - Source citation formatting
    - Context organization
    - Custom instructions
    """
    
    # System prompts for different styles
    SYSTEM_PROMPTS = {
        ResponseStyle.CONCISE: """You are a helpful assistant that provides brief, accurate answers based on the provided context.
Keep responses short and to the point. If the answer isn't in the context, say so clearly.""",
        
        ResponseStyle.DETAILED: """You are a knowledgeable assistant that provides comprehensive answers based on the provided context.
Give detailed explanations with relevant information from the context. Structure your response clearly.
If the context doesn't contain enough information, acknowledge the limitations.""",
        
        ResponseStyle.TECHNICAL: """You are a technical documentation expert that provides precise, technical answers.
Use exact terminology from the source material. Include specific steps, parameters, and technical details.
Reference specific sections when applicable. If information is missing, clearly state what's not covered.""",
        
        ResponseStyle.CONVERSATIONAL: """You are a friendly helper explaining things in simple terms.
Use everyday language and helpful analogies. Break down complex topics into easy-to-understand parts.
If you can't find the answer, suggest what the user might look for instead.""",
    }
    
    # User prompt templates
    USER_PROMPT_TEMPLATE = """Based on the following context, please answer the question.

## Context
{context}

## Question
{query}

## Instructions
{instructions}

Please provide your answer:"""

    CONTEXT_CHUNK_TEMPLATE = """### Source {index}: {source}
{content}
"""

    NO_CONTEXT_TEMPLATE = """I don't have any relevant context to answer this question.
The question was: {query}

Please note that I can only answer questions based on the provided documentation."""

    def __init__(
        self,
        style: ResponseStyle = ResponseStyle.DETAILED,
        include_sources: bool = True,
        custom_instructions: Optional[str] = None,
    ):
        """
        Initialize the prompt builder.
        
        Args:
            style: Response style to use
            include_sources: Whether to request source citations
            custom_instructions: Additional custom instructions
        """
        self.style = style
        self.include_sources = include_sources
        self.custom_instructions = custom_instructions
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the current style."""
        return self.SYSTEM_PROMPTS.get(self.style, self.SYSTEM_PROMPTS[ResponseStyle.DETAILED])
    
    def _format_context(self, chunks: List[ContextChunk]) -> str:
        """Format context chunks into a readable string."""
        if not chunks:
            return "*No relevant context found.*"
        
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            source_info = chunk.format_source()
            formatted = self.CONTEXT_CHUNK_TEMPLATE.format(
                index=i,
                source=source_info,
                content=chunk.content.strip(),
            )
            formatted_chunks.append(formatted)
        
        return "\n".join(formatted_chunks)
    
    def _get_instructions(self) -> str:
        """Build instruction text based on configuration."""
        instructions = []
        
        # Style-specific instructions
        if self.style == ResponseStyle.CONCISE:
            instructions.append("- Keep your answer brief (2-3 sentences if possible)")
        elif self.style == ResponseStyle.DETAILED:
            instructions.append("- Provide a comprehensive answer with relevant details")
        elif self.style == ResponseStyle.TECHNICAL:
            instructions.append("- Include specific technical details, steps, and parameters")
        elif self.style == ResponseStyle.CONVERSATIONAL:
            instructions.append("- Explain in simple, everyday language")
        
        # Source citation instructions
        if self.include_sources:
            instructions.append("- Cite sources using [Source N] format when referencing specific information")
        
        # Custom instructions
        if self.custom_instructions:
            instructions.append(f"- {self.custom_instructions}")
        
        # General instructions
        instructions.extend([
            "- Only use information from the provided context",
            "- If the context doesn't contain the answer, clearly state that",
            "- Do not make up information not present in the context",
        ])
        
        return "\n".join(instructions)
    
    def build(self, query: str, context: List[ContextChunk]) -> str:
        """
        Build the complete user prompt.
        
        Args:
            query: User's question
            context: List of relevant context chunks
            
        Returns:
            Formatted prompt string
        """
        if not context:
            return self.NO_CONTEXT_TEMPLATE.format(query=query)
        
        formatted_context = self._format_context(context)
        instructions = self._get_instructions()
        
        return self.USER_PROMPT_TEMPLATE.format(
            context=formatted_context,
            query=query,
            instructions=instructions,
        )
    
    def build_with_history(
        self, 
        query: str, 
        context: List[ContextChunk],
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Build prompt with conversation history for multi-turn conversations.
        
        Args:
            query: Current user question
            context: Relevant context chunks
            chat_history: List of {"role": "user"|"assistant", "content": "..."} dicts
            
        Returns:
            Formatted prompt with history
        """
        # Format chat history
        history_text = ""
        if chat_history:
            history_parts = []
            for msg in chat_history[-4:]:  # Keep last 4 messages for context
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"**{role}**: {msg['content']}")
            history_text = "\n\n## Previous Conversation\n" + "\n\n".join(history_parts)
        
        # Build base prompt
        base_prompt = self.build(query, context)
        
        # Insert history before the question
        if history_text:
            base_prompt = base_prompt.replace(
                "## Question",
                f"{history_text}\n\n## Current Question"
            )
        
        return base_prompt


class FollowUpPromptBuilder:
    """Builder for follow-up question prompts."""
    
    FOLLOW_UP_TEMPLATE = """Based on our previous conversation and the context, please answer this follow-up question.

## Previous Answer Summary
{previous_answer_summary}

## Additional Context
{context}

## Follow-up Question
{query}

Please provide your answer, building on the previous discussion:"""

    def __init__(self, style: ResponseStyle = ResponseStyle.DETAILED):
        self.style = style
        self.base_builder = RAGPromptBuilder(style=style)
    
    def build(
        self,
        query: str,
        context: List[ContextChunk],
        previous_answer: str,
    ) -> str:
        """Build a follow-up prompt."""
        # Summarize previous answer (first 500 chars)
        summary = previous_answer[:500]
        if len(previous_answer) > 500:
            summary += "..."
        
        formatted_context = self.base_builder._format_context(context)
        
        return self.FOLLOW_UP_TEMPLATE.format(
            previous_answer_summary=summary,
            context=formatted_context,
            query=query,
        )
    
    def get_system_prompt(self) -> str:
        """Get system prompt."""
        return self.base_builder.get_system_prompt()
