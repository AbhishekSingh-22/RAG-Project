"""
LLM Client Module

Provides a unified interface for interacting with LLM providers.
Implements retry logic, error handling, and response parsing.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generator
from abc import ABC, abstractmethod
from enum import Enum

import google.generativeai as genai

# Handle both relative and absolute imports
try:
    from .config import GenerationConfig, LLMProvider
except ImportError:
    from config import GenerationConfig, LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Error connecting to LLM provider."""
    pass


class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""
    pass


class LLMContentFilterError(LLMError):
    """Content was filtered by safety settings."""
    pass


@dataclass
class LLMResponse:
    """
    Structured response from LLM.
    
    Attributes:
        content: Generated text content
        model: Model used for generation
        finish_reason: Why generation stopped
        usage: Token usage statistics
        latency_ms: Response latency in milliseconds
        raw_response: Original response object
    """
    content: str
    model: str
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw_response: Any = None
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)."""
        return self.usage.get("total_tokens", 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding raw response)."""
        return {
            "content": self.content,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Generate a streaming response."""
        pass


class GeminiClient(BaseLLMClient):
    """
    Google Gemini LLM client.
    
    Implements generation with retry logic and proper error handling.
    """
    
    def __init__(self, config: GenerationConfig):
        """
        Initialize the Gemini client.
        
        Args:
            config: Generation configuration
        """
        self.config = config
        self._model = None
        self._initialized = False
    
    def _initialize(self) -> None:
        """Lazily initialize the Gemini client."""
        if self._initialized:
            return
        
        genai.configure(api_key=self.config.api_key)
        
        # Configure generation settings
        generation_config = genai.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )
        
        # Configure safety settings (allow most content for RAG use case)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        
        self._model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        
        self._initialized = True
        logger.info(f"Initialized Gemini client with model: {self.config.model_name}")
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            LLMError: On generation failure after retries
        """
        self._initialize()
        
        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                response = self._model.generate_content(full_prompt)
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Check for blocked content
                if not response.candidates:
                    raise LLMContentFilterError("Response was blocked by safety filters")
                
                candidate = response.candidates[0]
                
                # Extract text content
                content = ""
                if candidate.content and candidate.content.parts:
                    content = "".join(
                        part.text for part in candidate.content.parts if hasattr(part, 'text')
                    )
                
                # Build usage stats
                usage = {}
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = {
                        "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                        "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                        "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
                    }
                
                return LLMResponse(
                    content=content,
                    model=self.config.model_name,
                    finish_reason=str(candidate.finish_reason) if candidate.finish_reason else "stop",
                    usage=usage,
                    latency_ms=latency_ms,
                    raw_response=response,
                )
                
            except LLMContentFilterError:
                raise  # Don't retry content filter errors
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check for rate limiting
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                # Check for connection errors
                if "connection" in error_str or "timeout" in error_str:
                    wait_time = (2 ** attempt)
                    logger.warning(f"Connection error. Retry {attempt + 1}/{self.config.max_retries}...")
                    time.sleep(wait_time)
                    continue
                
                # Unknown error - log and retry
                logger.error(f"Generation error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue
                
                raise LLMError(f"Generation failed after {self.config.max_retries} attempts: {e}")
        
        raise LLMError(f"Generation failed: {last_error}")
    
    def generate_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            
        Yields:
            Text chunks as they are generated
        """
        self._initialize()
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            response = self._model.generate_content(full_prompt, stream=True)
            
            for chunk in response:
                if chunk.candidates:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                yield part.text
                                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise LLMError(f"Streaming generation failed: {e}")


class LLMClient:
    """
    Unified LLM client that delegates to provider-specific implementations.
    
    Factory pattern for creating appropriate client based on config.
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """
        Initialize the LLM client.
        
        Args:
            config: Generation configuration (uses defaults if not provided)
        """
        self.config = config or GenerationConfig()
        self._client = self._create_client()
    
    def _create_client(self) -> BaseLLMClient:
        """Create the appropriate client based on provider."""
        if self.config.provider == LLMProvider.GOOGLE_GEMINI:
            return GeminiClient(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate a response."""
        return self._client.generate(prompt, system_prompt)
    
    def generate_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Generate a streaming response."""
        yield from self._client.generate_stream(prompt, system_prompt)
    
    def __repr__(self) -> str:
        return f"LLMClient(provider={self.config.provider.value}, model={self.config.model_name})"
