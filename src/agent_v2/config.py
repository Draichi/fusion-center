"""
LLM Configuration for Agent v2.

Handles dual-LLM setup using Ollama only.
"""

from dataclasses import dataclass

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from src.shared.config import settings


@dataclass
class DualLLMConfig:
    """Configuration for dual-LLM architecture using Ollama."""
    
    # Structured LLM - for JSON outputs (planning, analysis, correlation)
    structured_model: str = settings.structured_llm_model
    
    # Thinking LLM - for complex reasoning (reflection, verification, synthesis)
    thinking_model: str = settings.thinking_llm_model


def create_model(model_name: str) -> OpenAIChatModel:
    """
    Create an Ollama model instance for PydanticAI.
    
    Args:
        model_name: Name of the Ollama model (e.g., "qwen2.5:7b", "llama3.2")
    
    Returns:
        OpenAIChatModel instance configured with OllamaProvider.
    """
    return OpenAIChatModel(
        model_name=model_name,
        provider=OllamaProvider(base_url=settings.ollama_base_url),
    )

