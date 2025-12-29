"""LLM backends for voice_agent."""

from abc import ABC, abstractmethod
from typing import Generator
import ollama as ollama_client


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a complete response.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> Generator:
        """Stream response tokens.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions

        Yields:
            Response chunks (with .response attribute)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        pass


class OllamaBackend(LLMBackend):
    """LLM backend using Ollama for local inference."""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model

    @property
    def name(self) -> str:
        return f"ollama-{self.model}"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a complete response."""
        response = ollama_client.generate(
            model=self.model,
            prompt=prompt,
            system=system_prompt,
        )
        return response.response

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> Generator:
        """Stream response tokens."""
        return ollama_client.generate(
            model=self.model,
            prompt=prompt,
            system=system_prompt,
            stream=True,
        )


def create_llm_backend(
    backend: str = "ollama",
    model: str = "llama3.2:3b",
) -> LLMBackend:
    """Factory function to create an LLM backend.

    Args:
        backend: Backend type ("ollama" for now)
        model: Model identifier

    Returns:
        Configured LLM backend instance
    """
    if backend == "ollama":
        return OllamaBackend(model=model)
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
