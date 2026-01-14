"""LLM backends for voice_agent."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator

import anthropic
import ollama as ollama_client
from dotenv import load_dotenv

if TYPE_CHECKING:
    from ..core.types import ConversationMessage

load_dotenv(Path(__file__).parents[3] / ".env")


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list["ConversationMessage"] | None = None,
    ) -> str:
        """Generate a complete response.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            history: Optional conversation history for context

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list["ConversationMessage"] | None = None,
    ) -> Generator:
        """Stream response tokens.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            history: Optional conversation history for context

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

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list["ConversationMessage"] | None = None,
    ) -> list[dict]:
        """Build messages list from history and current prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list["ConversationMessage"] | None = None,
    ) -> str:
        """Generate a complete response."""
        messages = self._build_messages(prompt, system_prompt, history)
        response = ollama_client.chat(
            model=self.model,
            messages=messages,
        )
        return response.message.content

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list["ConversationMessage"] | None = None,
    ) -> Generator:
        """Stream response tokens."""
        messages = self._build_messages(prompt, system_prompt, history)
        for chunk in ollama_client.chat(
            model=self.model,
            messages=messages,
            stream=True,
        ):
            yield _OllamaStreamChunk(chunk.message.content)


class _OllamaStreamChunk:
    """Wrapper to provide consistent interface for Ollama chat streaming."""

    def __init__(self, text: str):
        self.response = text


class ClaudeBackend(LLMBackend):
    """LLM backend using Anthropic's Claude API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        tools: list[dict] | None = None,
        execute_tool: Callable[[Any, str, dict], str] | None = None,
        max_tool_iterations: int = 10,
    ):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in .env file or passed as argument")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.tools = tools
        self.execute_tool = execute_tool
        self.max_tool_iterations = max_tool_iterations
        self._context: Any = None  # Context for tool execution

    def set_context(self, context: Any) -> None:
        """Set the context for tool execution.

        Args:
            context: Context object passed to execute_tool
        """
        self._context = context

    @property
    def name(self) -> str:
        return f"claude-{self.model}"

    def _build_messages(
        self,
        prompt: str,
        history: list["ConversationMessage"] | None = None,
    ) -> list[dict]:
        """Build messages list from history and current prompt."""
        messages = []
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list["ConversationMessage"] | None = None,
    ) -> str:
        """Generate a complete response."""
        messages = self._build_messages(prompt, history)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt or "",
            messages=messages,
        )
        return message.content[0].text

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list["ConversationMessage"] | None = None,
    ) -> Generator:
        """Stream response tokens with tool use support.

        If tools are configured, this method handles the tool use loop:
        1. Call Claude with the prompt
        2. If Claude uses a tool, execute it and send the result back
        3. Continue until Claude provides a final text response
        """
        # Build messages with history
        messages = self._build_messages(prompt, history)

        # If no tools configured, use simple streaming
        if not self.tools:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=1024,
                system=system_prompt or "",
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield _ClaudeStreamChunk(text)
            return

        # Tool use flow
        for _ in range(self.max_tool_iterations):
            # Call Claude with tools
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt or "",
                tools=self.tools,
                messages=messages,
            )

            # Extract text and tool_use blocks
            text_blocks = []
            tool_uses = []

            for block in response.content:
                if block.type == "text":
                    text_blocks.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            # Yield accumulated text
            full_text = "".join(text_blocks)
            if full_text:
                yield _ClaudeStreamChunk(full_text)

            # If no tool use, we're done
            if not tool_uses:
                return

            # Execute tools
            if self.execute_tool is None:
                yield _ClaudeStreamChunk("\n[Error: Tool use requested but no executor configured]")
                return

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool and collect results
            tool_results = []
            for tool_use in tool_uses:
                result = self.execute_tool(self._context, tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                })

            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results})

        # Max iterations reached
        yield _ClaudeStreamChunk("\n[Error: Max tool iterations reached]")


class _ClaudeStreamChunk:
    """Wrapper to match Ollama's stream chunk interface."""

    def __init__(self, text: str):
        self.response = text


def create_llm_backend(
    backend: str = "ollama",
    model: str | None = None,
    tools: list[dict] | None = None,
    execute_tool: Callable[[Any, str, dict], str] | None = None,
    max_tool_iterations: int = 10,
) -> LLMBackend:
    """Factory function to create an LLM backend.

    Args:
        backend: Backend type ("ollama" or "claude")
        model: Model identifier (optional, uses defaults if not specified)
        tools: List of tool definitions for Claude tool use (Claude only)
        execute_tool: Function to execute tools (signature: (context, tool_name, tool_input) -> str)
        max_tool_iterations: Maximum number of tool use iterations (default: 10)

    Returns:
        Configured LLM backend instance
    """
    if backend == "ollama":
        return OllamaBackend(model=model or "llama3.2:3b")
    elif backend == "claude":
        return ClaudeBackend(
            model=model,
            tools=tools,
            execute_tool=execute_tool,
            max_tool_iterations=max_tool_iterations,
        )
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")
