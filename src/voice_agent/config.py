"""Configuration for voice_agent."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# Backend type definitions for IDE autocompletion
STTBackendType = Literal["whisper", "faster_whisper"]
TTSBackendType = Literal["edge"]
LLMBackendType = Literal["ollama", "claude"]

# Parameter types for autocompletion
WhisperModelSize = Literal["tiny", "base", "small", "medium", "large-v3", "turbo"]
DeviceType = Literal["cpu", "cuda"]

# Claude model types
ClaudeModel = Literal[
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]

# Ollama recommended models
OllamaModel = Literal[
    "llama3.2:3b",
    "ministral-3",
]

# Common Edge TTS voices (use `edge-tts --list-voices` for full list)
EdgeVoice = Literal[
    # English
    "en-US-AvaMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
    "en-US-EmmaMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    # French
    "fr-FR-DeniseNeural",
    "fr-FR-HenriNeural",
    "fr-FR-VivienneMultilingualNeural",
    "fr-FR-RemyMultilingualNeural",
    # Spanish
    "es-ES-ElviraNeural",
    "es-ES-AlvaroNeural",
    # German
    "de-DE-KatjaNeural",
    "de-DE-ConradNeural",
]


# Backend registry with descriptions and defaults
STT_BACKENDS: dict[str, dict] = {
    "whisper": {
        "description": "OpenAI Whisper - Original implementation",
        "default_model": "small",
        "default_device": "cpu",
    },
    "faster_whisper": {
        "description": "Faster Whisper - CTranslate2 based, ~4x faster",
        "default_model": "small",
        "default_device": "cpu",
    },
}

TTS_BACKENDS: dict[str, dict] = {
    "edge": {
        "description": "Microsoft Edge TTS - Free, high-quality voices",
        "default_voice": "en-US-AvaMultilingualNeural",
    },
}

LLM_BACKENDS: dict[str, dict] = {
    "ollama": {
        "description": "Ollama - Local inference with various models",
        "default_model": "llama3.2:3b",
    },
    "claude": {
        "description": "Anthropic Claude - Cloud API (requires ANTHROPIC_API_KEY)",
        "default_model": "claude-haiku-4-5",
    },
}


def list_stt_backends() -> dict[str, str]:
    """List available STT backends with descriptions.

    Returns:
        Dictionary mapping backend name to description
    """
    return {name: info["description"] for name, info in STT_BACKENDS.items()}


def list_tts_backends() -> dict[str, str]:
    """List available TTS backends with descriptions.

    Returns:
        Dictionary mapping backend name to description
    """
    return {name: info["description"] for name, info in TTS_BACKENDS.items()}


def list_llm_backends() -> dict[str, str]:
    """List available LLM backends with descriptions.

    Returns:
        Dictionary mapping backend name to description
    """
    return {name: info["description"] for name, info in LLM_BACKENDS.items()}


def get_stt_defaults(backend: STTBackendType) -> dict:
    """Get default configuration for an STT backend."""
    return STT_BACKENDS.get(backend, {})


def get_tts_defaults(backend: TTSBackendType) -> dict:
    """Get default configuration for a TTS backend."""
    return TTS_BACKENDS.get(backend, {})


def get_llm_defaults(backend: LLMBackendType) -> dict:
    """Get default configuration for an LLM backend."""
    return LLM_BACKENDS.get(backend, {})


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text."""

    backend: STTBackendType
    model_size: WhisperModelSize | None = None
    device: DeviceType | None = None


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech."""

    backend: TTSBackendType
    voice: EdgeVoice | None = None


@dataclass
class LLMConfig:
    """Configuration for LLM."""

    backend: LLMBackendType
    model: ClaudeModel | OllamaModel | str | None = None  # str allows Ollama models


@dataclass
class AgentConfig:
    """Main configuration for VoiceAgent."""

    system_prompt: str = "You are a helpful voice assistant."
    system_prompt_file: str | None = None

    stt: STTConfig = field(
        default_factory=lambda: STTConfig(
            backend="faster_whisper",
            model_size="small",
            device="cpu",
        )
    )
    tts: TTSConfig = field(
        default_factory=lambda: TTSConfig(
            backend="edge",
            voice="en-US-AvaMultilingualNeural",
        )
    )
    llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            backend="ollama",
            model="llama3.2:3b",
        )
    )

    def get_system_prompt(self) -> str:
        """Load system prompt from file if specified, otherwise return default."""
        if self.system_prompt_file:
            path = Path(self.system_prompt_file)
            if path.exists():
                return path.read_text().strip()
        return self.system_prompt
