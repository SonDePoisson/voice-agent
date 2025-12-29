"""Configuration for voice_agent."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text."""

    backend: str = "faster_whisper"
    model_size: str = "small"
    device: str = "cpu"


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech."""

    backend: str = "edge"
    voice: str = "en-US-AvaMultilingualNeural"


@dataclass
class LLMConfig:
    """Configuration for LLM."""

    backend: str = "ollama"
    model: str = "llama3.2:3b"


@dataclass
class AgentConfig:
    """Main configuration for VoiceAgent."""

    system_prompt: str = "You are a helpful voice assistant."
    system_prompt_file: str | None = None

    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def get_system_prompt(self) -> str:
        """Load system prompt from file if specified, otherwise return default."""
        if self.system_prompt_file:
            path = Path(self.system_prompt_file)
            if path.exists():
                return path.read_text().strip()
        return self.system_prompt
