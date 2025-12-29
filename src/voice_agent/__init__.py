"""voice_agent - A modular voice agent with swappable STT/TTS/LLM backends."""

from .core import VoiceAgent, AudioData, ConversationMessage
from .config import AgentConfig, STTConfig, TTSConfig, LLMConfig
from .stt import create_stt_backend, STTBackend
from .tts import create_tts_backend, TTSBackend
from .llm import create_llm_backend, LLMBackend


def create_agent(config: AgentConfig | None = None) -> VoiceAgent:
    """Create a voice agent with the specified configuration.

    Args:
        config: Agent configuration. If None, uses defaults.

    Returns:
        Configured VoiceAgent instance
    """
    config = config or AgentConfig()

    stt = create_stt_backend(
        backend=config.stt.backend,
        model_size=config.stt.model_size,
        device=config.stt.device,
    )
    tts = create_tts_backend(
        backend=config.tts.backend,
        voice=config.tts.voice,
    )
    llm = create_llm_backend(
        backend=config.llm.backend,
        model=config.llm.model,
    )

    return VoiceAgent(stt=stt, tts=tts, llm=llm, config=config)


__all__ = [
    # Main classes
    "VoiceAgent",
    "create_agent",
    # Types
    "AudioData",
    "ConversationMessage",
    # Config
    "AgentConfig",
    "STTConfig",
    "TTSConfig",
    "LLMConfig",
    # Backends
    "STTBackend",
    "TTSBackend",
    "LLMBackend",
    "create_stt_backend",
    "create_tts_backend",
    "create_llm_backend",
]
