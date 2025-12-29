"""TTS module for voice_agent."""

from .tts import (
    TTSBackend,
    TTSOptions,
    EdgeTTSBackend,
    create_tts_backend,
)

__all__ = [
    "TTSBackend",
    "TTSOptions",
    "EdgeTTSBackend",
    "create_tts_backend",
]
