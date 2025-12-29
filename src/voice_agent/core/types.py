"""Shared type definitions for voice_agent."""

from typing import TypeAlias, Generator, AsyncGenerator
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

# Audio data type: (sample_rate, audio_array)
AudioData: TypeAlias = tuple[int, NDArray[np.float32]]
AudioStream: TypeAlias = Generator[AudioData, None, None]
AsyncAudioStream: TypeAlias = AsyncGenerator[AudioData, None]


@dataclass
class ConversationMessage:
    """A message in a conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str
