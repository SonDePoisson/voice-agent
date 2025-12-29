"""Text-to-Speech backends for voice_agent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, AsyncGenerator
import asyncio
import io

import numpy as np
from numpy.typing import NDArray
import edge_tts
from pydub import AudioSegment
import re


class TTSBackend(ABC):
    """Abstract base class for Text-to-Speech backends."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate."""
        pass

    @abstractmethod
    def synthesize(self, text: str) -> tuple[int, NDArray[np.float32]]:
        """Synthesize complete audio from text.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (sample_rate, audio_array)
        """
        pass

    @abstractmethod
    def stream_synthesize(self, text: str) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        """Stream audio chunks for lower latency.

        Args:
            text: Text to synthesize

        Yields:
            Tuples of (sample_rate, audio_chunk)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        pass


@dataclass
class EdgeTTSOptions:
    """Options for TTS synthesis."""

    voice: str = "en-US-AvaMultilingualNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


class EdgeTTSBackend(TTSBackend):
    """TTS backend using Microsoft Edge TTS."""

    SAMPLE_RATE = 24000

    def __init__(self, voice: str = "en-US-AvaMultilingualNeural"):
        self.voice = voice
        self.options = EdgeTTSOptions(voice=voice)

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def name(self) -> str:
        return "edge-tts"

    def _decode_mp3(self, mp3_bytes: bytes) -> NDArray[np.float32]:
        """Decode MP3 bytes to numpy array at 24kHz mono float32."""
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        audio = audio.set_frame_rate(self.SAMPLE_RATE).set_channels(1)
        return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

    async def _generate_sentence(self, text: str) -> bytes:
        """Generate audio bytes for a single sentence."""
        communicate = edge_tts.Communicate(text, self.options.voice, rate=self.options.rate, pitch=self.options.pitch)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        return audio_bytes

    def synthesize(self, text: str) -> tuple[int, NDArray[np.float32]]:
        """Generate complete audio from text."""
        loop = asyncio.new_event_loop()
        try:
            audio_bytes = loop.run_until_complete(self._generate_sentence(text))
            return self.SAMPLE_RATE, self._decode_mp3(audio_bytes)
        finally:
            loop.close()

    async def _stream_tts_async(self, text: str) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        """Async generator yielding audio chunks per sentence for lower latency."""
        # Split by sentences for faster first-chunk delivery
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Generate and decode audio for this sentence
            audio_bytes = await self._generate_sentence(sentence)
            audio = self._decode_mp3(audio_bytes)

            # Yield in chunks
            chunk_size = self.SAMPLE_RATE // 5  # 200ms chunks
            for i in range(0, len(audio), chunk_size):
                yield self.SAMPLE_RATE, audio[i : i + chunk_size]

    def stream_synthesize(self, text: str) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        """Sync generator yielding audio chunks."""
        loop = asyncio.new_event_loop()
        iterator = self._stream_tts_async(text).__aiter__()
        try:
            while True:
                try:
                    yield loop.run_until_complete(iterator.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()


def create_tts_backend(
    backend: str = "edge",
    voice: str = "en-US-AvaMultilingualNeural",
) -> TTSBackend:
    """Factory function to create a TTS backend.

    Args:
        backend: Backend type ("edge" for now)
        voice: Voice identifier

    Returns:
        Configured TTS backend instance
    """
    if backend == "edge":
        return EdgeTTSBackend(voice=voice)
    else:
        raise ValueError(f"Unknown TTS backend: {backend}")
