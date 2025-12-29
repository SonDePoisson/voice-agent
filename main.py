from fastrtc import ReplyOnPause, Stream
import numpy as np
from numpy.typing import NDArray
from whisper import load_model, load_audio, Whisper
from faster_whisper import WhisperModel
import ollama
import librosa
import edge_tts
import asyncio
import io
import re
from pydub import AudioSegment
from typing import Generator, AsyncGenerator
from dataclasses import dataclass

STT_MODEL = "small"
OLLAMA_MODEL = "llama3.2:3b"
TTS_VOICE = "en-US-AvaMultilingualNeural"
SYSTEM_PROMPT_FILE = "system_prompt.txt"

# Load system prompt
with open(SYSTEM_PROMPT_FILE, "r") as f:
    SYSTEM_PROMPT = f.read().strip()


class WhisperSTTModel:
    def __init__(self):
        self.model: Whisper = load_model(name=STT_MODEL, device="cpu")

    def stt(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        # Convert int16 to float32 and normalize to [-1, 1]
        audio = audio.astype(np.float32).flatten() / 32768.0
        # Resample to 16kHz (Whisper's expected sample rate)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        result = self.model.transcribe(audio, fp16=False)
        return result["text"]

    def load_audio(self, path: str) -> np.ndarray:
        return load_audio(path)


class FasterWhisperSTTModel:
    """Fast STT using faster_whisper"""

    def __init__(self, model_size: str = STT_MODEL, device: str = "auto"):
        # device: "cpu" or "cuda"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(self.model)

    def stt(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        # Convert to float32 and normalize to [-1, 1]
        audio = audio.astype(np.float32).flatten() / 32768.0

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # Transcribe
        segments, _ = self.model.transcribe(audio, beam_size=5)
        return "".join(segment.text for segment in segments)


@dataclass
class EdgeTTSOptions:
    voice: str = TTS_VOICE
    rate: str = "+0%"
    pitch: str = "+0Hz"


class EdgeTTSModel:
    SAMPLE_RATE = 24000

    def __init__(self, voice: str = TTS_VOICE):
        self.voice = voice

    def _decode_mp3(self, mp3_bytes: bytes) -> NDArray[np.float32]:
        """Decode MP3 bytes to numpy array at 24kHz mono float32."""
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        audio = audio.set_frame_rate(self.SAMPLE_RATE).set_channels(1)
        return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

    async def _generate_sentence(self, text: str, options: EdgeTTSOptions) -> bytes:
        """Generate audio bytes for a single sentence."""
        communicate = edge_tts.Communicate(text, options.voice, rate=options.rate, pitch=options.pitch)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        return audio_bytes

    def tts(self, text: str, options: EdgeTTSOptions | None = None) -> tuple[int, NDArray[np.float32]]:
        """Generate complete audio from text."""
        options = options or EdgeTTSOptions(voice=self.voice)
        loop = asyncio.new_event_loop()
        try:
            audio_bytes = loop.run_until_complete(self._generate_sentence(text, options))
            return self.SAMPLE_RATE, self._decode_mp3(audio_bytes)
        finally:
            loop.close()

    async def stream_tts(
        self, text: str, options: EdgeTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        """Async generator yielding audio chunks per sentence for lower latency."""
        options = options or EdgeTTSOptions(voice=self.voice)

        # Split by sentences for faster first-chunk delivery
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Generate and decode audio for this sentence
            audio_bytes = await self._generate_sentence(sentence, options)
            audio = self._decode_mp3(audio_bytes)

            # Yield in chunks
            chunk_size = self.SAMPLE_RATE // 5  # 200ms chunks
            for i in range(0, len(audio), chunk_size):
                yield self.SAMPLE_RATE, audio[i : i + chunk_size]

    def stream_tts_sync(
        self, text: str, options: EdgeTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        """Sync generator yielding audio chunks."""
        loop = asyncio.new_event_loop()
        iterator = self.stream_tts(text, options).__aiter__()
        try:
            while True:
                try:
                    yield loop.run_until_complete(iterator.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()


def stream_sentences(text_generator) -> Generator[str, None, None]:
    """Buffer streamed tokens and yield complete sentences."""
    buffer = ""
    for chunk in text_generator:
        buffer += chunk.response
        # Check for sentence endings
        while True:
            match = re.search(r"[.!?]\s*", buffer)
            if match:
                sentence = buffer[: match.end()].strip()
                buffer = buffer[match.end() :]
                if sentence:
                    yield sentence
            else:
                break
    # Yield remaining text
    if buffer.strip():
        yield buffer.strip()


def response(audio: tuple[int, np.ndarray]):
    sample_rate, audio_data = audio
    user_prompt = stt_model.stt(audio_data, sample_rate)
    print(f"User: {user_prompt}")

    # Stream Ollama response and TTS sentence by sentence
    text_stream = ollama.generate(
        model=OLLAMA_MODEL,
        system=SYSTEM_PROMPT,
        prompt=user_prompt,
        stream=True,
    )

    print("Assistant: ", end="", flush=True)
    for sentence in stream_sentences(text_stream):
        print(sentence, end=" ", flush=True)
        # Generate TTS for this sentence immediately
        for audio_chunk in tts_model.stream_tts_sync(sentence):
            yield audio_chunk
    print()  # Newline after complete response


stt_model = FasterWhisperSTTModel()
tts_model = EdgeTTSModel()

stream = Stream(ReplyOnPause(response), modality="audio", mode="send-receive")


if __name__ == "__main__":
    stream.ui.launch()
