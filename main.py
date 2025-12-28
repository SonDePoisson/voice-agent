from fastrtc import ReplyOnPause, Stream
import numpy as np
from whisper import load_model, load_audio, Whisper
import ollama
import librosa
import edge_tts
import asyncio
import io
from pydub import AudioSegment
from typing import Generator, AsyncGenerator

STT_MODEL = "small"
OLLAMA_MODEL = "ministral-3"
TTS_VOICE = "en-US-AvaMultilingualNeural"


class STTModel:
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


class TTSModel:
    SAMPLE_RATE = 24000
    CHUNK_SIZE = 4800  # 200ms chunks at 24kHz

    def __init__(self, voice: str = TTS_VOICE):
        self.voice = voice

    def _decode_mp3(self, mp3_bytes: bytes) -> np.ndarray:
        """Decode MP3 bytes to numpy array at 24kHz mono."""
        audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        audio = audio.set_frame_rate(self.SAMPLE_RATE).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
        return samples

    def tts(self, text: str) -> tuple[int, np.ndarray]:
        """Generate complete audio from text."""

        async def _generate():
            communicate = edge_tts.Communicate(text, self.voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            return audio_bytes

        mp3_bytes = asyncio.run(_generate())
        audio = self._decode_mp3(mp3_bytes)
        return (self.SAMPLE_RATE, audio)

    async def stream_tts(self, text: str) -> AsyncGenerator[tuple[int, np.ndarray], None]:
        """Async generator yielding audio chunks."""
        communicate = edge_tts.Communicate(text, self.voice)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        audio = self._decode_mp3(audio_bytes)
        for i in range(0, len(audio), self.CHUNK_SIZE):
            yield (self.SAMPLE_RATE, audio[i : i + self.CHUNK_SIZE])

    def stream_tts_sync(self, text: str) -> Generator[tuple[int, np.ndarray], None, None]:
        """Sync generator yielding audio chunks."""

        async def _collect():
            communicate = edge_tts.Communicate(text, self.voice)
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            return audio_bytes

        mp3_bytes = asyncio.run(_collect())
        audio = self._decode_mp3(mp3_bytes)
        for i in range(0, len(audio), self.CHUNK_SIZE):
            yield (self.SAMPLE_RATE, audio[i : i + self.CHUNK_SIZE])


def response(audio: tuple[int, np.ndarray]):
    sample_rate, audio_data = audio
    user_prompt = stt_model.stt(audio_data, sample_rate)
    print(f"User: {user_prompt}")
    ollama_response = ollama.generate(model=OLLAMA_MODEL, prompt=user_prompt)
    print(f"Assistant: {ollama_response.response}")
    for audio_chunk in tts_model.stream_tts_sync(ollama_response.response):
        yield audio_chunk


stt_model = STTModel()
tts_model = TTSModel()

stream = Stream(ReplyOnPause(response), modality="audio", mode="send-receive")


if __name__ == "__main__":
    stream.ui.launch()
