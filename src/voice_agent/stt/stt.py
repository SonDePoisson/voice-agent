"""Speech-to-Text backends for voice_agent."""

from abc import ABC, abstractmethod
import numpy as np
import librosa


class STTBackend(ABC):
    """Abstract base class for Speech-to-Text backends."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (int16 expected)
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        pass

    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Common audio preprocessing: convert to float32, normalize, resample to 16kHz."""
        audio = audio.astype(np.float32).flatten() / 32768.0
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        return audio


class WhisperSTTBackend(STTBackend):
    """STT backend using OpenAI Whisper."""

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        from whisper import load_model, Whisper

        self.model: Whisper = load_model(name=model_size, device=device)
        self._model_size = model_size

    @property
    def name(self) -> str:
        return f"whisper-{self._model_size}"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        audio = self._preprocess_audio(audio, sample_rate)
        result = self.model.transcribe(audio, fp16=False)
        return result["text"]


class FasterWhisperSTTBackend(STTBackend):
    """STT backend using faster_whisper (CTranslate2 based, ~4x faster)."""

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._model_size = model_size

    @property
    def name(self) -> str:
        return f"faster-whisper-{self._model_size}"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 48000) -> str:
        audio = self._preprocess_audio(audio, sample_rate)
        segments, _ = self.model.transcribe(audio, beam_size=5)
        return "".join(segment.text for segment in segments)


def create_stt_backend(
    backend: str = "faster_whisper",
    model_size: str = "small",
    device: str = "cpu",
) -> STTBackend:
    """Factory function to create an STT backend.

    Args:
        backend: Backend type ("whisper" or "faster_whisper")
        model_size: Model size (e.g., "tiny", "small", "medium", "large")
        device: Device to run on ("cpu" or "cuda")

    Returns:
        Configured STT backend instance
    """
    if backend == "whisper":
        return WhisperSTTBackend(model_size=model_size, device=device)
    elif backend == "faster_whisper":
        return FasterWhisperSTTBackend(model_size=model_size, device=device)
    else:
        raise ValueError(f"Unknown STT backend: {backend}")
