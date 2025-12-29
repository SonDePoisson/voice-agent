"""VoiceAgent - Main orchestrator for voice-based LLM interactions."""

from typing import Generator, Callable

from .types import AudioData, ConversationMessage
from .streaming import stream_sentences
from ..stt.stt import STTBackend
from ..tts.tts import TTSBackend
from ..llm.llm import LLMBackend
from ..config import AgentConfig


class VoiceAgent:
    """Main orchestrator for voice-based LLM interactions.

    Combines STT, LLM, and TTS backends to create a complete voice agent
    that can process audio input and generate audio responses.
    """

    def __init__(
        self,
        stt: STTBackend,
        tts: TTSBackend,
        llm: LLMBackend,
        config: AgentConfig | None = None,
    ):
        """Initialize the voice agent.

        Args:
            stt: Speech-to-Text backend
            tts: Text-to-Speech backend
            llm: LLM backend
            config: Agent configuration
        """
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.config = config or AgentConfig()
        self.conversation_history: list[ConversationMessage] = []
        self._system_prompt = self.config.get_system_prompt()

    def process_audio(self, audio: AudioData) -> Generator[AudioData, None, None]:
        """Process incoming audio and yield response audio.

        This is the main entry point for voice interactions.

        Args:
            audio: Tuple of (sample_rate, audio_array)

        Yields:
            Audio chunks as (sample_rate, audio_array) tuples
        """
        sample_rate, audio_data = audio

        # STT: Convert speech to text
        user_text = self.stt.transcribe(audio_data, sample_rate)
        print(f"User: {user_text}")

        self.conversation_history.append(ConversationMessage(role="user", content=user_text))

        # LLM: Generate response with streaming
        text_stream = self.llm.stream_generate(
            prompt=user_text,
            system_prompt=self._system_prompt,
        )

        # TTS: Convert response to speech, sentence by sentence
        print("Assistant: ", end="", flush=True)
        full_response = ""

        for sentence in stream_sentences(text_stream):
            print(sentence, end=" ", flush=True)
            full_response += sentence + " "

            # Generate TTS for this sentence immediately
            for audio_chunk in self.tts.stream_synthesize(sentence):
                yield audio_chunk

        print()  # Newline after complete response

        self.conversation_history.append(ConversationMessage(role="assistant", content=full_response.strip()))

    def create_fastrtc_handler(
        self,
    ) -> Callable[[AudioData], Generator[AudioData, None, None]]:
        """Create a handler function for fastrtc ReplyOnPause.

        Returns:
            Handler function compatible with fastrtc
        """

        def handler(audio: AudioData) -> Generator[AudioData, None, None]:
            yield from self.process_audio(audio)

        return handler

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
