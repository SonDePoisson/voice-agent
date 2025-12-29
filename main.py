"""Main entry point for the voice agent application."""

from fastrtc import ReplyOnPause, Stream

from src.voice_agent import create_agent, AgentConfig, STTConfig, TTSConfig, LLMConfig


# Configuration
config = AgentConfig(
    system_prompt_file="system_prompt.txt",
    stt=STTConfig(
        backend="faster_whisper",
        model_size="small",
        device="cpu",
    ),
    tts=TTSConfig(
        backend="edge",
        voice="en-US-AvaMultilingualNeural",
    ),
    llm=LLMConfig(
        backend="ollama",
        model="llama3.2:3b",
    ),
)

# Create the agent
agent = create_agent(config)

# Create fastrtc stream
stream = Stream(
    ReplyOnPause(agent.create_fastrtc_handler()),
    modality="audio",
    mode="send-receive",
)


if __name__ == "__main__":
    stream.ui.launch()
