# fastrtc-voice-agent

A modular voice agent built on [FastRTC](https://github.com/gradio-app/fastrtc)

## Installation

```bash
pip install fastrtc-voice-agent
```

## Example

```python
from fastrtc import ReplyOnPause, Stream
from voice_agent import create_agent, AgentConfig, STTConfig, TTSConfig, LLMConfig

config = AgentConfig(
    system_prompt="You are a helpful voice assistant.",
    stt=STTConfig(backend="faster_whisper", model_size="small"),
    tts=TTSConfig(backend="edge", voice="en-US-AvaMultilingualNeural"),
    llm=LLMConfig(backend="ollama", model="llama3.2:3b"),
)

agent = create_agent(config)

stream = Stream(
    ReplyOnPause(agent.create_fastrtc_handler()),
    modality="audio",
    mode="send-receive",
)

stream.ui.launch()
```
