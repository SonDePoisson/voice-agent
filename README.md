# fastrtc-voice-agent

A modular voice agent built on [FastRTC](https://github.com/gradio-app/fastrtc)

## Installation

### Recommended: Using uv

[uv](https://github.com/astral-sh/uv) is the recommended way to manage your Python environment and dependencies.

```bash
# Create a virtual environment with Python 3.12+
uv venv --python 3.12

# Activate the environment
source .venv/bin/activate

# Install the package
uv add fastrtc-voice-agent

# Install with optional dependencies (e.g., ollama)
uv add "fastrtc-voice-agent[ollama]"

# Or install all optional dependencies
uv add "fastrtc-voice-agent[all]"
```

### Using pip

```bash
pip install fastrtc-voice-agent

# Install your desired STT and LLM with (for example):
pip install "fastrtc-voice-agent[ollama]"

# Or for all optional dependencies:
pip install "fastrtc-voice-agent[all]"
```

## CLI Usage Example

For default config :

```bash
fastrtc-voice-agent --run
```

Please refere to the help for custom config :

```bash
fastrtc-voice-agent --help
```

## Python Usage Example

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

## Custom Frontend Integration

If you want to use your own frontend (React, Vue, etc.) instead of the built-in Gradio UI, you can run the agent as an API server.

### CLI - API Mode

```bash
# Install with API support
pip install "fastrtc-voice-agent[api]"

# Run as API server (no Gradio UI)
fastrtc-voice-agent --run --api --port 8000
```

This exposes WebRTC endpoints:

- `POST /webrtc/offer` - WebRTC signaling
- `WS /websocket/offer` - WebSocket alternative

### Python - API Server

```python
from voice_agent import create_api_server, AgentConfig, STTConfig, TTSConfig, LLMConfig

# Create a FastAPI app with the voice agent
app = create_api_server(
    config=AgentConfig(
        system_prompt="You are a helpful assistant.",
        stt=STTConfig(backend="faster_whisper"),
        tts=TTSConfig(backend="edge"),
        llm=LLMConfig(backend="ollama"),
    )
)

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

You can also mount it in an existing FastAPI app:

```python
from fastapi import FastAPI
from voice_agent import create_api_server

main_app = FastAPI()
voice_app = create_api_server()
main_app.mount("/voice", voice_app)
```

### React Example

See the [examples/react-client](examples/react-client) directory for a complete React example with a `useVoiceAgent` hook.

Quick example:

```tsx
import { useVoiceAgent } from "./useVoiceAgent";

function App() {
  const { isConnected, connect, disconnect } = useVoiceAgent({
    serverUrl: "http://localhost:8000",
  });

  return (
    <button onClick={isConnected ? disconnect : connect}>
      {isConnected ? "Stop" : "Start"}
    </button>
  );
}
```

## Note

To use Anthropic API (may be OpenAI or else later) please copy .env.example as .env file and fill it with your API KEY and the desired model
