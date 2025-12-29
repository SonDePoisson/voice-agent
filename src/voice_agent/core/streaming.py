"""Streaming utilities for voice_agent."""

import re
from typing import Generator, Iterator


def stream_sentences(text_generator: Iterator) -> Generator[str, None, None]:
    """Buffer streamed tokens and yield complete sentences.

    Args:
        text_generator: Iterator yielding chunks with .response attribute

    Yields:
        Complete sentences as they become available
    """
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
