import asyncio
import base64
import json
import os
import time
from typing import AsyncIterator, Optional, Any

from fastapi import FastAPI, WebSocket
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableGenerator
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
import websockets
from websockets.client import WebSocketClientProtocol

# 1. Event Definitions


class VoiceAgentEvent:
    def __init__(self, type: str, **kwargs: Any) -> None:
        self.type = type
        for key, value in kwargs.items():
            setattr(self, key, value)


class STTOutputEvent(VoiceAgentEvent):
    @classmethod
    def create(cls, transcript: str) -> "STTOutputEvent":
        return cls("stt_output", transcript=transcript)


class STTChunkEvent(VoiceAgentEvent):
    @classmethod
    def create(cls, transcript: str) -> "STTChunkEvent":
        return cls("stt_chunk", transcript=transcript)


class AgentChunkEvent(VoiceAgentEvent):
    @classmethod
    def create(cls, text: str) -> "AgentChunkEvent":
        return cls("agent_chunk", text=text)


class TTSChunkEvent(VoiceAgentEvent):
    @classmethod
    def create(cls, audio: bytes) -> "TTSChunkEvent":
        return cls("tts_chunk", audio=audio)


# 2. Helper Functions


async def merge_async_iters(
    *iters: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """Merge multiple async iterators into a single stream concurrently."""
    queue: asyncio.Queue[Optional[VoiceAgentEvent]] = asyncio.Queue()
    active_tasks = len(iters)

    async def worker(iterator: AsyncIterator[VoiceAgentEvent]) -> None:
        nonlocal active_tasks
        try:
            async for item in iterator:
                await queue.put(item)
        finally:
            active_tasks -= 1
            if active_tasks == 0:
                await queue.put(None)

    for it in iters:
        asyncio.create_task(worker(it))

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item


# 3. Speech-to-Text Client & Stream


class AssemblyAISTT:
    def __init__(
        self, api_key: Optional[str] = None, sample_rate: int = 16000
    ) -> None:
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        self.sample_rate = sample_rate
        self._ws: Optional[WebSocketClientProtocol] = None

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send PCM audio bytes to AssemblyAI."""
        ws = await self._ensure_connection()
        await ws.send(audio_chunk)

    async def receive_events(self) -> AsyncIterator[VoiceAgentEvent]:
        """Yield STT events as they arrive from AssemblyAI."""
        ws = await self._ensure_connection()
        async for raw_message in ws:
            message = json.loads(raw_message)
            if message.get("type") == "Turn":
                if message.get("turn_is_formatted"):
                    yield STTOutputEvent.create(message.get("transcript", ""))
                else:
                    yield STTChunkEvent.create(message.get("transcript", ""))

    async def _ensure_connection(self) -> WebSocketClientProtocol:
        if self._ws is None:
            url = f"wss://streaming.assemblyai.com/v3/ws?sample_rate={self.sample_rate}&format_turns=true"
            headers = {"Authorization": self.api_key} if self.api_key else {}
            self._ws = await websockets.connect(url, additional_headers=headers)
        return self._ws

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None


async def stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """Transform stream: Audio (Bytes) -> Voice Events (VoiceAgentEvent)."""
    stt = AssemblyAISTT(sample_rate=16000)

    async def send_audio() -> None:
        try:
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            await stt.close()

    send_task = asyncio.create_task(send_audio())
    try:
        async for event in stt.receive_events():
            yield event
    finally:
        send_task.cancel()
        await stt.close()


# 4. Agent Tools & Stream Configuration


def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."


def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."


agent = create_agent(
    model="google_genai:gemini-3.6-flash",
    tools=[add_to_order, confirm_order],
    system_prompt=(
        "You are a helpful sandwich shop assistant. Your goal is to take the user's order. "
        "Be concise and friendly. Do NOT use emojis, special characters, or markdown. "
        "Your responses will be read by a text-to-speech engine."
    ),
    checkpointer=InMemorySaver(),
)


async def agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """Transform stream: Voice Events -> Voice Events (with Agent Responses)."""
    thread_id = str(uuid7())
    async for event in event_stream:
        yield event
        if event.type == "stt_output":
            stream = await agent.astream_events(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                version="v3",
            )
            async for message in stream.messages:
                async for token in message.text:
                    yield AgentChunkEvent.create(token)


# 5. Text-to-Speech Client & Stream


class CartesiaTTS:
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "f6ff7c0c-e396-40a9-a70b-f7607edb6937",
        model_id: str = "sonic-3",
        sample_rate: int = 24000,
        encoding: str = "pcm_s16le",
        cartesia_version: str = "2024-06-10",
        language: str = "en",
    ) -> None:
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.cartesia_version = cartesia_version
        self.language = language
        self._ws: Optional[WebSocketClientProtocol] = None
        self._context_counter = 0

    def _generate_context_id(self) -> str:
        timestamp = int(time.time() * 1000)
        counter = self._context_counter
        self._context_counter += 1
        return f"ctx_{timestamp}_{counter}"

    async def send_text(self, text: Optional[str]) -> None:
        """Send text to Cartesia for synthesis."""
        if not text or not text.strip():
            return
        ws = await self._ensure_connection()
        payload = {
            "model_id": self.model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self.voice_id,
            },
            "output_format": {
                "container": "raw",
                "encoding": self.encoding,
                "sample_rate": self.sample_rate,
            },
            "language": self.language,
            "context_id": self._generate_context_id(),
        }
        await ws.send(json.dumps(payload))

    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
        """Yield audio chunks as they arrive from Cartesia."""
        ws = await self._ensure_connection()
        async for raw_message in ws:
            message = json.loads(raw_message)
            if "data" in message and message["data"]:
                audio_chunk = base64.b64decode(message["data"])
                if audio_chunk:
                    yield TTSChunkEvent.create(audio_chunk)

    async def _ensure_connection(self) -> WebSocketClientProtocol:
        if self._ws is None:
            url = (
                f"wss://api.cartesia.ai/tts/websocket"
                f"?api_key={self.api_key}&cartesia_version={self.cartesia_version}"
            )
            self._ws = await websockets.connect(url)
        return self._ws

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None


async def tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """Transform stream: Voice Events -> Voice Events (with Audio)."""
    tts = CartesiaTTS()

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        async for event in event_stream:
            yield event
            if event.type == "agent_chunk":
                await tts.send_text(event.text)

    try:
        async for event in merge_async_iters(
            process_upstream(), tts.receive_events()
        ):
            yield event
    finally:
        await tts.close()


# 6. Pipeline Assembly & FastAPI Application Setup

pipeline = (
    RunnableGenerator(stt_stream)
    | RunnableGenerator(agent_stream)
    | RunnableGenerator(tts_stream)
)

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        while True:
            try:
                data = await websocket.receive_bytes()
                yield data
            except Exception:
                break

    output_stream = pipeline.atransform(websocket_audio_stream())

    async for event in output_stream:
        if event.type == "tts_chunk":
            await websocket.send_bytes(event.audio)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)