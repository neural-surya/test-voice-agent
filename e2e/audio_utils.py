"""
Audio helpers for E2E tests:
  - synthesize_text_to_pcm  → OpenAI TTS → raw PCM bytes
  - create_audio_source     → LiveKit LocalAudioTrack source
  - push_audio_to_room      → publish PCM frames into a LiveKit room
  - collect_agent_audio     → buffer agent audio until silence, return raw PCM
  - transcribe_pcm          → Deepgram transcription of raw PCM
"""
import asyncio
import os
from typing import Optional

import numpy as np


# ── TTS helper ────────────────────────────────────────────────────────────────

async def synthesize_text_to_pcm(text: str, sample_rate: int = 24000) -> bytes:
    """Call OpenAI TTS and return raw 16-bit PCM mono bytes."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(timeout=15.0)
    response = await client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="pcm",  # 24 kHz, 16-bit, mono
    )
    return response.content


# ── LiveKit audio publishing ───────────────────────────────────────────────────

async def create_audio_source_and_track():
    """Create a LiveKit AudioSource + LocalAudioTrack for publishing."""
    from livekit import rtc

    sample_rate = 24000
    num_channels = 1
    audio_source = rtc.AudioSource(sample_rate=sample_rate, num_channels=num_channels)
    audio_track = rtc.LocalAudioTrack.create_audio_track("caller-mic", audio_source)
    return audio_source, audio_track


async def push_audio_to_room(
    room,
    audio_source,
    pcm_bytes: bytes,
    sample_rate: int = 24000,
    chunk_ms: int = 20,
) -> None:
    """Push raw 16-bit PCM in `chunk_ms` chunks into the LiveKit room."""
    from livekit import rtc

    samples_per_chunk = int(sample_rate * chunk_ms / 1000)
    bytes_per_chunk = samples_per_chunk * 2  # 16-bit = 2 bytes

    for offset in range(0, len(pcm_bytes), bytes_per_chunk):
        chunk = pcm_bytes[offset : offset + bytes_per_chunk]
        if len(chunk) < bytes_per_chunk:
            # Pad last chunk with silence
            chunk = chunk + b"\x00" * (bytes_per_chunk - len(chunk))

        frame = rtc.AudioFrame(
            data=chunk,
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=samples_per_chunk,
        )
        await audio_source.capture_frame(frame)
        await asyncio.sleep(chunk_ms / 1000)


# ── Agent audio collection ─────────────────────────────────────────────────────

async def collect_agent_audio(
    room,
    silence_threshold_ms: int = 800,
    timeout_s: float = 20.0,
    sample_rate: int = 16000,
) -> bytes:
    """
    Wait for the agent's audio track, buffer frames, and return PCM bytes
    once silence is detected for `silence_threshold_ms` milliseconds.
    """
    from livekit import rtc

    agent_track: Optional[rtc.AudioTrack] = None
    deadline = asyncio.get_event_loop().time() + timeout_s

    # Wait for the agent to publish an audio track
    while agent_track is None:
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError("Agent did not publish audio within timeout")
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                    agent_track = pub.track
                    break
        if agent_track is None:
            await asyncio.sleep(0.1)

    audio_stream = rtc.AudioStream(
        agent_track,
        sample_rate=sample_rate,
        num_channels=1,
    )

    frames: list[bytes] = []
    silence_samples = int(sample_rate * silence_threshold_ms / 1000)
    consecutive_silence = 0

    async with asyncio.timeout(timeout_s):
        async for frame_event in audio_stream:
            frame = frame_event.frame
            raw = bytes(frame.data)
            frames.append(raw)

            pcm = np.frombuffer(raw, dtype=np.int16)
            if np.max(np.abs(pcm)) < 150:  # ~0.5% of full scale
                consecutive_silence += len(pcm)
            else:
                consecutive_silence = 0

            if consecutive_silence >= silence_samples and len(frames) > 3:
                break

    return b"".join(frames)


# ── Deepgram transcription ─────────────────────────────────────────────────────

async def transcribe_pcm(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
) -> str:
    """Transcribe raw PCM bytes using Deepgram nova-2 (async, SDK-version-agnostic)."""
    import io
    import json as _json
    import urllib.request
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_bytes = buf.getvalue()

    api_key = os.environ["DEEPGRAM_API_KEY"]
    url = "https://api.deepgram.com/v1/listen?model=nova-2&language=en&punctuate=true"
    req = urllib.request.Request(
        url,
        data=wav_bytes,
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
        method="POST",
    )

    def _do_request():
        import ssl
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            return _json.loads(resp.read())

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _do_request)
    return result["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
