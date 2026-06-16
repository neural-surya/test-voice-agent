"""
SkyWay Airlines voice agent — system under test.
Pipeline: VAD (Silero) → STT (Deepgram nova-2) → LLM (GPT-4o) → TTS (Cartesia)
"""
import os
import logging
from pathlib import Path

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
load_dotenv()

from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli
from livekit.plugins import openai, deepgram, cartesia, silero

logger = logging.getLogger("skyway-agent")

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text()


def prewarm(proc: JobProcess):
    # Load the VAD model once per worker process instead of per job — loading it
    # cold inside entrypoint() made the first job after worker startup slow enough
    # to miss callers' turn-1 audio timeout.
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    logger.info("Participant joined: %s", participant.identity)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2", language="en"),
        llm=openai.LLM(model="gpt-4o"),
        tts=cartesia.TTS(
            model="sonic-3",
            voice=os.environ.get("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091"),
        ),
    )

    agent = Agent(instructions=SYSTEM_PROMPT)
    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(
        instructions="Greet the caller warmly as Aria from SkyWay Airlines and ask how you can help."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
