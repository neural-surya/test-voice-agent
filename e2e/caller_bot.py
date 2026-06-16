"""
CallerBot — LLM-driven simulated caller that joins a LiveKit room,
drives the conversation with the SkyWay agent, and tracks goal completion.

Usage (called by test_e2e.py):
    result = await run_e2e_scenario(scenario)
"""
import asyncio
import os
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger("caller-bot")


GOAL_COMPLETE_MARKER = "GOAL_COMPLETE"

BASE_PERSONA = """You are a test caller interacting with SkyWay Airlines' voice assistant.
Your goal is described in the scenario below.

Rules:
- Keep each response short (1–2 sentences, spoken style — no markdown).
- Stay in character.
- When your goal is fully achieved, output EXACTLY this on its own line: GOAL_COMPLETE
- Never output GOAL_COMPLETE unless the agent has confirmed the action.
- Max conversation turns: 10. If goal not met, say "I'll call back later" and stop.
"""


@dataclass
class CallerBot:
    scenario: dict
    history: list = field(default_factory=list)
    turn_count: int = 0
    goal_completed: bool = False
    _oai: Optional[AsyncOpenAI] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._oai = AsyncOpenAI(timeout=15.0)

    async def next_utterance(self, agent_said: str) -> str:
        """Generate the caller's next utterance given what the agent just said."""
        self.history.append({"role": "user", "content": f"Agent: {agent_said}"})

        persona_block = (
            f"Scenario: {self.scenario['name']}\n"
            f"Persona:  {self.scenario['persona']}\n"
            f"Goal:     {self.scenario['goal_description']}\n"
        )

        resp = await self._oai.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": BASE_PERSONA + "\n" + persona_block},
                *self.history,
            ],
        )
        caller_turn = resp.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": caller_turn})
        self.turn_count += 1

        if GOAL_COMPLETE_MARKER in caller_turn:
            self.goal_completed = True
            caller_turn = caller_turn.replace(GOAL_COMPLETE_MARKER, "").strip()

        logger.info("[turn %d] caller: %s", self.turn_count, caller_turn[:80])
        return caller_turn


async def run_e2e_scenario(scenario: dict) -> dict:
    """
    Run one end-to-end scenario:
      1. Connect a CallerBot to the LiveKit room.
      2. Loop: receive agent audio → transcribe → generate caller response → send audio.
      3. Return result dict with turns, goal_completed, and full transcript.
    """
    from livekit import rtc
    from e2e.token_utils import generate_test_token
    from e2e.audio_utils import (
        synthesize_text_to_pcm,
        create_audio_source_and_track,
        push_audio_to_room,
        collect_agent_audio,
        transcribe_pcm,
    )

    livekit_url = os.environ["LIVEKIT_URL"]
    # Suffix with a unique id so every invocation gets a brand-new room — LiveKit's
    # automatic agent dispatch fires on room *creation*, not on participant join, so
    # reusing a room name across test functions in the same session can leave a
    # caller alone in a room nobody re-dispatches an agent into.
    room_name = f"{scenario['room']}-{uuid.uuid4().hex[:8]}"
    identity = f"test-caller-{scenario['name']}"

    token = generate_test_token(room_name=room_name, identity=identity)

    room = rtc.Room()
    await room.connect(livekit_url, token)
    logger.info("CallerBot connected to room: %s", room_name)

    # Set up caller's audio source
    audio_source, audio_track = await create_audio_source_and_track()
    await room.local_participant.publish_track(
        audio_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    bot = CallerBot(scenario=scenario)
    transcript = []

    try:
        # Give the agent a moment to join
        await asyncio.sleep(2.0)

        while not bot.goal_completed and bot.turn_count < 10:
            # 1. Collect agent speech
            try:
                agent_pcm = await collect_agent_audio(room, timeout_s=15.0)
            except TimeoutError:
                logger.warning("Timeout waiting for agent audio on turn %d", bot.turn_count + 1)
                break

            agent_text = await transcribe_pcm(agent_pcm)
            if not agent_text:
                logger.warning("Empty transcript from agent, skipping turn")
                continue

            transcript.append({"role": "agent", "text": agent_text})
            logger.info("[turn %d] agent: %s", bot.turn_count + 1, agent_text[:80])

            # 2. Generate caller response
            caller_text = await bot.next_utterance(agent_text)
            transcript.append({"role": "caller", "text": caller_text})

            if bot.goal_completed:
                break

            # 3. Send caller audio to room
            caller_pcm = await synthesize_text_to_pcm(caller_text)
            await push_audio_to_room(room, audio_source, caller_pcm)

            # Brief pause to let the agent process
            await asyncio.sleep(0.5)

    finally:
        await room.disconnect()
        logger.info(
            "Scenario '%s' done: goal=%s turns=%d",
            scenario["name"],
            bot.goal_completed,
            bot.turn_count,
        )

    return {
        "scenario": scenario["name"],
        "turns": bot.turn_count,
        "goal_completed": bot.goal_completed,
        "transcript": transcript,
    }
