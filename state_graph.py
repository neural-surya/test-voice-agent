"""
LLM Sandwich state-transition graph for the SkyWay Airlines voice agent.
Defines valid states, transitions, required tools, and guardrails at each node.
"""
import json
import os
from openai import OpenAI

STATE_GRAPH: dict[str, dict] = {
    "GREETING": {
        "transitions":    ["INTENT_CAPTURE", "ESCALATE_HUMAN"],
        "required_tools": [],
        "guardrails":     ["no_pii_in_greeting"],
        "description":    "Initial welcome state",
    },
    "INTENT_CAPTURE": {
        "transitions":    ["SLOT_FILL", "OUT_OF_SCOPE", "ESCALATE_HUMAN"],
        "required_tools": [],
        "guardrails":     ["reject_harmful_intent", "reject_jailbreak"],
        "description":    "Identify what the caller wants",
    },
    "SLOT_FILL": {
        "transitions":    ["TOOL_CALL", "CONFIRM", "INTENT_CAPTURE"],
        "required_tools": [],
        "guardrails":     [],
        "description":    "Collect missing details (dates, destinations, etc.)",
    },
    "CONFIRM": {
        "transitions":    ["TOOL_CALL", "SLOT_FILL"],
        "required_tools": [],
        "guardrails":     [],
        "description":    "Confirm collected details with the caller",
    },
    "TOOL_CALL": {
        "transitions":    ["RESPONSE_GEN", "ESCALATE_HUMAN"],
        "required_tools": ["search_kb", "book_action"],
        "guardrails":     ["tool_output_validation"],
        "description":    "Execute backend action or KB lookup",
    },
    "RESPONSE_GEN": {
        "transitions":    ["GREETING", "INTENT_CAPTURE"],
        "required_tools": [],
        "guardrails":     ["no_hallucination", "voice_conciseness"],
        "description":    "Generate and speak the final response",
    },
    "OUT_OF_SCOPE": {
        "transitions":    ["INTENT_CAPTURE"],
        "required_tools": [],
        "guardrails":     [],
        "description":    "Politely redirect off-topic queries",
    },
    "ESCALATE_HUMAN": {
        "transitions":    [],
        "required_tools": ["transfer_to_human"],
        "guardrails":     [],
        "description":    "Transfer to human agent",
    },
}

_TRANSITION_CACHE: dict[tuple, str] = {}


def get_next_state(current_state: str, utterance: str, use_cache: bool = True) -> str:
    """Use an LLM to pick the next valid state given the utterance."""
    key = (current_state, utterance.strip().lower())
    if use_cache and key in _TRANSITION_CACHE:
        return _TRANSITION_CACHE[key]

    if current_state not in STATE_GRAPH:
        raise ValueError(f"Unknown state: {current_state}")

    valid = STATE_GRAPH[current_state]["transitions"]
    if not valid:
        return current_state  # terminal state

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a dialog state machine for a flight-booking voice agent.\n"
                    f"Current state: {current_state} — {STATE_GRAPH[current_state]['description']}\n"
                    f"Valid next states: {valid}\n"
                    "Based solely on the caller utterance, pick the most appropriate next state.\n"
                    'Return JSON: {"next_state": "<state>"}'
                ),
            },
            {"role": "user", "content": utterance},
        ],
    )
    result = json.loads(resp.choices[0].message.content)
    next_state = result["next_state"]

    if next_state not in valid:
        next_state = valid[0]

    if use_cache:
        _TRANSITION_CACHE[key] = next_state
    return next_state


# ── Simulated tool registry ────────────────────────────────────────────────────

def search_kb(query: str) -> dict:
    """Mock KB search — replace with real retrieval in production."""
    KB = {
        "baggage": "Economy: 1 checked bag up to 23 kg. Business: 2 bags up to 32 kg each.",
        "cancel": "Full refund if cancelled 24+ hours before departure. Change fee $75 for economy.",
        "checkin": "Check-in opens 24 hours before departure, closes 1 hour before.",
        "loyalty": "SkyRewards program: 1 mile per dollar spent.",
        "meal": "Economy meals available for purchase. Business class includes meals.",
    }
    q = query.lower()
    for key, val in KB.items():
        if key in q:
            return {"found": True, "content": val, "source": f"kb/{key}"}
    return {"found": False, "content": "", "source": ""}


def book_action(destination: str, date: str, passenger: str) -> dict:
    """Mock booking action — replace with real API in production."""
    import random, string
    ref = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return {
        "success": True,
        "booking_reference": ref,
        "destination": destination,
        "date": date,
        "passenger": passenger,
        "message": f"Booking confirmed. Your reference is {ref}.",
    }


def transfer_to_human() -> dict:
    return {"success": True, "message": "Transferring to human agent."}
