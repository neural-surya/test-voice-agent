"""
Shared pytest fixtures and configuration.
Loaded automatically by pytest before any test file.
"""
import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_audio: requires pre-built corpus audio files")
    config.addinivalue_line("markers", "requires_livekit: requires live LiveKit credentials")
    config.addinivalue_line("markers", "slow: calls external APIs, may be slow")


def pytest_collection_modifyitems(config, items):
    """Skip tests that need audio files when corpus is not built."""
    corpus_ready = (Path(__file__).parent / "corpus" / "manifest.json").exists()
    livekit_ready = bool(
        os.environ.get("LIVEKIT_URL")
        and os.environ.get("LIVEKIT_API_KEY")
        and os.environ.get("LIVEKIT_API_SECRET")
    )

    for item in items:
        if "requires_audio" in item.keywords and not corpus_ready:
            item.add_marker(
                pytest.mark.skip(reason="Run corpus_builder.py first to generate audio files")
            )
        if "requires_livekit" in item.keywords and not livekit_ready:
            item.add_marker(
                pytest.mark.skip(reason="Set LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET in .env")
            )


@pytest.fixture(scope="session")
def kb_context() -> str:
    """Sample knowledge-base context for L4 LLM tests."""
    return (
        "Economy class baggage allowance: 1 checked bag up to 23 kg included. "
        "Carry-on: up to 7 kg. Business class: 2 checked bags up to 32 kg each. "
        "Change fee: $75 for economy, waived for business. "
        "Cancellation: full refund if cancelled 24+ hours before departure. "
        "Meals: economy meals available for purchase; business class meals included. "
        "Check-in: opens 24 hours before departure and closes 1 hour before departure. "
        "SkyRewards loyalty program: earn 1 mile per dollar spent on SkyWay flights; "
        "redeem miles for flight discounts and other rewards."
    )


@pytest.fixture(scope="session")
def openai_client():
    from openai import OpenAI
    return OpenAI()


@pytest.fixture(scope="session")
def system_prompt() -> str:
    return (Path(__file__).parent / "system_prompt.txt").read_text()
