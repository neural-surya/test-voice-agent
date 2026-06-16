"""
Generate short-lived LiveKit access tokens for E2E test participants.
"""
import os
from datetime import timedelta
from livekit.api import AccessToken, VideoGrants


def generate_test_token(
    room_name: str,
    identity: str,
    ttl_seconds: int = 300,
) -> str:
    """Return a signed JWT token granting room join rights."""
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(timedelta(seconds=ttl_seconds))
        .with_grants(VideoGrants(room_join=True, room=room_name))
    )
    return token.to_jwt()
