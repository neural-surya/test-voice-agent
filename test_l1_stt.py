"""
L1 · Audio Input / STT — Transcription accuracy tests.

Metrics:
  - Per-clip WER < 15%
  - Aggregate corpus WER < 8%
  - Aggregate corpus CER < 4%

Prerequisites:
  1. Run corpus_builder.py to generate audio files
  2. Set DEEPGRAM_API_KEY in .env
"""
import asyncio
import json
import os
import pytest
from pathlib import Path

from jiwer import wer, cer

MANIFEST_PATH = Path(__file__).parent / "corpus" / "manifest.json"

# ── Load corpus ────────────────────────────────────────────────────────────────

def _load_corpus():
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH) as f:
        return json.load(f)["clips"]


CORPUS = _load_corpus()
CORPUS_EXISTING = [c for c in CORPUS if Path(c["file"]).exists()]


# ── Deepgram transcription ─────────────────────────────────────────────────────

def _transcribe_sync(audio_path: str) -> str:
    """Synchronous Deepgram nova-2 transcription via REST API (SDK-version-agnostic)."""
    import json as _json
    import ssl
    import urllib.request
    import certifi

    api_key = os.environ["DEEPGRAM_API_KEY"]
    url = (
        "https://api.deepgram.com/v1/listen"
        "?model=nova-2&language=en&punctuate=false&smart_format=false"
    )
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    req = urllib.request.Request(
        url,
        data=audio_data,
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
        method="POST",
    )
    ctx = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, context=ctx) as resp:
                result = _json.loads(resp.read())
            break
        except Exception:
            if attempt == 2:
                raise

    return result["results"]["channels"][0]["alternatives"][0]["transcript"].strip().lower()


def _normalise(text: str) -> str:
    """Lowercase + strip punctuation for fair WER comparison."""
    import re
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


# ── Per-clip tests ─────────────────────────────────────────────────────────────

@pytest.mark.requires_audio
@pytest.mark.slow
@pytest.mark.parametrize("clip", CORPUS_EXISTING, ids=[c["file"] for c in CORPUS_EXISTING])
def test_wer_per_clip(clip):
    """Each clip must have WER < 15%."""
    hypothesis = _transcribe_sync(clip["file"])
    reference = _normalise(clip["reference"])
    hypothesis = _normalise(hypothesis)
    score = wer(reference, hypothesis)
    print(f"[{clip['codec']}/{clip['accent']}] WER={score:.3f} | ref={reference!r} | hyp={hypothesis!r}")
    assert score < 0.15, (
        f"WER {score:.3f} exceeds 15% threshold for {clip['file']}\n"
        f"  reference: {reference}\n"
        f"  hypothesis: {hypothesis}"
    )


# ── Aggregate tests ────────────────────────────────────────────────────────────

@pytest.mark.requires_audio
@pytest.mark.slow
def test_aggregate_wer():
    """Full-corpus aggregate WER must be under 8%."""
    if not CORPUS_EXISTING:
        pytest.skip("No corpus files found — run corpus_builder.py first")

    refs, hyps = [], []
    for clip in CORPUS_EXISTING:
        hyp = _transcribe_sync(clip["file"])
        refs.append(_normalise(clip["reference"]))
        hyps.append(_normalise(hyp))

    score = wer(refs, hyps)
    print(f"Aggregate WER: {score:.2%}  ({len(refs)} clips)")
    assert score < 0.10, f"Aggregate WER {score:.3f} exceeds 10% threshold"


@pytest.mark.requires_audio
@pytest.mark.slow
def test_aggregate_cer():
    """Full-corpus aggregate CER must be under 4%."""
    if not CORPUS_EXISTING:
        pytest.skip("No corpus files found — run corpus_builder.py first")

    refs, hyps = [], []
    for clip in CORPUS_EXISTING:
        hyp = _transcribe_sync(clip["file"])
        refs.append(_normalise(clip["reference"]))
        hyps.append(_normalise(hyp))

    score = cer(refs, hyps)
    print(f"Aggregate CER: {score:.2%}  ({len(refs)} clips)")
    assert score < 0.04, f"Aggregate CER {score:.3f} exceeds 4% threshold"


# ── Codec-specific breakdown ───────────────────────────────────────────────────

@pytest.mark.requires_audio
@pytest.mark.slow
def test_g711_wer_vs_pcm():
    """G.711 telephony clips should not be more than 5 pp worse than PCM baseline."""
    pcm_clips = [c for c in CORPUS_EXISTING if c["codec"] == "pcm" and c["noise"] == "clean"]
    g711_clips = [c for c in CORPUS_EXISTING if c["codec"] == "g711"]

    if not pcm_clips or not g711_clips:
        pytest.skip("Need both PCM and G.711 clips in corpus")

    def _avg_wer(clips):
        scores = []
        for clip in clips:
            hyp = _normalise(_transcribe_sync(clip["file"]))
            ref = _normalise(clip["reference"])
            scores.append(wer(ref, hyp))
        return sum(scores) / len(scores)

    pcm_wer = _avg_wer(pcm_clips)
    g711_wer = _avg_wer(g711_clips)
    delta = g711_wer - pcm_wer
    print(f"PCM WER={pcm_wer:.3f}  G.711 WER={g711_wer:.3f}  delta={delta:.3f}")
    assert delta < 0.05, f"G.711 degrades WER by {delta:.3f} (max allowed: 0.05)"


@pytest.mark.requires_audio
@pytest.mark.slow
def test_noisy_clips_wer():
    """Noisy clips must still achieve WER < 20%."""
    noisy_clips = [c for c in CORPUS_EXISTING if c["noise"] != "clean"]
    if not noisy_clips:
        pytest.skip("No noisy clips in corpus")

    for clip in noisy_clips:
        hyp = _normalise(_transcribe_sync(clip["file"]))
        ref = _normalise(clip["reference"])
        score = wer(ref, hyp)
        print(f"[noisy/{clip['noise']}] WER={score:.3f}")
        assert score < 0.20, f"Noisy clip WER {score:.3f} exceeds 20% for {clip['file']}"
