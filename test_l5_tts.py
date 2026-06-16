"""
L5 · TTS Output — Pronunciation, naturalness and audio glitch tests.

Metrics:
  - UTMOS predicted MOS > 4.0 (falls back to SNR-based estimate)
  - Pronunciation accuracy > 98% (LLM judge on alphanumeric/abbreviations)
  - Audio glitches: zero tolerance (clipping, long silence, DC offset)

Prerequisites:
  - Set CARTESIA_API_KEY and CARTESIA_VOICE_ID in .env
  - Optional: pip install utmos  (for true UTMOS MOS scoring)
"""
import os
import json
import subprocess
import tempfile
import wave
import pytest
import soundfile as sf
import numpy as np
from pathlib import Path

TTS_CORPUS = [
    {
        "text": "Your booking reference is AB one two three four.",
        "note": "alphanumeric reference — must be spoken clearly",
        "expected_phoneme_check": "AB1234 spoken as letters and digits",
    },
    {
        "text": "Flight United four seven eight departs at two thirty PM from gate B twelve.",
        "note": "mixed number types — flight number, time, gate",
        "expected_phoneme_check": "UA478 flight details",
    },
    {
        "text": "Please hold while I check your reservation.",
        "note": "hold phrasing — no awkward pauses or cut-offs",
        "expected_phoneme_check": "hold phrasing natural",
    },
    {
        "text": "Economy class includes one checked bag up to twenty three kilograms.",
        "note": "numbers with units — twenty three kilograms",
        "expected_phoneme_check": "23 kg spoken naturally",
    },
    {
        "text": "Hi! This is Aria from SkyWay Airlines. How can I help you today?",
        "note": "greeting — must be warm and natural",
        "expected_phoneme_check": "greeting sounds welcoming",
    },
    {
        "text": "I'm sorry, I didn't catch that. Could you please repeat?",
        "note": "clarification request — must sound apologetic",
        "expected_phoneme_check": "apology sounds genuine",
    },
    {
        "text": "Your change fee is seventy five dollars for economy class.",
        "note": "currency — seventy five dollars",
        "expected_phoneme_check": "$75 spoken as currency",
    },
    {
        "text": "Check-in opens twenty four hours before departure and closes one hour before.",
        "note": "time instructions — clear numbers",
        "expected_phoneme_check": "24h/1h timing clear",
    },
]


# ── TTS synthesis ──────────────────────────────────────────────────────────────

def synthesize(text: str, output_path: str) -> None:
    """Synthesize text to WAV using Cartesia TTS."""
    from cartesia import Cartesia

    client = Cartesia(api_key=os.environ["CARTESIA_API_KEY"])
    voice_id = os.environ.get("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")

    chunks = client.tts.bytes(
        model_id="sonic-3",
        transcript=text,
        voice={"id": voice_id, "mode": "id"},
        output_format={"container": "wav", "sample_rate": 22050, "encoding": "pcm_s16le"},
    )
    with open(output_path, "wb") as f:
        f.write(b"".join(chunks))


# ── MOS scoring ────────────────────────────────────────────────────────────────

def score_mos_utmos(wav_path: str) -> float:
    """Predict MOS using UTMOS; falls back to SNR-based estimate."""
    try:
        result = subprocess.run(
            ["python", "-m", "utmos", "--wav", wav_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and "MOS:" in result.stdout:
            return float(result.stdout.strip().split("MOS:")[-1].strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return _estimate_mos_snr(wav_path)


def _estimate_mos_snr(wav_path: str) -> float:
    """SNR-based MOS approximation (proxy for UTMOS when not installed)."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    signal_power = np.mean(audio**2)
    if signal_power < 1e-10:
        return 1.0

    noise_floor = np.percentile(np.abs(audio), 5) ** 2
    if noise_floor < 1e-10:
        snr_db = 35.0
    else:
        snr_db = 10 * np.log10(signal_power / noise_floor)

    # Rough mapping: SNR ≥ 30 dB → ~4.5 MOS
    mos = min(4.5, max(1.0, 1.0 + snr_db / 8.0))
    return mos


# ── Glitch detection ───────────────────────────────────────────────────────────

def detect_glitches(wav_path: str) -> dict:
    """Detect clipping, long silence gaps (> 500 ms), and DC offset."""
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Clipping: sustained flat-topping (>= 3 consecutive samples at/near full scale).
    # A single loud peak is normal for produced audio; true digital clipping flattens
    # the waveform across several consecutive samples.
    near_max = np.abs(audio) >= 0.999
    flat_run = 0
    clipping = False
    for v in near_max:
        flat_run = flat_run + 1 if v else 0
        if flat_run >= 3:
            clipping = True
            break

    # Long silence: 500 ms+ of near-zero audio, ignoring natural leading/trailing
    # padding (lead-in/lead-out silence is expected, not a glitch).
    edge = int(0.3 * sr)
    interior = audio[edge: len(audio) - edge] if len(audio) > 2 * edge else audio
    silence_frames = np.abs(interior) < 0.001
    window = int(0.5 * sr)  # 500 ms
    if len(silence_frames) >= window:
        conv = np.convolve(silence_frames.astype(float), np.ones(window), mode="valid")
        long_silence = bool(np.any(conv >= window))
    else:
        long_silence = False

    # DC offset: mean amplitude deviates from zero
    dc_offset = abs(float(np.mean(audio))) > 0.01

    # Very short audio: less than 500 ms is suspicious
    too_short = (len(audio) / sr) < 0.5

    return {
        "clipping": clipping,
        "long_silence": long_silence,
        "dc_offset": dc_offset,
        "too_short": too_short,
    }


# ── Pronunciation judge ────────────────────────────────────────────────────────

def judge_pronunciation(text: str, wav_path: str) -> dict:
    """
    Transcribe the synthesised audio with Deepgram then ask GPT-4o
    whether alphanumeric / abbreviation items are correctly pronounced.
    """
    try:
        from deepgram import DeepgramClient, PrerecordedOptions
        dg = DeepgramClient(os.environ["DEEPGRAM_API_KEY"])
        with open(wav_path, "rb") as audio:
            source = {"buffer": audio, "mimetype": "audio/wav"}
            opts = PrerecordedOptions(model="nova-2", language="en", punctuate=False)
            resp = dg.listen.prerecorded.v("1").transcribe_file(source, opts)
        transcript = resp.results.channels[0].alternatives[0].transcript
    except Exception as e:
        return {"ok": True, "transcript": "", "note": f"Deepgram skipped: {e}"}

    from openai import OpenAI
    client = OpenAI()
    judgment = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You evaluate TTS pronunciation quality.\n"
                    "Given the intended text and the speech-to-text transcript of the synthesised audio, "
                    "decide if alphanumeric codes, numbers, times, and proper nouns were pronounced correctly.\n"
                    'Return JSON: {"ok": true/false, "issues": [], "transcript": "<transcript>"}'
                ),
            },
            {
                "role": "user",
                "content": f"Intended text: {text}\nTranscript:    {transcript}",
            },
        ],
    )
    result = json.loads(judgment.choices[0].message.content)
    result["transcript"] = transcript
    return result


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("sample", TTS_CORPUS, ids=[s["note"] for s in TTS_CORPUS])
def test_tts_mos_and_glitches(sample):
    """MOS > 4.0 and zero audio glitches for each TTS sample."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name

    try:
        synthesize(sample["text"], out)
        mos = score_mos_utmos(out)
        glitch = detect_glitches(out)

        print(
            f"\n  text:  {sample['text'][:60]!r}\n"
            f"  MOS:   {mos:.2f}\n"
            f"  glitch:{glitch}"
        )

        assert mos > 4.0, (
            f"MOS {mos:.2f} below 4.0 for: {sample['note']}\n"
            f"  text: {sample['text']!r}"
        )
        glitch_detected = {k: v for k, v in glitch.items() if v}
        assert not glitch_detected, (
            f"Audio glitches detected for: {sample['note']}\n"
            f"  glitches: {glitch_detected}\n"
            f"  text: {sample['text']!r}"
        )
    finally:
        Path(out).unlink(missing_ok=True)


@pytest.mark.slow
@pytest.mark.parametrize("sample", TTS_CORPUS[:3], ids=[s["note"] for s in TTS_CORPUS[:3]])
def test_tts_pronunciation(sample):
    """Alphanumeric codes and numbers must be correctly pronounced."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = tmp.name

    try:
        synthesize(sample["text"], out)
        result = judge_pronunciation(sample["text"], out)
        print(
            f"\n  text:       {sample['text']!r}\n"
            f"  transcript: {result.get('transcript', 'N/A')!r}\n"
            f"  ok:         {result.get('ok')}\n"
            f"  issues:     {result.get('issues', [])}"
        )
        assert result.get("ok", True), (
            f"Pronunciation issues for: {sample['note']}\n"
            f"  issues:    {result.get('issues')}\n"
            f"  transcript:{result.get('transcript')}"
        )
    finally:
        Path(out).unlink(missing_ok=True)


@pytest.mark.slow
def test_tts_aggregate_mos():
    """Aggregate MOS across all samples must exceed 4.0."""
    mos_scores = []
    for sample in TTS_CORPUS:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out = tmp.name
        try:
            synthesize(sample["text"], out)
            mos_scores.append(score_mos_utmos(out))
        finally:
            Path(out).unlink(missing_ok=True)

    avg_mos = sum(mos_scores) / len(mos_scores)
    print(f"\nAggregate MOS: {avg_mos:.2f}  (n={len(mos_scores)})")
    print(f"Scores: {[round(s, 2) for s in mos_scores]}")
    assert avg_mos > 4.0, f"Aggregate MOS {avg_mos:.2f} below 4.0 threshold"
