"""
corpus_builder.py — Generate the STT test corpus using OpenAI TTS.

Run once before L1 tests:
    python corpus_builder.py

Produces:
  corpus/native/*.wav      — clean PCM at 16 kHz
  corpus/accented/*.wav    — accent-voice variants (simulated via different TTS voices)
  corpus/noisy/*.wav       — Gaussian noise added at target SNR
  corpus/telephony/*.wav   — G.711 μ-law 8 kHz downsampled
"""
import json
import os
import io
import struct
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

MANIFEST_PATH = Path("corpus/manifest.json")
SAMPLE_RATE = 16000

NATIVE_UTTERANCES = [
    ("clip_001.wav", "I would like to book a flight to New York"),
    ("clip_002.wav", "What is the baggage allowance on economy class"),
    ("clip_003.wav", "I need to cancel my reservation for next Monday"),
    ("clip_004.wav", "Can I check in online twenty four hours before departure"),
    ("clip_005.wav", "My booking reference is AB one two three four"),
    ("clip_006.wav", "I want to speak to a real person please"),
    ("clip_007.wav", "Are meals included on the transatlantic flight"),
    ("clip_008.wav", "Flight United four seven eight departs at two thirty PM"),
]

ACCENTED_UTTERANCES = [
    ("clip_001.wav", "I would like to book a flight to New York", "en-IN"),
    ("clip_002.wav", "What is the baggage allowance on economy class", "en-GB"),
    ("clip_003.wav", "Can I get a refund if I cancel my ticket", "en-AU"),
]

NOISY_UTTERANCES = [
    ("clip_001.wav", "I would like to book a flight to New York", 10),
    ("clip_002.wav", "What is the baggage allowance on economy class", 15),
]

# OpenAI TTS voices that approximate accent variety
ACCENT_VOICES = {
    "en-US": "alloy",
    "en-GB": "echo",
    "en-IN": "onyx",
    "en-AU": "nova",
}


def synthesize_pcm(text: str, voice: str = "alloy") -> bytes:
    """Call OpenAI TTS and return raw 16-bit PCM @ 24 kHz."""
    from openai import OpenAI
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="pcm",  # raw 16-bit LE PCM at 24 kHz
    )
    return response.content


def pcm24k_to_wav16k(pcm_data: bytes, output_path: Path) -> None:
    """Convert raw 24 kHz PCM → 16 kHz WAV using simple decimation."""
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
    # Resample 24000 → 16000 (factor 2/3)
    # Simple polyphase: upsample by 2, downsample by 3
    upsampled = np.repeat(samples, 2)
    downsampled = upsampled[::3]
    resampled = downsampled.astype(np.int16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(resampled.tobytes())
    print(f"  ✓ {output_path}")


def add_noise(wav_path: Path, snr_db: float) -> Path:
    """Add white Gaussian noise at the given SNR (dB). Returns noisy path."""
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        data = wf.readframes(wf.getnframes())

    signal = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noisy = np.clip(signal + noise, -32768, 32767).astype(np.int16)

    noisy_path = wav_path
    with wave.open(str(noisy_path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(noisy.tobytes())
    return noisy_path


def convert_to_g711(input_path: Path, output_path: Path) -> None:
    """Downsample to 8 kHz μ-law via ffmpeg — simulates telephony G.711."""
    try:
        import ffmpeg
        (
            ffmpeg.input(str(input_path))
            .audio.filter("aresample", 8000)
            .output(str(output_path), acodec="pcm_mulaw", ar=8000)
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"  ✓ {output_path} (G.711 μ-law)")
    except Exception as e:
        print(f"  ✗ G.711 conversion failed (ffmpeg required): {e}")
        print("    Install ffmpeg: brew install ffmpeg  OR  apt install ffmpeg")


def build_corpus() -> None:
    print("\n── Building audio corpus ─────────────────────────────────────────────")

    # Native (en-US, clean)
    print("\n[1/4] Native en-US clips (clean):")
    for filename, text in NATIVE_UTTERANCES:
        out = Path("corpus/native") / filename
        pcm = synthesize_pcm(text, voice="alloy")
        pcm24k_to_wav16k(pcm, out)

    # Accented
    print("\n[2/4] Accented clips:")
    for filename, text, accent in ACCENTED_UTTERANCES:
        out = Path("corpus/accented") / filename
        voice = ACCENT_VOICES.get(accent, "alloy")
        pcm = synthesize_pcm(text, voice=voice)
        pcm24k_to_wav16k(pcm, out)

    # Noisy (clone native, add noise)
    print("\n[3/4] Noisy clips:")
    for filename, text, snr in NOISY_UTTERANCES:
        out = Path("corpus/noisy") / filename
        pcm = synthesize_pcm(text, voice="alloy")
        pcm24k_to_wav16k(pcm, out)
        add_noise(out, snr_db=snr)
        print(f"      SNR={snr} dB noise added")

    # Telephony G.711
    print("\n[4/4] Telephony (G.711 μ-law 8 kHz):")
    telephony_sources = [
        ("clip_001.wav", NATIVE_UTTERANCES[0][1]),
        ("clip_002.wav", NATIVE_UTTERANCES[1][1]),
        ("clip_003.wav", NATIVE_UTTERANCES[2][1]),
    ]
    for filename, text in telephony_sources:
        native_out = Path("corpus/native") / filename
        tele_out = Path("corpus/telephony") / filename
        tele_out.parent.mkdir(parents=True, exist_ok=True)
        convert_to_g711(native_out, tele_out)

    print("\n✅ Corpus built. Verify with: python corpus_builder.py --verify\n")


def verify_corpus() -> None:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    missing = []
    for clip in manifest["clips"]:
        p = Path(clip["file"])
        if not p.exists():
            missing.append(str(p))

    if missing:
        print(f"✗ Missing {len(missing)} audio files:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun: python corpus_builder.py")
    else:
        print(f"✓ All {len(manifest['clips'])} corpus files present.")


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv:
        verify_corpus()
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")
            sys.exit(1)
        build_corpus()
