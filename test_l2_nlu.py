"""
L2 · NLU / Intent — Intent classification and slot filling tests.

Metrics:
  - Per-utterance intent accuracy
  - Slot key/value match
  - Macro F1 > 0.90 across all intent classes
  - Out-of-scope (OOS) detection rate > 90%

Prerequisites:
  - Set OPENAI_API_KEY in .env
"""
import json
import pytest
from pathlib import Path
from openai import OpenAI
from sklearn.metrics import classification_report, f1_score

INTENTS = ["book_flight", "cancel_booking", "check_status", "get_info", "out_of_scope"]

with open(Path(__file__).parent / "intents" / "test_set.json") as f:
    _data = json.load(f)
    TEST_CASES = _data["test_cases"]

OOS_CASES = [t for t in TEST_CASES if t["intent"] == "out_of_scope"]
IN_SCOPE_CASES = [t for t in TEST_CASES if t["intent"] != "out_of_scope"]
SLOT_CASES = [t for t in TEST_CASES if t.get("slots")]

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def classify_intent(utterance: str) -> dict:
    """LLM-based intent classifier + slot extractor."""
    client = _get_client()
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an intent classifier for a flight-booking voice assistant.\n"
                    f"Classify the utterance into exactly one of: {INTENTS}\n"
                    "Also extract any slots (destination, date, passengers, class, "
                    "booking_reference, flight_number, topic) if present.\n"
                    'Return ONLY JSON: {"intent": "<intent>", "slots": {"key": "value"}}\n'
                    "If no slots, return empty object for slots."
                ),
            },
            {"role": "user", "content": utterance},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# ── Per-utterance intent accuracy ──────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("tc", TEST_CASES, ids=[t["utterance"][:50] for t in TEST_CASES])
def test_intent_classification(tc):
    result = classify_intent(tc["utterance"])
    assert result["intent"] == tc["intent"], (
        f"Utterance: {tc['utterance']!r}\n"
        f"  Expected intent: {tc['intent']}\n"
        f"  Got:             {result['intent']}"
    )


# ── Slot filling ───────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("tc", SLOT_CASES, ids=[t["utterance"][:50] for t in SLOT_CASES])
def test_slot_filling(tc):
    result = classify_intent(tc["utterance"])
    for key, expected_val in tc["slots"].items():
        assert key in result.get("slots", {}), (
            f"Missing slot '{key}' in response for: {tc['utterance']!r}"
        )
        got_val = result["slots"][key]
        assert expected_val.lower() in got_val.lower(), (
            f"Slot '{key}': expected {expected_val!r}, got {got_val!r}"
        )


# ── Aggregate F1 ───────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_macro_f1_score():
    """Macro F1 must exceed 0.90 across all intent classes."""
    y_true, y_pred = [], []
    for tc in TEST_CASES:
        result = classify_intent(tc["utterance"])
        y_true.append(tc["intent"])
        y_pred.append(result.get("intent", "unknown"))

    report = classification_report(y_true, y_pred, labels=INTENTS, output_dict=True)
    print("\n" + classification_report(y_true, y_pred, labels=INTENTS))
    macro_f1 = report["macro avg"]["f1-score"]
    assert macro_f1 > 0.90, f"Macro F1 {macro_f1:.3f} is below 0.90 threshold"


# ── OOS detection ──────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_oos_detection_rate():
    """Out-of-scope utterances must be correctly identified at > 90% rate."""
    if not OOS_CASES:
        pytest.skip("No out-of-scope test cases in test_set.json")

    correct = 0
    for tc in OOS_CASES:
        result = classify_intent(tc["utterance"])
        if result.get("intent") == "out_of_scope":
            correct += 1
        else:
            print(f"  OOS miss: {tc['utterance']!r} → classified as {result.get('intent')}")

    rate = correct / len(OOS_CASES)
    print(f"\nOOS detection rate: {rate:.2%} ({correct}/{len(OOS_CASES)})")
    assert rate > 0.90, f"OOS detection rate {rate:.2%} is below 90% threshold"


# ── Per-class precision ────────────────────────────────────────────────────────

@pytest.mark.slow
def test_per_intent_f1():
    """Each individual intent class must achieve F1 >= 0.80."""
    y_true, y_pred = [], []
    for tc in TEST_CASES:
        result = classify_intent(tc["utterance"])
        y_true.append(tc["intent"])
        y_pred.append(result.get("intent", "unknown"))

    report = classification_report(y_true, y_pred, labels=INTENTS, output_dict=True)
    failed = []
    for intent in INTENTS:
        f1 = report.get(intent, {}).get("f1-score", 0.0)
        if f1 < 0.80:
            failed.append(f"{intent}: F1={f1:.3f}")

    assert not failed, "Per-intent F1 below 0.80:\n  " + "\n  ".join(failed)
