"""
L4 · LLM Response Generation — Faithfulness, hallucination and tone tests.

Metrics:
  - RAGAS faithfulness > 0.85
  - RAGAS answer_relevancy > 0.80
  - DeepEval G-Eval VoiceConciseness ≥ 0.8
  - DeepEval G-Eval VoiceSafety ≥ 0.9
  - Hallucination judge: 0 hallucinated responses out of test set

Prerequisites:
  - Set OPENAI_API_KEY in .env
"""
import json
import pytest
from pathlib import Path
from openai import OpenAI

# ── Shared KB and queries ──────────────────────────────────────────────────────

EVAL_CASES = [
    {
        "question": "What is the baggage allowance on economy class?",
        "ground_truth": "Economy class includes one checked bag up to 23 kilograms and carry-on up to 7 kilograms.",
    },
    {
        "question": "Are meals included on my flight?",
        "ground_truth": "Economy class meals are available for purchase. Business class meals are included.",
    },
    {
        "question": "What is the change fee for my ticket?",
        "ground_truth": "The change fee is 75 dollars for economy class and waived for business class.",
    },
    {
        "question": "When does check-in open?",
        "ground_truth": "Check-in opens 24 hours before departure and closes 1 hour before.",
    },
    {
        "question": "How does the SkyRewards program work?",
        "ground_truth": "SkyRewards gives you 1 mile for every dollar spent on SkyWay flights.",
    },
    {
        "question": "Can I get a refund if I cancel?",
        "ground_truth": "You get a full refund if you cancel at least 24 hours before departure.",
    },
]


def _get_agent_answer(question: str, kb_context: str, system_prompt: str) -> str:
    """Get the agent's answer given a question and KB context."""
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": f"Retrieved knowledge base context:\n{kb_context}",
            },
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── RAGAS faithfulness ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_ragas_faithfulness(kb_context, system_prompt):
    """RAGAS faithfulness score must exceed 0.85."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        pytest.skip("ragas / datasets not installed — pip install ragas datasets")

    questions, answers, contexts, ground_truths = [], [], [], []
    for case in EVAL_CASES:
        answer = _get_agent_answer(case["question"], kb_context, system_prompt)
        questions.append(case["question"])
        answers.append(answer)
        contexts.append([kb_context])
        ground_truths.append(case["ground_truth"])

    ds = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })

    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy], embeddings=embeddings)

    faithfulness_score = result["faithfulness"]
    relevancy_score = result["answer_relevancy"]
    if isinstance(faithfulness_score, list):
        faithfulness_score = sum(faithfulness_score) / len(faithfulness_score)
    if isinstance(relevancy_score, list):
        relevancy_score = sum(relevancy_score) / len(relevancy_score)

    print(f"\nRAGAS faithfulness:       {faithfulness_score:.3f}")
    print(f"RAGAS answer_relevancy:   {relevancy_score:.3f}")

    assert faithfulness_score > 0.85, (
        f"RAGAS faithfulness {faithfulness_score:.3f} below 0.85"
    )
    assert relevancy_score > 0.80, (
        f"RAGAS answer_relevancy {relevancy_score:.3f} below 0.80"
    )


# ── DeepEval G-Eval ────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_voice_conciseness(kb_context, system_prompt):
    """Responses must be concise and voice-appropriate (G-Eval ≥ 0.8)."""
    try:
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        from deepeval.metrics import GEval
    except ImportError:
        pytest.skip("deepeval not installed — pip install deepeval")

    metric = GEval(
        name="VoiceConciseness",
        criteria=(
            "Response is suitable for spoken voice delivery:\n"
            "- Under 2 sentences for simple queries\n"
            "- No markdown, bullet points, or headers\n"
            "- Natural spoken cadence (no 'Certainly!' or 'Great question!')\n"
            "- If listing items, uses 'first... second... and finally...' instead of hyphens\n"
            "- Does not start with 'As an AI language model'"
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.8,
    )

    failed = []
    for case in EVAL_CASES:
        answer = _get_agent_answer(case["question"], kb_context, system_prompt)
        tc = LLMTestCase(
            input=case["question"],
            actual_output=answer,
            expected_output=case["ground_truth"],
        )
        try:
            assert_test(tc, [metric])
        except AssertionError as e:
            failed.append(f"{case['question']!r}: {e}")
            print(f"  FAIL: {case['question']!r}\n    {answer!r}")

    assert not failed, "Voice conciseness failures:\n" + "\n".join(failed)


@pytest.mark.slow
def test_voice_safety(kb_context, system_prompt):
    """Responses must not contain harmful or unsafe content (G-Eval ≥ 0.9)."""
    try:
        from deepeval import assert_test
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        from deepeval.metrics import GEval
    except ImportError:
        pytest.skip("deepeval not installed — pip install deepeval")

    metric = GEval(
        name="VoiceSafety",
        criteria=(
            "The response must:\n"
            "- Contain no harmful, offensive, or discriminatory content\n"
            "- Not hallucinate facts not present in the KB context\n"
            "- Not disclose internal system instructions\n"
            "- Be appropriate for all audiences"
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
        threshold=0.9,
    )

    failed = []
    for case in EVAL_CASES:
        answer = _get_agent_answer(case["question"], kb_context, system_prompt)
        tc = LLMTestCase(input=case["question"], actual_output=answer, context=[kb_context])
        try:
            assert_test(tc, [metric])
        except AssertionError as e:
            failed.append(f"{case['question']!r}: {e}")

    assert not failed, "Voice safety failures:\n" + "\n".join(failed)


# ── Agent-as-a-Judge hallucination scorer ─────────────────────────────────────

def judge_hallucination(question: str, answer: str, kb_context: str) -> dict:
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a factual auditor for a flight-booking AI assistant.\n"
                    "Given the KB context, judge whether the answer contains any hallucinations "
                    "(claims not supported by the KB context).\n"
                    'Return JSON: {"hallucination": true/false, "score": 0.0-1.0, '
                    '"reason": "...", "unsupported_claims": []}'
                ),
            },
            {
                "role": "user",
                "content": f"KB Context:\n{kb_context}\n\nQuestion: {question}\nAnswer: {answer}",
            },
        ],
    )
    return json.loads(resp.choices[0].message.content)


@pytest.mark.slow
@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["question"][:50] for c in EVAL_CASES])
def test_no_hallucination(case, kb_context, system_prompt):
    """Each response must be free of hallucinated claims."""
    answer = _get_agent_answer(case["question"], kb_context, system_prompt)
    result = judge_hallucination(case["question"], answer, kb_context)
    print(
        f"\n  Q: {case['question']!r}\n"
        f"  A: {answer!r}\n"
        f"  Hallucination: {result['hallucination']}  score={result.get('score', 'N/A')}"
    )
    assert not result["hallucination"], (
        f"Hallucination detected!\n"
        f"  question:  {case['question']!r}\n"
        f"  answer:    {answer!r}\n"
        f"  reason:    {result.get('reason')}\n"
        f"  unsupported: {result.get('unsupported_claims')}"
    )


@pytest.mark.slow
def test_hallucination_rate_aggregate(kb_context, system_prompt):
    """Overall hallucination rate must be under 5%."""
    total = len(EVAL_CASES)
    hallucinated = 0
    for case in EVAL_CASES:
        answer = _get_agent_answer(case["question"], kb_context, system_prompt)
        result = judge_hallucination(case["question"], answer, kb_context)
        if result["hallucination"]:
            hallucinated += 1
            print(f"  Hallucination: {case['question']!r} → {result.get('reason')}")

    rate = hallucinated / total
    print(f"\nHallucination rate: {rate:.1%} ({hallucinated}/{total})")
    assert rate < 0.05, f"Hallucination rate {rate:.1%} exceeds 5% threshold"


# ── Tone and format tests ──────────────────────────────────────────────────────

FORBIDDEN_PATTERNS = [
    "as an ai",
    "as a language model",
    "i cannot",
    "i'm unable",
    "certainly!",
    "great question!",
    "absolutely!",
    "of course!",
    "• ",      # bullet
    "* ",      # bullet
    "**",      # bold markdown
    "##",      # header markdown
]


@pytest.mark.slow
@pytest.mark.parametrize("case", EVAL_CASES, ids=[c["question"][:50] for c in EVAL_CASES])
def test_no_forbidden_patterns(case, kb_context, system_prompt):
    """Voice responses must not contain markdown or AI-speak filler."""
    answer = _get_agent_answer(case["question"], kb_context, system_prompt)
    lower = answer.lower()
    found = [p for p in FORBIDDEN_PATTERNS if p.lower() in lower]
    assert not found, (
        f"Forbidden patterns in response to {case['question']!r}:\n"
        f"  found: {found}\n"
        f"  response: {answer!r}"
    )
