# SkyWay Airlines Voice Agent — Test Suite

A 6-layer test harness for a LiveKit voice agent (Deepgram STT → GPT-4o → Cartesia TTS,
Silero VAD), plus a live dashboard for running tests and watching results.

## Test layers

| Layer | File | What it checks |
|---|---|---|
| L1 — STT | `test_l1_stt.py` | Deepgram transcription accuracy on sample audio |
| L2 — NLU | `test_l2_nlu.py` | Intent/entity extraction from transcripts |
| L3 — Orchestration | `test_l3_orchestration.py` | Conversation state graph, plus PromptFoo guardrail evals |
| L4 — LLM | `test_l4_llm.py` | Response quality via RAGAS (faithfulness, answer relevancy) |
| L5 — TTS | `test_l5_tts.py` | Cartesia audio output (MOS proxy, clipping/silence/glitch detection) |
| E2E | `test_e2e.py` | Full simulated phone call against a *live* running agent |

L1–L5 test components in isolation (no live agent needed). E2E is the only layer that
drives a real conversation and therefore needs the agent worker running first.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your real API keys
```

All commands below assume `.venv` is activated. **Always use `.venv`'s Python** —
mixing in the system/Framework Python will cause dependency and SSL-certificate issues.

Required keys in `.env`: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`,
`OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`.

## Running tests from the command line

Run everything (L1–L5, skipping E2E since no agent is running yet):

```bash
bash run_all_tests.sh --no-e2e
```

Run a single layer:

```bash
bash run_all_tests.sh --layer l4
```

Run E2E (requires the agent worker running in another terminal first — see below):

```bash
python agent.py start &      # start the agent worker
bash run_all_tests.sh         # runs all layers including E2E
```

Or run any layer directly with pytest:

```bash
pytest test_l1_stt.py -v
pytest test_e2e.py -v --json-report --json-report-file=reports/e2e.json
```

Each layer writes its results to `reports/<layer>.json`. After any run, build the
combined HTML report:

```bash
python generate_report.py
open reports/index.html
```

## Running and monitoring via the UI dashboard

The dashboard lets you trigger any layer (or all of them) with one click and watch
test output stream live, instead of reading terminal logs.

```bash
uvicorn ui_server:app --reload --port 8000
open http://localhost:8000
```

In the dashboard:
- Click a layer card (L1–L5, E2E) to run just that layer, or "Run All" to run everything.
- Output streams live over a WebSocket as each test executes — pass/fail status updates
  per layer in real time as `PASSED`/`FAILED` lines are parsed out of the pytest output.
- Click "Generate Report" to rebuild `reports/index.html` from the latest JSON results
  and view it inline in the dashboard.
- To stop a running layer, click it again (sends a `stop` action that kills the
  underlying pytest subprocess).

**E2E from the UI also requires the agent worker running separately first** — the
dashboard only runs the test harness, not the agent itself. Start it in a terminal:

```bash
python agent.py start
```

and leave it running while you trigger E2E from the dashboard.

## How to read the results

- **L1–L5**: each is a normal pytest pass/fail. A failure means a specific quality
  threshold was missed (e.g. RAGAS faithfulness < 0.85, a glitch detected in synthesized
  audio, a misclassified intent) — check the assertion message for the exact metric and
  threshold.
- **E2E** (`test_e2e.py`) runs the same 4 scenarios (`happy_path_booking`,
  `interruption_mid_response`, `escalation_flow`, `hold_and_resume`) **three separate
  times** across three test functions:
  - `test_e2e_scenario[<name>]` — one pytest result per scenario (4 results).
  - `test_e2e_task_completion_rate` — reruns all 4 scenarios and asserts the overall
    completion rate is ≥ 85%. Reported as a single pass/fail even though 4 conversations
    ran underneath it.
  - `test_e2e_average_turns` — reruns all 4 scenarios again and asserts the average
    turn count is within bounds. Also a single pass/fail covering 4 conversations.

  So a full E2E run drives **12 real conversations** against the live agent but reports
  only **6** pytest results — don't be surprised the numbers don't match 1:1.

  When an E2E scenario fails, the assertion message includes the full transcript
  (`[AGENT] ... [CALLER] ...`) — read that first; it almost always shows exactly where
  the conversation went off-script (e.g. the agent deflecting to a human transfer instead
  of completing a booking).

- **Worker log** (`/tmp/agent_worker.log` if you redirect it there) shows what the agent
  itself is doing — useful when a scenario fails with no clear transcript reason, or to
  confirm the worker registered with LiveKit Cloud and is receiving job dispatches.

## Common gotchas

- If E2E reports every scenario timing out on turn 1, the agent worker likely isn't
  running or didn't register — check `python agent.py start` output for `"registered
  worker"`.
- If E2E stalls with no new log lines for minutes (but the process is still alive),
  suspect a network call without a timeout (OpenAI/Deepgram requests in `e2e/audio_utils.py`
  and `e2e/caller_bot.py` are configured with explicit short timeouts for this reason —
  if you add new network calls there, give them a timeout too).
- Repeated `ignoring byte stream with topic 'lk.agent.session', no callback attached`
  log lines are normal LiveKit SDK noise, not an error.
