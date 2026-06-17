"""
Voice Agent Test Dashboard — FastAPI + WebSocket backend.

  pip install fastapi "uvicorn[standard]"
  uvicorn ui_server:app --reload --port 8000
  open http://localhost:8000
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

# Detect optional pytest-json-report plugin
try:
    import pytest_jsonreport  # noqa: F401
    _HAS_JSON_REPORT = True
except ImportError:
    _HAS_JSON_REPORT = False

load_dotenv()

BASE = Path(__file__).parent
app = FastAPI(title="Voice Agent Test Dashboard")

# ── Broadcast hub ──────────────────────────────────────────────────────────────

_sockets: list[WebSocket] = []


async def _broadcast(msg: dict):
    data = json.dumps(msg)
    dead = []
    for ws in list(_sockets):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _sockets:
            _sockets.remove(ws)


# ── Layer registry ─────────────────────────────────────────────────────────────

LAYERS = [
    ("l1",  "test_l1_stt.py"),
    ("l2",  "test_l2_nlu.py"),
    ("l3",  "test_l3_orchestration.py"),
    ("l4",  "test_l4_llm.py"),
    ("l5",  "test_l5_tts.py"),
    ("e2e", "test_e2e.py"),
]
LAYER_MAP = dict(LAYERS)

_procs: dict[str, asyncio.subprocess.Process] = {}
_tasks: dict[str, asyncio.Task] = {}

# ── Agent worker lifecycle (needed for E2E, which talks to a live agent) ───────

_agent_proc: asyncio.subprocess.Process | None = None
_agent_ready = asyncio.Event()
_agent_lock = asyncio.Lock()


async def _external_agent_running() -> bool:
    """True if some other process (e.g. a terminal) already has agent.py start up."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "pgrep", "-f", "agent.py start",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        return bool(out.strip())
    except FileNotFoundError:
        return False


async def _agent_status_payload() -> dict:
    """Return current agent worker status for broadcasting."""
    pid = None
    running = False
    if _agent_proc is not None and _agent_proc.returncode is None:
        running = True
        pid = _agent_proc.pid
    elif await _external_agent_running():
        running = True
        try:
            p = await asyncio.create_subprocess_exec(
                "pgrep", "-f", "agent.py start",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await p.communicate()
            pids = out.strip().split()
            if pids:
                pid = int(pids[0])
        except Exception:
            pass
    return {"type": "agent_status", "running": running, "pid": pid}


async def _watch_agent_worker(proc: asyncio.subprocess.Process):
    await _broadcast(await _agent_status_payload())
    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue
        if "registered worker" in line.lower():
            _agent_ready.set()
            await _broadcast(await _agent_status_payload())
        await _broadcast({"type": "log", "layer": "e2e", "text": f"[agent worker] {line}", "level": "info"})
    await proc.wait()
    _agent_ready.clear()
    await _broadcast({"type": "agent_status", "running": False, "pid": None})


async def _ensure_agent_worker():
    """Start the LiveKit agent worker if nothing is already serving E2E's room dispatch."""
    global _agent_proc
    async with _agent_lock:
        if _agent_proc is not None and _agent_proc.returncode is None:
            return  # we already started one and it's still alive

        if await _external_agent_running():
            await _broadcast({
                "type": "log", "layer": "e2e",
                "text": "Using already-running agent worker (started outside the UI).",
                "level": "section",
            })
            await _broadcast(await _agent_status_payload())
            return

        await _broadcast({
            "type": "log", "layer": "e2e",
            "text": "No agent worker detected — starting one (python agent.py start)...",
            "level": "section",
        })
        _agent_ready.clear()
        _agent_proc = await asyncio.create_subprocess_exec(
            sys.executable, "agent.py", "start",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(BASE),
            env={**os.environ},
        )
        asyncio.create_task(_watch_agent_worker(_agent_proc))

        try:
            await asyncio.wait_for(_agent_ready.wait(), timeout=30)
            await _broadcast({"type": "log", "layer": "e2e", "text": "Agent worker registered with LiveKit.", "level": "pass"})
        except asyncio.TimeoutError:
            await _broadcast({
                "type": "log", "layer": "e2e",
                "text": "Agent worker did not confirm registration within 30s — continuing anyway.",
                "level": "warn",
            })


# ── Test runner ────────────────────────────────────────────────────────────────

async def _run_layer(layer_id: str):
    test_file = LAYER_MAP[layer_id]
    report_out = f"reports/{layer_id}.json"
    Path("reports").mkdir(exist_ok=True)

    await _broadcast({"type": "status", "layer": layer_id, "status": "running"})
    t0 = time.monotonic()

    if layer_id == "e2e":
        await _ensure_agent_worker()

    cmd = [
        sys.executable, "-m", "pytest", test_file,
        "-v", "-s", "--tb=short", "--no-header",
    ]
    if _HAS_JSON_REPORT:
        cmd += ["--json-report", f"--json-report-file={report_out}"]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(BASE),
        env={**os.environ},
    )
    _procs[layer_id] = proc

    passed = failed = 0

    async for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue

        level = "info"
        if " PASSED" in line:
            passed += 1
            level = "pass"
        elif " FAILED" in line or " ERROR" in line or line.startswith("FAILED"):
            failed += 1
            level = "fail"
        elif "warning" in line.lower() or "warn" in line.lower():
            level = "warn"
        elif line.startswith("=") or line.startswith("-"):
            level = "section"

        await _broadcast({"type": "log", "layer": layer_id, "text": line, "level": level})

        # Parse E2E lines — two sources:
        #   1. Live per-turn logs from caller_bot:  "INFO caller-bot: [turn N] agent: ..." / "...caller: ..."
        #   2. End-of-scenario transcript block:    "  [AGENT ] text" / "  [CALLER] text"
        if layer_id == "e2e":
            low = line.lower()
            # Live turn logs (real-time, most useful for E2E Live tab)
            if "] agent:" in low:
                txt = line.split("] agent:", 1)[-1].strip()
                if txt:
                    await _broadcast({"type": "e2e_turn", "role": "agent", "text": txt})
            elif "] caller:" in low:
                txt = line.split("] caller:", 1)[-1].strip()
                if txt:
                    await _broadcast({"type": "e2e_turn", "role": "caller", "text": txt})
            # End-of-scenario transcript block: "[AGENT ]" or "[CALLER]"
            elif "[agent " in low or "[agent]" in low:
                txt = re.sub(r"^\s*\[agent\s*\]\s*", "", line, flags=re.IGNORECASE).strip()
                if txt:
                    await _broadcast({"type": "e2e_turn", "role": "agent", "text": txt})
            elif "[caller]" in low:
                txt = re.sub(r"^\s*\[caller\]\s*", "", line, flags=re.IGNORECASE).strip()
                if txt:
                    await _broadcast({"type": "e2e_turn", "role": "caller", "text": txt})
            elif "── starting scenario:" in low:
                scenario = line.split(":", 1)[-1].strip().replace("─", "").strip()
                await _broadcast({"type": "e2e_scenario", "name": scenario})
            # caller_bot logs: "Scenario 'X' done: goal=True turns=N"
            elif "done: goal=" in low:
                m = re.search(r"scenario '([^']+)' done: goal=(\w+)\s+turns=(\d+)", line, re.IGNORECASE)
                if m:
                    await _broadcast({
                        "type": "e2e_goal",
                        "text": line.strip(),
                        "success": m.group(2).lower() == "true",
                        "scenario": m.group(1),
                        "turns": int(m.group(3)),
                    })
            elif "task completion rate" in low:
                await _broadcast({"type": "e2e_metric", "text": line.strip()})

    await proc.wait()
    elapsed = round(time.monotonic() - t0, 1)

    # Pull accurate counts from the JSON report
    try:
        rpath = Path(report_out)
        if rpath.exists():
            summary = json.loads(rpath.read_text()).get("summary", {})
            passed = summary.get("passed", passed)
            failed = summary.get("failed", failed)
    except Exception:
        pass

    status = "passed" if proc.returncode == 0 else "failed"
    await _broadcast({
        "type": "done",
        "layer": layer_id,
        "status": status,
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "duration": elapsed,
    })
    _procs.pop(layer_id, None)
    _tasks.pop(layer_id, None)


def _stop(layer_id: str):
    proc = _procs.get(layer_id)
    if proc:
        try:
            proc.terminate()
        except Exception:
            pass
    task = _tasks.get(layer_id)
    if task and not task.done():
        task.cancel()


async def _run_all(include_e2e: bool = False):
    for lid, _ in LAYERS:
        if lid == "e2e" and not include_e2e:
            continue
        await _run_layer(lid)
    await _broadcast({"type": "all_done"})


# ── HTTP routes ────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    p = BASE / "ui" / "index.html"
    return HTMLResponse(p.read_text() if p.exists() else "<h1>ui/index.html not found</h1>")


@app.get("/report-content")
async def serve_report():
    p = BASE / "reports" / "index.html"
    return HTMLResponse(p.read_text() if p.exists() else "<p style='color:#aaa;padding:2rem'>No report yet — run tests first.</p>")


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _sockets.append(ws)
    # Send current state immediately so a reconnecting browser sees reality
    await ws.send_text(json.dumps(await _agent_status_payload()))
    e2e_running = "e2e" in _procs and _procs["e2e"].returncode is None
    await ws.send_text(json.dumps({"type": "status", "layer": "e2e", "status": "running" if e2e_running else "idle"}))
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")
            layer = msg.get("layer", "")

            if action == "run":
                if layer == "all":
                    _stop("all")
                    _tasks["all"] = asyncio.create_task(
                        _run_all(include_e2e=msg.get("include_e2e", False))
                    )
                elif layer in LAYER_MAP:
                    _stop(layer)
                    _tasks[layer] = asyncio.create_task(_run_layer(layer))

            elif action == "stop":
                if layer == "all":
                    for lid in list(LAYER_MAP.keys()):
                        _stop(lid)
                else:
                    _stop(layer)
                await _broadcast({"type": "status", "layer": layer, "status": "idle"})

            elif action == "kill_agent":
                global _agent_proc
                if _agent_proc is not None:
                    try:
                        _agent_proc.terminate()
                    except Exception:
                        pass
                    _agent_proc = None
                kill_p = await asyncio.create_subprocess_exec(
                    "pkill", "-9", "-f", "agent.py start",
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await kill_p.wait()
                _agent_ready.clear()
                await _broadcast({"type": "agent_status", "running": False, "pid": None})

            elif action == "start_agent":
                asyncio.create_task(_ensure_agent_worker())

            elif action == "generate_report":
                async def _gen():
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "generate_report.py",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT,
                        cwd=str(BASE),
                    )
                    async for raw_line in proc.stdout:
                        await _broadcast({
                            "type": "log", "layer": "report",
                            "text": raw_line.decode().rstrip(), "level": "info",
                        })
                    await proc.wait()
                    await _broadcast({"type": "report_ready"})
                asyncio.create_task(_gen())

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in _sockets:
            _sockets.remove(ws)


@app.on_event("shutdown")
async def _shutdown():
    # Only tear down the worker if *we* started it — leave externally-started ones alone.
    if _agent_proc is not None and _agent_proc.returncode is None:
        _agent_proc.terminate()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ui_server:app", host="0.0.0.0", port=8000, reload=True)
