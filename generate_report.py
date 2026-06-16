"""
generate_report.py — Aggregate all layer JSON reports into a single HTML scorecard.

Run after run_all_tests.sh:
    python generate_report.py

Output: reports/index.html
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from jinja2 import Template

LAYERS = {
    "L1 STT":   {"file": "reports/l1.json",  "metric": "WER < 10%",          "threshold": 0.85},
    "L2 NLU":   {"file": "reports/l2.json",  "metric": "F1 > 0.90",          "threshold": 0.90},
    "L3 Orch":  {"file": "reports/l3.json",  "metric": "Pass rate > 95%",    "threshold": 0.95},
    "L4 LLM":   {"file": "reports/l4.json",  "metric": "Faithfulness > 0.85","threshold": 0.85},
    "L5 TTS":   {"file": "reports/l5.json",  "metric": "MOS > 4.0",          "threshold": 0.90},
    "E2E":      {"file": "reports/e2e.json", "metric": "Completion > 85%",   "threshold": 0.85},
}

LAYER_COLORS = {
    "L1 STT":  "#4F8EF7",
    "L2 NLU":  "#3DD68C",
    "L3 Orch": "#F5A623",
    "L4 LLM":  "#F06060",
    "L5 TTS":  "#E87DC8",
    "E2E":     "#7C6FF0",
}


def parse_pytest_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"passed": 0, "failed": 0, "skipped": 0, "total": 0, "rate": 0.0, "missing": True}
    with open(p) as f:
        data = json.load(f)
    summary = data.get("summary", {})
    passed  = summary.get("passed", 0)
    failed  = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    total   = summary.get("total", 0)
    ran = passed + failed  # rate excludes skipped — they didn't run, so they shouldn't count against pass rate
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total":  total,
        "rate":   passed / ran if ran > 0 else 0.0,
        "missing": False,
    }


# ── Per-test detail extraction ──────────────────────────────────────────────────
# Each test layer prints/raises a different shape of metric. These pull the
# single most useful number/line out of stdout (passed) or longrepr (failed)
# so the drill-down can show "what actually happened" per run, not just pass/fail.

def _short_name(nodeid: str) -> str:
    if "[" in nodeid:
        return nodeid.split("[", 1)[1].rstrip("]")
    return nodeid.split("::")[-1]


def _detail_l1(stdout, longrepr, outcome):
    m = re.search(r"WER=([\d.]+)", stdout)
    return f"WER {float(m.group(1)):.1%}" if m else ("✓ pass" if outcome == "passed" else "✗ fail")


def _detail_l2(stdout, longrepr, outcome):
    m = re.search(r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", stdout)
    if m:
        return f"macro F1 {float(m.group(1)):.3f}"
    m = re.search(r"OOS detection rate:\s*(.+)", stdout)
    if m:
        return m.group(1).strip()
    if outcome == "passed":
        return "✓ correct"
    m, g = re.search(r"Expected intent:\s*(\S+)", longrepr), re.search(r"Got:\s*(\S+)", longrepr)
    if m and g:
        return f"expected {m.group(1)}, got {g.group(1)}"
    m = re.search(r"expected '([^']*)', got '([^']*)'", longrepr)
    if m:
        return f"expected '{m.group(1)}', got '{m.group(2)}'"
    m = re.search(r"F1=([\d.]+)", longrepr)
    if m:
        return f"F1 {float(m.group(1)):.3f}"
    m = re.search(r"Missing slot '(\w+)'", longrepr)
    if m:
        return f"missing slot '{m.group(1)}'"
    return "✗ mismatch"


def _detail_l3(stdout, longrepr, outcome):
    return "✓ pass" if outcome == "passed" else "✗ fail"


def _detail_l4(stdout, longrepr, outcome):
    parts = []
    f = re.search(r"faithfulness:\s*([\d.]+)", stdout)
    r = re.search(r"answer_relevancy:\s*([\d.]+)", stdout)
    if f:
        parts.append(f"faithfulness {float(f.group(1)):.2f}")
    if r:
        parts.append(f"relevancy {float(r.group(1)):.2f}")
    h = re.search(r"score=([\d.]+)", stdout)
    if h:
        parts.append(f"score {float(h.group(1)):.2f}")
    rate = re.search(r"rate:\s*([\d.]+%[^\n]*)", stdout, re.I)
    if rate:
        parts.append(rate.group(1))
    if parts:
        return " · ".join(parts)
    return "✓ pass" if outcome == "passed" else "✗ fail"


def _detail_l5(stdout, longrepr, outcome):
    m = re.search(r"MOS:\s*([\d.]+)", stdout)
    detail = f"MOS {float(m.group(1)):.2f}" if m else ("✓ pass" if outcome == "passed" else "✗ fail")
    g = re.search(r"glitch:\s*(\{[^}]*\})", stdout)
    if g:
        flags = re.findall(r"'(\w+)':\s*True", g.group(1))
        if flags:
            detail += " · " + ", ".join(flags)
    return detail


def _detail_e2e(stdout, longrepr, outcome):
    text = stdout or longrepr
    turns = text.count("[CALLER]")
    goal = "✅ goal met" if outcome == "passed" else "❌ goal not met"
    return f"{goal} · {turns} turn(s)" if turns else goal


DETAIL_FNS = {
    "L1 STT":  _detail_l1,
    "L2 NLU":  _detail_l2,
    "L3 Orch": _detail_l3,
    "L4 LLM":  _detail_l4,
    "L5 TTS":  _detail_l5,
    "E2E":     _detail_e2e,
}


def parse_test_details(layer_name: str, path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    fn = DETAIL_FNS.get(layer_name, lambda *a: "—")
    out = []
    for t in data.get("tests", []):
        call = t.get("call", {})
        stdout = call.get("stdout") or ""
        longrepr = call.get("longrepr") or ""
        if not isinstance(longrepr, str):
            longrepr = ""
        outcome = t.get("outcome", "unknown")
        out.append({
            "name":     _short_name(t["nodeid"]),
            "outcome":  outcome,
            "detail":   fn(stdout, longrepr, outcome),
            "duration": call.get("duration", 0.0),
        })
    return out


def build_rows() -> list[dict]:
    rows = []
    for name, cfg in LAYERS.items():
        stats  = parse_pytest_json(cfg["file"])
        if stats["missing"]:
            status = "MISSING"
        else:
            status = "PASS" if stats["rate"] >= cfg["threshold"] else "FAIL"
        rows.append({
            "layer":     name,
            "metric":    cfg["metric"],
            "value":     f"{stats['rate']:.1%}" if not stats["missing"] else "—",
            "status":    status,
            "passed":    stats["passed"],
            "failed":    stats["failed"],
            "skipped":   stats["skipped"],
            "total":     stats["total"],
            "color":     LAYER_COLORS.get(name, "#4F8EF7"),
            "missing":   stats["missing"],
            "tests":     parse_test_details(name, cfg["file"]),
        })
    return rows


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Voice Agent Test Report</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0B0D12; color: #e8eaf0; font-family: system-ui, sans-serif;
         font-size: 15px; line-height: 1.6; padding: 40px; }
  h1 { font-size: 28px; font-weight: 700; margin-bottom: 6px; color: #fff; }
  .sub { color: #7a7f9a; font-size: 13px; margin-bottom: 32px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,0.07); }
  th { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
       color: #7a7f9a; }
  tr.layer-row { cursor: pointer; }
  tr.layer-row:hover td { background: rgba(255,255,255,0.03); }
  tr.layer-row.expanded td { background: rgba(255,255,255,0.04); }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 4px;
           font-size: 11px; font-weight: 600; letter-spacing: 0.06em; }
  .PASS    { background: rgba(61,214,140,0.15); color: #3DD68C; }
  .FAIL    { background: rgba(240,96,96,0.15);  color: #F06060; }
  .MISSING { background: rgba(245,166,35,0.15); color: #F5A623; }
  .layer-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
               margin-right: 8px; }
  .chevron { display: inline-block; margin-right: 8px; color: #7a7f9a; font-size: 11px;
             transition: transform 0.15s ease; }
  tr.layer-row.expanded .chevron { transform: rotate(90deg); }
  .summary { display: flex; gap: 24px; margin-bottom: 32px; flex-wrap: wrap; }
  .summary-card { background: #1a1e2e; border: 1px solid rgba(255,255,255,0.07);
                  border-radius: 10px; padding: 16px 24px; min-width: 140px; }
  .summary-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
                   color: #7a7f9a; margin-bottom: 4px; }
  .summary-value { font-size: 28px; font-weight: 700; color: #fff; }
  .summary-value.green { color: #3DD68C; }
  .summary-value.red   { color: #F06060; }
  .ts { color: #7a7f9a; font-size: 11px; margin-top: 32px; }

  .drill-row td { padding: 0; border-bottom: 1px solid rgba(255,255,255,0.07); }
  .drill-wrap { display: none; padding: 4px 16px 18px 44px; background: rgba(255,255,255,0.015); }
  .drill-row.open .drill-wrap { display: block; }
  .drill-table { width: 100%; border-collapse: collapse; margin-top: 4px; }
  .drill-table th, .drill-table td { padding: 7px 12px; font-size: 13px;
                                      border-bottom: 1px solid rgba(255,255,255,0.05); }
  .drill-table th { font-size: 10px; }
  .run-outcome { font-size: 12px; font-weight: 600; }
  .run-outcome.passed { color: #3DD68C; }
  .run-outcome.failed { color: #F06060; }
  .run-outcome.skipped { color: #F5A623; }
  .run-name { color: #c7cae0; max-width: 420px; }
  .run-detail { color: #9ea2bd; }
  .run-duration { color: #5d6080; font-size: 12px; }
  .no-runs { color: #5d6080; font-style: italic; padding: 10px 0; }
</style>
</head>
<body>
<h1>SkyWay Voice Agent — Test Report</h1>
<p class="sub">Generated: {{ timestamp }} · click a layer row to see individual test runs</p>

{% set pass_count = rows | selectattr('status', 'eq', 'PASS') | list | length %}
{% set fail_count = rows | selectattr('status', 'eq', 'FAIL') | list | length %}
{% set miss_count = rows | selectattr('status', 'eq', 'MISSING') | list | length %}
{% set total_tests = rows | map(attribute='total') | sum %}
{% set total_passed = rows | map(attribute='passed') | sum %}

<div class="summary">
  <div class="summary-card">
    <div class="summary-label">Layers passed</div>
    <div class="summary-value {% if fail_count == 0 %}green{% else %}red{% endif %}">
      {{ pass_count }}/{{ rows|length }}
    </div>
  </div>
  <div class="summary-card">
    <div class="summary-label">Tests passed</div>
    <div class="summary-value">{{ total_passed }}/{{ total_tests }}</div>
  </div>
  <div class="summary-card">
    <div class="summary-label">Overall status</div>
    <div class="summary-value {% if fail_count == 0 and miss_count == 0 %}green{% else %}red{% endif %}">
      {% if fail_count == 0 and miss_count == 0 %}PASS{% else %}FAIL{% endif %}
    </div>
  </div>
</div>

<table>
  <thead>
    <tr>
      <th>Layer</th>
      <th>Target metric</th>
      <th>Measured</th>
      <th>Passed</th>
      <th>Failed</th>
      <th>Skipped</th>
      <th>Total</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    {% for row in rows %}
    <tr class="layer-row" onclick="toggleDrill(this, 'drill-{{ loop.index }}')">
      <td>
        <span class="chevron">▶</span>
        <span class="layer-dot" style="background:{{ row.color }}"></span>
        {{ row.layer }}
      </td>
      <td>{{ row.metric }}</td>
      <td>{{ row.value }}</td>
      <td>{{ row.passed }}</td>
      <td>{{ row.failed }}</td>
      <td>{{ row.skipped }}</td>
      <td>{{ row.total }}</td>
      <td><span class="badge {{ row.status }}">{{ row.status }}</span></td>
    </tr>
    <tr class="drill-row" id="drill-{{ loop.index }}">
      <td colspan="8">
        <div class="drill-wrap">
          {% if row.tests %}
          <table class="drill-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Test run</th>
                <th>Outcome</th>
                <th>Measured</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {% for t in row.tests %}
              <tr>
                <td class="run-duration">{{ loop.index }}</td>
                <td class="run-name">{{ t.name }}</td>
                <td class="run-outcome {{ t.outcome }}">{{ t.outcome }}</td>
                <td class="run-detail">{{ t.detail }}</td>
                <td class="run-duration">{{ "%.2f"|format(t.duration) }}s</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
          {% else %}
          <div class="no-runs">No individual run data available for this layer yet — run it first.</div>
          {% endif %}
        </div>
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<p class="ts">
  Voice Agent QA Framework · Built with PromptFoo · RAGAS · DeepEval · LiveKit Agents SDK
</p>

<script>
function toggleDrill(rowEl, drillId) {
  rowEl.classList.toggle('expanded');
  document.getElementById(drillId).classList.toggle('open');
}
</script>
</body>
</html>"""


def main():
    Path("reports").mkdir(exist_ok=True)
    rows = build_rows()
    html = Template(TEMPLATE).render(rows=rows, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    out = Path("reports/index.html")
    out.write_text(html)
    print(f"✅ Report written to {out}")

    # Exit with non-zero if any layer failed
    failed = [r for r in rows if r["status"] == "FAIL"]
    if failed:
        print(f"✗ {len(failed)} layer(s) FAILED: {', '.join(r['layer'] for r in failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
