#!/usr/bin/env bash
# run_all_tests.sh — Run all 6 test layers and generate the unified HTML report.
#
# Usage:
#   bash run_all_tests.sh            # run everything
#   bash run_all_tests.sh --no-e2e  # skip E2E (when agent not running)
#   bash run_all_tests.sh --layer l1 # run single layer

set -euo pipefail

SKIP_E2E=false
SINGLE_LAYER=""

for arg in "$@"; do
  case "$arg" in
    --no-e2e)      SKIP_E2E=true ;;
    --layer) ;;
    l1|l2|l3|l4|l5|e2e) SINGLE_LAYER="$arg" ;;
  esac
done

mkdir -p reports
START_TIME=$(date +%s)

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RESET='\033[0m'
pass() { echo -e "${GREEN}✓ $1${RESET}"; }
fail() { echo -e "${RED}✗ $1${RESET}"; }
info() { echo -e "${YELLOW}▸ $1${RESET}"; }

run_layer() {
  local label="$1"
  local file="$2"
  local report="$3"
  info "Running $label..."
  if pytest "$file" -v \
      --json-report \
      --json-report-file="$report" \
      2>&1; then
    pass "$label"
  else
    fail "$label (see $report)"
    FAILURES+=("$label")
  fi
}

FAILURES=()

# ── Layer selection ────────────────────────────────────────────────────────────
if [[ -z "$SINGLE_LAYER" ]]; then

  # L1 — STT
  run_layer "L1 STT" "test_l1_stt.py" "reports/l1_stt.json"

  # L2 — NLU
  run_layer "L2 NLU" "test_l2_nlu.py" "reports/l2_nlu.json"

  # L3 — Orchestration (state graph tests)
  run_layer "L3 Orchestration" "test_l3_orchestration.py" "reports/l3_state.json"

  # L3 — PromptFoo guardrails (requires npx / Node.js)
  if command -v npx &>/dev/null; then
    info "Running L3 PromptFoo eval..."
    if npx promptfoo eval \
        --config promptfoo.yaml \
        --output reports/l3_promptfoo.json 2>&1; then
      pass "L3 PromptFoo"
    else
      fail "L3 PromptFoo"
      FAILURES+=("L3 PromptFoo")
    fi
  else
    echo "⚠  npx not found — skipping PromptFoo eval (install Node.js to enable)"
  fi

  # L4 — LLM quality
  run_layer "L4 LLM" "test_l4_llm.py" "reports/l4_llm.json"

  # L5 — TTS
  run_layer "L5 TTS" "test_l5_tts.py" "reports/l5_tts.json"

  # E2E — full conversation
  if [[ "$SKIP_E2E" == "true" ]]; then
    echo "⚠  Skipping E2E (--no-e2e flag set). Start 'python agent.py dev' and re-run without the flag."
  else
    run_layer "E2E" "test_e2e.py" "reports/e2e.json"
  fi

else
  # Single layer mode
  case "$SINGLE_LAYER" in
    l1)  run_layer "L1 STT"            "test_l1_stt.py"            "reports/l1_stt.json" ;;
    l2)  run_layer "L2 NLU"            "test_l2_nlu.py"            "reports/l2_nlu.json" ;;
    l3)  run_layer "L3 Orchestration"  "test_l3_orchestration.py"  "reports/l3_state.json" ;;
    l4)  run_layer "L4 LLM"            "test_l4_llm.py"            "reports/l4_llm.json" ;;
    l5)  run_layer "L5 TTS"            "test_l5_tts.py"            "reports/l5_tts.json" ;;
    e2e) run_layer "E2E"               "test_e2e.py"               "reports/e2e.json" ;;
  esac
fi

# ── Generate report ────────────────────────────────────────────────────────────
info "Generating HTML report..."
python generate_report.py || true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Total time: ${ELAPSED}s"
if [[ ${#FAILURES[@]} -eq 0 ]]; then
  pass "All layers passed!"
else
  fail "Failed layers: ${FAILURES[*]}"
  echo ""
  echo "  Report: reports/index.html"
  exit 1
fi
echo "══════════════════════════════════════════════════════════════"
echo "  Report: reports/index.html"
