#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-run}"
if [[ "$MODE" != "run" && "$MODE" != "--dry-run" ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
GAME="FinalFantasyDawnOfSouls-GbAdvance"
ROM="$PROJECT_ROOT/custom_integrations/$GAME/rom.gba"
OUTPUT_ROOT="$PROJECT_ROOT/videos/gba_feasibility"
LOG_ROOT="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_ROOT"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${RANDOM}_$$"
while [[ -e "$LOG_ROOT/$RUN_ID" ]]; do RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${RANDOM}_$$"; done
mkdir "$LOG_ROOT/$RUN_ID"
LOG_FILE="$LOG_ROOT/$RUN_ID/execution.log"
SUMMARY_FILE="$LOG_ROOT/$RUN_ID/execution_summary.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "FF1 Dawn of Souls GBA feasibility check"
echo "Run ID: $RUN_ID"
echo "Project: $PROJECT_ROOT"
echo "Mode: $MODE"
echo "Execution log: $LOG_FILE"

fail() {
  local message="$1"
  echo "ERROR: $message" >&2
  {
    echo "Run ID: $RUN_ID"
    echo "Status: failed"
    echo "Failure: $message"
    echo "Log: $LOG_FILE"
  } > "$SUMMARY_FILE"
  echo "Execution summary: $SUMMARY_FILE"
  exit 1
}

[[ -f "$PROJECT_ROOT/.venv/bin/activate" && -x "$PYTHON" ]] || fail "Existing WSL virtual environment is missing or incomplete: $PROJECT_ROOT/.venv"
[[ -f "$PROJECT_ROOT/scripts/gba_feasibility_probe.py" ]] || fail "GBA feasibility probe is missing."
[[ -f "$ROM" ]] || fail "Local GBA ROM is missing: $ROM"
source "$PROJECT_ROOT/.venv/bin/activate"
"$PYTHON" -c 'import numpy, PIL, stable_retro; import sys; stable_retro.data.Integrations.add_custom_path(sys.argv[1]); assert sys.argv[2] in stable_retro.list_games(), "custom GBA integration was not discovered"' "$PROJECT_ROOT/custom_integrations" "$GAME" || fail "Existing Stable-Retro runtime or GBA integration validation failed; no packages were installed."
echo "Existing Python runtime, Stable-Retro, GBA integration, and local ROM validated."

if [[ "$MODE" == "--dry-run" ]]; then
  "$PYTHON" scripts/gba_feasibility_probe.py --help >/dev/null || fail "GBA probe CLI validation failed."
  {
    echo "Run ID: $RUN_ID"
    echo "Status: dry-run passed"
    echo "No emulator was started and no gameplay capture was created."
    echo "Planned probe: $PYTHON scripts/gba_feasibility_probe.py --runtime --automate-party-setup"
    echo "Log: $LOG_FILE"
  } > "$SUMMARY_FILE"
  echo "Validation summary: $SUMMARY_FILE"
  exit 0
fi

if ! "$PYTHON" scripts/gba_feasibility_probe.py --runtime --automate-party-setup; then
  fail "GBA feasibility probe failed; inspect its unique report directory and this execution log."
fi
REPORT_LINE="$(grep -E '^GBA feasibility report:' "$LOG_FILE" | tail -n 1 || true)"
[[ -n "$REPORT_LINE" ]] || fail "Probe returned success but did not print a report path."
REPORT_PATH="${REPORT_LINE#GBA feasibility report: }"
REPORT_DIR="$(dirname "$REPORT_PATH")"
[[ -f "$REPORT_DIR/run_report.txt" && -f "$REPORT_DIR/metrics.json" ]] || fail "Probe did not create both report files at $REPORT_DIR."
{
  echo "Run ID: $RUN_ID"
  echo "Status: probe completed; visual checkpoint review required"
  echo "Probe report: $REPORT_DIR/run_report.txt"
  echo "Probe metrics: $REPORT_DIR/metrics.json"
  echo "Log: $LOG_FILE"
  echo "Party creation is not automatically verified; inspect the candidate screenshot."
} > "$SUMMARY_FILE"
echo "Execution summary: $SUMMARY_FILE"
