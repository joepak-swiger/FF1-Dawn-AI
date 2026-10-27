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
MODEL="$PROJECT_ROOT/models/ff1_nes_ppo_v0_7_best_reported.zip"
ROM="$PROJECT_ROOT/custom_integrations/FinalFantasy-Nes/rom.nes"
STATE="$PROJECT_ROOT/states/ff1_nes_cornelia_start.state"
CAPTURE_ROOT="$PROJECT_ROOT/videos/nes_ram_auto_capture"
ANALYSIS_ROOT="$PROJECT_ROOT/videos/nes_ram_analysis"
LOG_ROOT="$CAPTURE_ROOT/logs"

mkdir -p "$LOG_ROOT"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${RANDOM}_$$"
while [[ -e "$CAPTURE_ROOT/$RUN_ID" || -e "$ANALYSIS_ROOT/$RUN_ID" || -e "$LOG_ROOT/$RUN_ID" ]]; do
  RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${RANDOM}_$$"
done
mkdir "$LOG_ROOT/$RUN_ID"
LOG_FILE="$LOG_ROOT/$RUN_ID/execution.log"
SUMMARY_FILE="$LOG_ROOT/$RUN_ID/execution_summary.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

CAPTURE_DIR="$CAPTURE_ROOT/$RUN_ID"
ANALYSIS_DIR="$ANALYSIS_ROOT/$RUN_ID"
echo "FF1-Dawn-AI automated RAM experiment"
echo "Run ID: $RUN_ID"
echo "Project: $PROJECT_ROOT"
echo "Log: $LOG_FILE"
echo "Mode: $MODE"

fail() {
  local message="$1"
  echo "ERROR: $message" >&2
  {
    echo "Run ID: $RUN_ID"
    echo "Status: failed"
    echo "Failure: $message"
    echo "Log: $LOG_FILE"
    [[ -d "$CAPTURE_DIR" ]] && echo "Partial capture: $CAPTURE_DIR"
  } > "$SUMMARY_FILE"
  exit 1
}

[[ -f "$PROJECT_ROOT/.venv/bin/activate" && -x "$PYTHON" ]] || fail "Existing WSL virtual environment is missing or incomplete: $PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"
for required in "$MODEL" "$ROM" "$STATE" "$PROJECT_ROOT/scripts/nes_ram_capture.py" "$PROJECT_ROOT/scripts/analyze_nes_ram.py"; do
  [[ -f "$required" ]] || fail "Required project file is missing: $required"
done
"$PYTHON" -c 'import sys, zipfile; archive = zipfile.ZipFile(sys.argv[1]); bad = archive.testzip(); archive.close(); assert bad is None, f"corrupt model member: {bad}"' "$MODEL" || fail "The local v0.7 model archive is invalid."
"$PYTHON" -c 'import numpy, PIL, pygame, stable_retro, stable_baselines3' || fail "One or more existing WSL runtime dependencies failed to import; no packages were installed."
echo "Runtime dependencies and local v0.7 model, ROM, and state are present."

if [[ "$MODE" == "--dry-run" ]]; then
  "$PYTHON" scripts/nes_ram_capture.py --episodes 3 --steps 1800 --run-id "$RUN_ID" --dry-run
  "$PYTHON" scripts/analyze_nes_ram.py --help >/dev/null || fail "Analyzer CLI validation failed"
  echo "Planned capture: $PYTHON scripts/nes_ram_capture.py --episodes 3 --steps 1800 --run-id $RUN_ID"
  echo "Planned analysis: $PYTHON scripts/analyze_nes_ram.py $CAPTURE_DIR --output-dir $ANALYSIS_DIR"
  {
    echo "Run ID: $RUN_ID"
    echo "Status: dry-run passed"
    echo "No emulator was started and no capture data was created."
    echo "Log: $LOG_FILE"
  } > "$SUMMARY_FILE"
  echo "Validation summary: $SUMMARY_FILE"
  exit 0
fi

if ! "$PYTHON" scripts/nes_ram_capture.py --episodes 3 --steps 1800 --run-id "$RUN_ID"; then
  fail "Capture process failed. Partial outputs, if any, were preserved; analysis was skipped."
fi

"$PYTHON" scripts/validate_ram_capture.py "$CAPTURE_DIR" --episodes 3 || fail "Capture output validation failed; analysis was skipped."

if ! "$PYTHON" scripts/analyze_nes_ram.py "$CAPTURE_DIR" --output-dir "$ANALYSIS_DIR"; then
  fail "Offline analysis failed. Complete capture data was preserved."
fi
[[ -f "$ANALYSIS_DIR/run_report.txt" && -f "$ANALYSIS_DIR/metrics.json" ]] || fail "Analyzer did not write its expected report files."
{
  echo "Run ID: $RUN_ID"
  echo "Status: complete"
  echo "Capture sessions: 3 independent resets, 1800 decisions maximum each"
  echo "Capture data: $CAPTURE_DIR"
  echo "Analysis report: $ANALYSIS_DIR/run_report.txt"
  echo "Analysis metrics: $ANALYSIS_DIR/metrics.json"
  echo "Log: $LOG_FILE"
  echo "Labels are pixel heuristics and remain unverified proxies; this run does not validate RAM semantics."
} > "$SUMMARY_FILE"
echo "Experiment summary: $SUMMARY_FILE"
echo "Capture: $CAPTURE_DIR"
echo "Analysis: $ANALYSIS_DIR/run_report.txt"
