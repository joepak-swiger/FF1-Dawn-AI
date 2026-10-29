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
OUT="$PROJECT_ROOT/videos/gba_autoplay"
LOG_ROOT="$OUT/logs"
mkdir -p "$LOG_ROOT"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${RANDOM}_$$"
while [[ -e "$LOG_ROOT/$RUN_ID" ]]; do RUN_ID="$(date -u +%Y%m%d_%H%M%S)_${RANDOM}_$$"; done
mkdir "$LOG_ROOT/$RUN_ID"
LOG_FILE="$LOG_ROOT/$RUN_ID/execution.log"
SUMMARY_FILE="$LOG_ROOT/$RUN_ID/execution_summary.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "FF1 Dawn of Souls autonomous gameplay capture"
echo "Launcher run ID: $RUN_ID"
echo "Project: $PROJECT_ROOT"
echo "Mode: $MODE"
echo "Execution log: $LOG_FILE"

fail() {
  local message="$1"
  echo "ERROR: $message" >&2
  {
    echo "Launcher run ID: $RUN_ID"
    echo "Status: failed"
    echo "Failure: $message"
    echo "Log: $LOG_FILE"
  } > "$SUMMARY_FILE"
  echo "Execution summary: $SUMMARY_FILE"
  exit 1
}

[[ -x "$PYTHON" && -f "$PROJECT_ROOT/.venv/bin/activate" ]] || fail "Existing WSL .venv is unavailable; no packages were installed."
[[ -f "$ROM" ]] || fail "Local GBA ROM is missing: $ROM"
[[ -f "$PROJECT_ROOT/scripts/gba_autoplay.py" ]] || fail "Autoplay runner is missing."
source "$PROJECT_ROOT/.venv/bin/activate"
"$PYTHON" -c 'import numpy, PIL, gymnasium, stable_retro; import sys; stable_retro.data.Integrations.add_custom_path(sys.argv[1]); assert sys.argv[2] in stable_retro.data.list_games(stable_retro.data.Integrations.ALL)' "$PROJECT_ROOT/custom_integrations" "$GAME" || fail "Existing Python/Stable-Retro runtime or GBA integration validation failed."
echo "Existing runtime and custom GBA integration validated."

if [[ "$MODE" == "--dry-run" ]]; then
  "$PYTHON" scripts/gba_autoplay.py --dry-run || fail "Autoplay dependency/checkpoint dry-run failed."
  {
    echo "Launcher run ID: $RUN_ID"
    echo "Status: dry-run passed"
    echo "No emulator started; no gameplay evidence was created."
    echo "Log: $LOG_FILE"
  } > "$SUMMARY_FILE"
  echo "Validation summary: $SUMMARY_FILE"
  exit 0
fi

"$PYTHON" scripts/gba_autoplay.py --policy scripted_baseline --decisions 280 --publish || fail "Scripted Castle excursion did not complete; inspect the run output and log."
"$PYTHON" scripts/gba_autoplay.py --policy seeded_random --decisions 1200 --seed 77 --max-attackers 4 --resolution-wait-frames 1800 --attack-on-confirmed-battle --publish || fail "Seeded encounter/combat probe failed; preserve any partial output."
{
  echo "Launcher run ID: $RUN_ID"
  echo "Status: both bounded autoplay captures completed"
  echo "Scripted excursion: inspect the printed GIF and run_report paths above."
  echo "Seeded-random encounter/combat probe: inspect the printed GIF and run_report paths above."
  echo "Combat inputs are bounded to four actor/target confirmations; only the reviewed Earned 6 EXP banner can confirm this specific encounter's victory."
  echo "Log: $LOG_FILE"
} > "$SUMMARY_FILE"
echo "Execution summary: $SUMMARY_FILE"
