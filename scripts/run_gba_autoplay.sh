#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-run}"
if [[ "$MODE" != "run" && "$MODE" != "learned" && "$MODE" != "--dry-run" && "$MODE" != "learned-dry-run" ]]; then
  echo "Usage: $0 [run|learned|--dry-run|learned-dry-run]" >&2
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
MODEL="$PROJECT_ROOT/models/gba_combat_bc_issue19.npz"
if [[ "$MODE" == "learned" || "$MODE" == "learned-dry-run" ]]; then
  [[ -f "$MODEL" ]] || fail "Local trained model is missing: $MODEL. Run the documented training step first."
fi
source "$PROJECT_ROOT/.venv/bin/activate"
"$PYTHON" -c 'import numpy, PIL, gymnasium, stable_retro; import sys; stable_retro.data.Integrations.add_custom_path(sys.argv[1]); assert sys.argv[2] in stable_retro.data.list_games(stable_retro.data.Integrations.ALL)' "$PROJECT_ROOT/custom_integrations" "$GAME" || fail "Existing Python/Stable-Retro runtime or GBA integration validation failed."
echo "Existing runtime and custom GBA integration validated."

if [[ "$MODE" == "--dry-run" || "$MODE" == "learned-dry-run" ]]; then
  "$PYTHON" scripts/gba_autoplay.py --dry-run || fail "Autoplay dependency/checkpoint dry-run failed."
  {
    echo "Launcher run ID: $RUN_ID"
    echo "Status: dry-run passed"
    if [[ "$MODE" == "learned-dry-run" ]]; then echo "Local learned model exists: $MODEL"; fi
    echo "No emulator started; no gameplay evidence was created."
    echo "Log: $LOG_FILE"
  } > "$SUMMARY_FILE"
  echo "Validation summary: $SUMMARY_FILE"
  exit 0
fi

if [[ "$MODE" == "learned" ]]; then
  "$PYTHON" scripts/gba_autoplay.py --policy autonomous_session --combat-policy learned --combat-model "$MODEL" --decisions 6000 --seed 77 --max-battles 4 --max-attackers 4 --resolution-wait-frames 1800 --publish || fail "Bounded learned-combat gameplay session failed; preserve partial output and inspect the run log."
else
  "$PYTHON" scripts/gba_autoplay.py --policy autonomous_session --decisions 6000 --seed 77 --max-battles 4 --max-attackers 4 --resolution-wait-frames 1800 --publish || fail "Bounded autonomous gameplay session failed; preserve partial output and inspect the run log."
fi
{
  echo "Launcher run ID: $RUN_ID"
  echo "Status: bounded autonomous gameplay session completed"
  if [[ "$MODE" == "learned" ]]; then
    echo "Policy: seeded-random cardinal navigation with local learned combat policy and scripted safety gates."
    echo "Model: $MODEL"
  else
    echo "Policy: seeded-random cardinal navigation with scripted Attack/target combat; not learned."
  fi
  echo "The session is capped at 6,000 decisions or four distinct battles and stops on unknown postbattle screens."
  echo "Inspect the printed GIF, metrics, and sanitized evidence paths above."
  echo "Log: $LOG_FILE"
} > "$SUMMARY_FILE"
echo "Execution summary: $SUMMARY_FILE"
