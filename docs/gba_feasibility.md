# Dawn of Souls GBA feasibility runbook

The GBA experiment is separate from NES training and does not modify the NES
environment or v0.7 model. Double-click `run_gba_feasibility.cmd` to use the
existing WSL virtual environment. `run_gba_feasibility.cmd --dry-run` validates
the runtime and command path without starting the emulator or installing
packages.

Each invocation writes a unique wrapper log and summary under
`videos/gba_feasibility/logs/<run-id>/`. The Python probe writes its own unique
report directory under `videos/gba_feasibility/<utc-id>/`, including JSON,
text report, native checkpoint, and small screenshots. Failed or incomplete
work is preserved and prior runs are not overwritten.

## Party setup and playable checkpoint

Runtime party creation uses `ff1_dawn_ai/gba_party_setup.py` and a bounded
screen-state route. It matches locally observed title, game-selection,
new-game, party-editor, class-picker, and party-confirmation anchors. Unknown
menu states stop the route instead of guessing. Stable-Retro must use
`Actions.ALL`: the custom integration has no valid-action list and default
`FILTERED` mode drops START, which is the party editor's Done input. The route
auto-names each roster row, verifies its name pixels changed, selects Warrior,
Monk, White Mage, and Black Mage by observed picker state, submits with START,
and accepts the preselected Yes prompt with A.

After submission, story advancement uses bounded A taps and only treats a
colorful map as a candidate. The probe saves a native Stable-Retro checkpoint,
reloads it for two equal-length idle controls, then restores it again and
tests UP. A field is called playable only when the movement frame differs
broadly from idle controls. This is direct movement evidence plus visual field
review; it is not a general-purpose screen classifier. Battery-save loading
remains a separate, unverified question.

The probe writes `party_setup_trace.csv`, `party_state_machine_trace.json`, and
small `party_trace_*.png` images after every pressed/released input. Those
diagnostics identify a failed transition. `gameplay_readiness` is derived from
separately passed four-character, playable-field, native-restore, and
post-restore-movement milestones. A completed probe only means execution
finished. The WSL wrapper reads the same metrics and reports unverified
gameplay milestones accurately.

Screenshots, traces, native states, ROMs, and saves remain in ignored local
directories; do not commit them. Publish only reviewed, sanitized summaries
under `docs/experiments/gba_feasibility/<run-id>/`.

## Current result and next check

The automated party route and Cornelia movement test passed in WSL on
2026-10-27. Sanitized evidence is published under
`docs/experiments/gba_feasibility/20261027_212818_932279Z/`. Native checkpoint
restore and post-restore movement passed. Battery-save loading has not been
tested. Repeat the clean-reset route and validate several movement directions
before large-scale training.

## Offline format inspection

With the active Python environment, the probe can inspect local save-state
container metadata without running the emulator:

```bash
python scripts/gba_feasibility_probe.py
```

Routine Windows operation should not require entering WSL or activating
`.venv`.
