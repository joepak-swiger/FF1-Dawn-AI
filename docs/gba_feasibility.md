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
creates the official GBA party Sif (Warrior), Myth (Thief), Milo (Monk), and
Mustard (displayed as Mustar; Red Mage). Names are typed from the actual on-screen keyboard; each
character cursor is checked against a tolerant visual anchor before A selects
it. The class picker's white-hand cursor is read from its four observed option
positions, then checked again before the class is confirmed. The final roster
screenshot is the visible name/class evidence. No auto-name sequence is used.

After submission, story advancement uses bounded A taps and only treats a
colorful map as a candidate. The probe compares two equal-length no-input
controls, then restores the exact candidate state four times and tests UP,
DOWN, LEFT, and RIGHT independently. It requires each resulting frame to
differ broadly from both idle controls. A field is called playable only after
the image is visually reviewed and all four movement checks pass. This is
direct movement evidence, not coordinate tracking or a general-purpose screen
classifier. Battery-save loading remains a separate, unverified question.

Each verified state is kept locally at
`states/gba_custom_party/<run-id>/stable_retro_core.state`; the existing
`states/ff1_after_party*` checkpoints are never replaced. The run also has a
separate ignored report directory under `videos/gba_feasibility/<run-id>/`.

The probe writes `party_setup_trace.csv`, `party_state_machine_trace.json`, and
small `party_trace_*.png` images after every pressed/released input. Those
diagnostics identify a failed transition. `gameplay_readiness` is derived from
separately passed four-character, playable-field, native-restore, and
post-restore-movement milestones. A completed probe only means execution
finished. The WSL wrapper reads the same metrics and reports unverified
gameplay milestones accurately.

For key-grid calibration, `python scripts/gba_name_keyboard_probe.py` starts a
separate fresh game, captures the name keyboard and locally observed letter
cursor positions, and writes only to a unique ignored diagnostics directory.
It does not submit a party or write a battery save.

Screenshots, traces, native states, ROMs, and saves remain in ignored local
directories; do not commit them. For a fully verified custom-party run, create
a reviewed, sanitized summary and small roster/field contact sheet with:

```bash
python scripts/package_gba_party_evidence.py videos/gba_feasibility/<run-id> \
  --repeat-run-id <second-successful-run-id>
```

The packager refuses incomplete milestones, unexpected names/classes, and
existing output paths. It publishes only `REPORT.md`, `metrics.json`, and a
small `proof.png` under `docs/experiments/gba_feasibility/<run-id>/`.

## Current result and next check

Earlier runs and evidence directories `20261027_222448_322832Z` and
`20261027_222507_734422Z` used Seed/Red Mage. They remain intact as historical
experiments but are superseded and do not satisfy the official GBA roster.
Since that historical experiment, two fresh independent sessions
(20261027_224615_320655Z and 20261027_224639_424839Z) passed with the correct
Mustard/Mustar Red Mage roster. Sanitized evidence is published under
docs/experiments/gba_feasibility/20261027_224615_320655Z/. Both sessions
verified party submission, Cornelia, native save/reset/reload, and all four
directions beyond idle controls.

Battery-save loading has not been tested. Next build a reward-neutral
Gymnasium adapter and short gameplay smoke against the corrected checkpoint.

## Offline format inspection

With the active Python environment, the probe can inspect local save-state
container metadata without running the emulator:

```bash
python scripts/gba_feasibility_probe.py
```

Routine Windows operation should not require entering WSL or activating
`.venv`.
