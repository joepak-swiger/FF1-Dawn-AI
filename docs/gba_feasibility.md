# Dawn of Souls GBA feasibility runbook

The GBA experiment is separate from NES training and does not modify the NES
environment or v0.7 model. The default Windows entrypoint is
`run_gba_feasibility.cmd`; double-click it to use the existing WSL virtual
environment, validate the local integration, ROM, and dependencies, then try
the Stable-Retro boot, party-setup macro, and native-core state round trip.
Use `run_gba_feasibility.cmd --dry-run` to validate command forwarding and the
existing GBA runtime without starting the emulator. No packages are installed.

Each invocation writes a unique wrapper log and summary under
`videos/gba_feasibility/logs/<run-id>/`. The Python probe writes its own unique
report directory under `videos/gba_feasibility/<utc-id>/`, including JSON,
text report, native checkpoint, and small screenshots. The launcher prints the
exact report paths. Failed or incomplete work is preserved for diagnosis and
does not overwrite prior runs.

The fixed START/A/SELECT/DOWN macro is only a candidate party-setup route. In
runtime mode, the probe now writes `party_setup_trace.csv` and a small
`party_trace_*.png` after each pressed and released input, including the boot
idle image and per-step changed-pixel fraction. Use the ordered trace to find
the first input where the screen stops progressing or reaches the wrong menu;
do not repeat the macro blindly. A successful state serialization and
deterministic resume do not prove that party creation succeeded. Review
`gba_candidate_party_state.png` visually, then record whether it shows the
expected playable scene. The probe's
`party_creation_verified` and `battery_save_verified` fields remain false until
separately established. Do not train on GBA until repeatable controls, a
visually confirmed post-party checkpoint, reset/reload/resume, and save-state
behavior are confirmed.

The `party_setup_trace.csv` is flushed after every captured input so an
interrupted run retains the diagnostic prefix. The screenshots and trace stay
in the ignored local run directory and must not be committed.

Each trace record also has a `screen_id` based on exact screenshot bytes. The
probe writes `party_setup_trace_summary.json` and compact contact sheet(s) of
the boot image and post-release frames, with action names and per-action pixel
change. Repeated IDs indicate identical saved pixels only; animation timing,
transitions, and gameplay meaning still require interpretation. The probe does
not automatically call a frame a title, name-entry, or playable screen.

The metrics and report separate these milestones: integration boot, input
control evidence, party creation, reaching a playable scene, native state
serialization, state restoration after reset, and correct gameplay resume.
Button mapping/action calls can be partially verified without proving that
the character moves or menu selection advances. A deterministic frame match
proves native serialization/replay only. `gameplay_resumes_correctly` stays
unverified until a playable checkpoint is visually confirmed and an in-game
action succeeds after reload.

For an offline report of available state formats, use the existing Python
probe with no options from the active environment:

```bash
python scripts/gba_feasibility_probe.py
```

Routine Windows use should not require entering WSL or activating `.venv`.
