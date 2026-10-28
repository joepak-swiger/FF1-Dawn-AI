# FF1-Dawn-AI engineering instructions

Codex is the project's primary software engineer. Work directly in the
repository: inspect existing state before editing, make reasonable decisions,
implement and debug the requested work, run available relevant tests, and
continue through related documentation and GitHub work without routine
approval requests. Do not ask the project owner to run work that can be
performed or tested here.

## Environment

- Windows checkout: `E:\JOEPAK\FF1-Dawn-AI`
- WSL checkout: `/mnt/e/JOEPAK/FF1-Dawn-AI`
- Runtime: the existing WSL virtual environment, Stable-Retro,
  Stable-Baselines3, and Pygame.
- For the automated three-session RAM capture, direct the project owner to
  double-click `run_nes_ram_capture.cmd`; do not require them to open WSL,
  activate `.venv`, or copy a sequence of commands. `--dry-run` validates the
  configured runtime and assets without starting the emulator.
- The Windows `run_nes_ram_capture.cmd` entrypoint delegates argument
  forwarding to `scripts/run_nes_ram_capture.ps1`, which invokes WSL with
  separate native arguments (no `call`, shell command string, or Bash `-c`).
  Default mode runs a quick NES runtime smoke before capture; `--dry-run`
  validates dependencies and assets without starting the emulator. Bash logs
  each run under a unique ID, validates capture completeness before analysis,
  and preserves partial output. If WSL cannot be reached from this execution
  environment, test Windows argument forwarding and offline Python pieces, and
  report that the real WSL smoke is unavailable; do not change or recreate
  `.venv`.
- All Bash runner scripts must be checked out with LF and PowerShell/CMD
  launchers with CRLF through `.gitattributes`. Before invoking WSL, Windows
  launchers must normalize the known Bash runner files in place to UTF-8 LF
  without BOM so an existing Windows checkout with CRLF can recover itself.
  Test both a deliberately CRLF/BOM fixture and Bash syntax when Git Bash is
  available; never ask the owner to convert files manually.
- If WSL is unavailable, use safe Windows/offline alternatives, write tests
  that can run in WSL, report checks accurately, and continue other useful
  work. Never compromise the working environment to bypass an access limit.

## Repository and experiments

- GitHub is the shared source of truth. After meaningful tested milestones,
  commit and push safe changes when possible; use a feature branch and PR for
  disruptive or uncertain changes.
- Never force-push unrelated work or commit ROMs, save states, secrets, large
  raw experiment dumps, or unapproved model weights.
- Preserve the trusted v0.7 model and existing recordings. Give new experiment
  sessions unique output directories; never overwrite prior evidence.
- Do not start a major PPO training run until game-state signals are reliable.
- Update `docs/STATUS.md` with implementation and evidence status.

## Two-track development

- GBA Dawn of Souls is the PRIMARY development target. NES FF1 remains a
  secondary experiment. Preserve NES v0.7 as the trusted baseline; keep RAM
  signals provisional and do not add unverified RAM rewards or start large
  training runs.
- Official GBA party: Sif (Warrior), Myth (Thief), Milo (Monk), Seed
  (Red Mage). Official NES secondary party: Sif (Warrior), Mustard (White Mage),
  Lock (Black Mage), Milo (Monk). Preserve canonical requested names and any
  platform-specific UI aliases in `ff1_dawn_ai/party_config.py`.
- Keep GBA exploratory code and findings on the dedicated `codex/gba-*`
  branch until reviewed. Do not disturb the NES environment, model, ROM,
  recordings, or save states.
- Make save states directly through the active Stable-Retro core and verify
  reset/reload/resume. Do not repeat speculative mGBA savestate
  conversion or copy guessed battery-save filenames into the real integration.
  Test battery-save loading only in an isolated copy and report whether the
  frontend actually persists libretro save RAM.
- The GBA Gymnasium adapter loads only a verified, run-specific native
  checkpoint and uses neutral rewards until progress signals are validated.
- Use `scripts/gba_eval.py` for repeated checkpoint-reset control baselines.
  It records no-op, seeded-random, cardinal, and physical-button trials,
  reset/step timing, action traces, and conservative image diagnostics. The
  phase-correlation output is screen-space background displacement only, not
  player/world coordinates or tile units. The central color-centroid candidate
  is unreliable and must not be used as progress. Compare all image signals
  against idle controls and manually reviewed screenshots before changing
  rewards.
- Before GBA training, confirm repeatable boot, mapped controls, a visually
  verified playable post-party checkpoint, and native state save/reset/reload/
  resume. Never commit ROMs, saves, native states, or raw captures.
- Verified GBA custom-party checkpoints are stored locally under unique
  `states/gba_custom_party/<run-id>/` paths. The probe must never overwrite
  earlier generic or custom checkpoints. Publish compact evidence with
  `scripts/package_gba_party_evidence.py`; exclude native states and raw traces.

## Reporting

Report changed files, tests actually run, findings, limitations, the commit or
PR link, and the next technical step. Keep reports concise and distinguish
observed results from hypotheses.


## Current two-track and evidence workflow

- GBA is primary; preserve NES v0.7 as the secondary track's trusted baseline. Keep NES RAM signals provisional and do not use them as rewards before independent game-state validation.
- Use `run_gba_feasibility.cmd` (or `--dry-run`) for the parallel GBA track. Do not ask the owner to enter WSL or activate the environment. Keep this feasibility work isolated. The bounded party state machine uses locally observed screen anchors, verifies all four name rows and the party confirmation prompt, then tests a colorful field candidate with reset/reload movement controls. Keep each milestone independent; never infer party/playability from process exit status or a save-state round trip alone.
- For the GBA party state machine, inspect `party_setup_trace.csv`, `party_state_machine_trace.json`, and per-input `party_trace_*.png` when a transition fails. Runtime traces, screenshots, native states, ROMs, and saves remain local and ignored. Stable-Retro must use `Actions.ALL` for this integration: its default filtered mode discards START, the party-editor Done input.
- GBA trace screen IDs are exact image hashes for grouping only; they do not classify screen meaning or prove a stall. A valid native reset/reload frame match does not verify that its checkpoint is playable or that resumed gameplay works.
- GBA screen hashes only match saved menu anchors; unknown screens stop the menu route rather than triggering guessed inputs. The field color cue is only a candidate filter; a playable field requires movement beyond idle controls after native reset/reload. Battery-save loading remains unverified.
- For GBA reports, keep top-level and nested runtime milestones synchronized. `status: completed` means the probe finished; gameplay readiness must remain independently reported and may still be `not_verified`.
- For NES scene verification, use `scripts/nes_scene_audit.py` to select saved frames around pixel-proxy transitions and a few independently reviewed visual anchors. Its layout cues are triage only; they cannot distinguish battle results from active battle or confirm victory, defeat, escape, HP, EXP, location, or game-over state.
- After each completed NES RAM analysis, publish only the sanitized package under `docs/experiments/nes_ram/<run-id>/` using `scripts/package_nes_ram_evidence.py`. Review it before staging. Exclude raw RAM, full analysis CSVs, ROMs, saves, models, videos, full-size captures, and private paths.
- Keep both tracks exploratory until their relevant gameplay-state checks are validated; no unverified RAM reward changes or major training runs.
