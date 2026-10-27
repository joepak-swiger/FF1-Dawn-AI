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

- Track A is NES FF1. Preserve v0.7 as the trusted baseline. Keep RAM signals
  provisional until independently checked; do not add unverified RAM rewards
  or start large training runs.
- Track B is GBA Dawn of Souls feasibility. Keep exploratory code and findings
  on a dedicated `codex/gba-*` branch until reviewed. Do not disturb the NES
  environment, model, ROM, recordings, or save states.
- Prefer making save states directly through the active Stable-Retro core and
  verifying reset/reload/resume. Do not repeat speculative mGBA savestate
  conversion or copy guessed battery-save filenames into the real integration.
  Test battery-save loading only in an isolated copy and report whether the
  frontend actually persists libretro save RAM.
- Before GBA training, confirm repeatable boot, mapped controls, a visually
  verified playable post-party checkpoint, and native state save/reset/reload/
  resume. Never commit ROMs, saves, native states, or raw captures.

## Reporting

Report changed files, tests actually run, findings, limitations, the commit or
PR link, and the next technical step. Keep reports concise and distinguish
observed results from hypotheses.


## Current two-track and evidence workflow

- Preserve NES v0.7 as the trusted baseline; keep RAM signals provisional and do not use them as rewards before independent game-state validation.
- Use `run_gba_feasibility.cmd` (or `--dry-run`) for the parallel GBA track. Do not ask the owner to enter WSL or activate the environment. Keep this feasibility work isolated; party setup requires visual confirmation before training.
- For the GBA party macro, inspect `party_setup_trace.csv` and its per-input `party_trace_*.png` sequence to identify the exact first failing transition. Runtime traces, screenshots, native states, ROMs, and saves remain local and ignored.
- GBA trace screen IDs are exact image hashes for grouping only; they do not classify screen meaning or prove a stall. A valid native reset/reload frame match does not verify that its checkpoint is playable or that resumed gameplay works.
- The GBA START/A macro stops when a completed pair repeats an exact earlier pair-boundary frame. Treat this as a conservative loop guard only; inspect the screenshot trace and never report the loop signal as proof of a stall or party creation.
- For GBA reports, keep top-level and nested runtime milestones synchronized. `status: completed` means the probe finished; gameplay readiness must remain independently reported and may still be `not_verified`.
- For NES scene verification, use `scripts/nes_scene_audit.py` to select saved frames around pixel-proxy transitions and a few independently reviewed visual anchors. Its layout cues are triage only; they cannot distinguish battle results from active battle or confirm victory, defeat, escape, HP, EXP, location, or game-over state.
- After each completed NES RAM analysis, publish only the sanitized package under `docs/experiments/nes_ram/<run-id>/` using `scripts/package_nes_ram_evidence.py`. Review it before staging. Exclude raw RAM, full analysis CSVs, ROMs, saves, models, videos, full-size captures, and private paths.
- Keep both tracks exploratory until their relevant gameplay-state checks are validated; no unverified RAM reward changes or major training runs.
