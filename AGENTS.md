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
- The Windows launcher calls `scripts/run_nes_ram_experiment.sh` through WSL.
  It logs every run, validates completed capture outputs before analysis, and
  uses distinct output directories. If WSL cannot be reached from this
  execution environment, test the offline Python pieces and report that the
  live launcher check was unavailable; do not change or recreate `.venv`.
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

## Reporting

Report changed files, tests actually run, findings, limitations, the commit or
PR link, and the next technical step. Keep reports concise and distinguish
observed results from hypotheses.
