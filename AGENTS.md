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
- Use `scripts/gba_locomotion_probe.py` for bounded one-emulator-frame GBA
  movement, reverse-route, no-op, terrain-candidate, and RAM-differential
  probes; package only reviewed whitelisted output with
  `scripts/package_gba_locomotion_evidence.py`. Its phase-correlation values
  describe rendered map displacement, not tile coordinates. RAM byte offsets
  stay provisional until validated across independent known positions and
  opposite/reversed routes. Do not enable coordinate/visit tracking or rewards
  from these candidates alone; report a blocked tile as unverified unless a
  known obstacle is visually identified and repeated directional input is
  compared with idle and reverse controls.
- Use `scripts/gba_castle_eval.py` to compare straight-north navigation with
  seeded-random, idle, straight-south, and START-menu controls from the
  verified Seed-party checkpoint. Its Castle Cornelia 1F detector matches the
  game's static floor-title banner against the reviewed
  `docs/experiments/gba_castle_goal/reference_cornelia_1f.png` and requires two
  consecutive matches. It is a destination detector only: other screens stay
  `unknown`, and title loss during a transition does not by itself establish a
  field return. Package reviewed results with
  `scripts/package_gba_castle_evidence.py`; keep raw traces, native states,
  ROMs, and full run directories local.
- Raw one-frame scene matches from `CastleCorneliaDetector` are candidates,
  never confirmed events. Use `SceneConfirmationGate` with two distinct
  consecutive emulator observations; reset it on every episode reset. A
  first Castle-entry event is valid only after the exterior field was
  confirmed first. Unknown/fade/mismatching frames clear the match streak.
- `SceneAnchorClassifier` recognizes only its reviewed Cornelia exterior,
  Cornelia town title, Castle Cornelia 1F title, and party/status menu anchors.
  It returns `unknown` elsewhere. The exterior template allows a bounded
  vertical camera shift, not coordinate inference. Keep title/new-game,
  transition, menu, and town negative controls in regression tests; summarize
  results by independent episode rather than correlated frames.
- Run `scripts/gba_scene_eval.py` for the bounded fresh-reset Castle
  entry/exit, seeded-town, and menu checks. The base `FF1GbaEnv` must stay
  reward-neutral. `FirstCastleEntryReward` is a separately constructed,
  disabled-by-default sparse wrapper for only the first confirmed
  exterior-to-Castle event; never use it until the scene gate tests pass.
  `scripts/gba_castle_learning_smoke.py` is capped at 4,096 PPO steps and is a
  feasibility smoke, not proof of learned navigation. Compare against the
  scripted route, random, and idle. Do not commit local PPO weights or raw
  traces; publish compact reports with
  `scripts/package_gba_scene_learning_evidence.py`.
- RAM bytes `0x33FC` and `0x3404` are still provisional. The pair `226/230`
  was observed on both Castle Cornelia 1F and the separate Cornelia town
  screen, so neither byte nor the pair is a unique castle-entry signal.
  Neither RAM candidates nor phase-correlation pixels may be used as rewards.
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

## GBA navigation imitation workflow

- GBA remains primary. Keep `FF1GbaEnv`'s full eleven-button control space
  unchanged. For field navigation use the opt-in
  `NavigationActionEnv` five-action interface (NOOP/UP/DOWN/LEFT/RIGHT), with
  eight held emulator frames and two released frames per decision.
- Use `scripts/gba_navigation_imitation.py --dry-run` to validate config, and
  `--max-decisions 32 --publish` for a bounded live comparison. It starts only
  from the official local Seed-party checkpoint, writes route-derived native
  starts to ignored per-run output, validates the exterior scene and repeated
  native reload, then compares idle, seeded random, always-UP, scripted, and
  nearest-neighbor behavior cloning. Never overwrite the official checkpoint.
- Keep scripted demonstrations distinct from behavior cloning and RL. The
  policy receives rendered observations and demonstration action labels; it
  does not receive route IDs. Always report training-start and held-out results
  separately. A held-out success that also succeeds under always-UP does not
  establish turn generalization. Never call the demonstrator itself learned.
- Current run `20261028_062140_608666Z` confirmed four reproducible Cornelia
  exterior starts. Scripted navigation reached the Castle 4/4, always-UP 2/4,
  random and idle 0/4. Behavior cloning reproduced the two demonstrated starts
  (center and west) and succeeded on one of two held-out starts. It failed the
  held-out east turn challenge, selecting RIGHT for 32 actions; always-UP also
  failed there. South-start behavior cloning succeeded with UP, matching the
  always-UP baseline. Treat turning as ungeneralized. The 48 px
  phase-correlation shift is screen-space only, not a coordinate.
- The current field anchor set includes small reviewed screenshots from west,
  east, and south routes. Preserve title, transition, town, menu, and Castle
  negative controls when adjusting thresholds or anchors. A confirmed field
  scene is still not a location coordinate. No PPO, progress reward, or RAM
  reward is enabled by this milestone.
- Publish only the compact package under
  `docs/experiments/gba_navigation_imitation/<run-id>/`; raw traces, screenshots,
  states, ROMs, and any future model weights remain local.

### GBA navigation generalization update (2026-10-28)

- PR #12 was merged at `c3d8509`. New navigation work branches from current
  `main`; preserve the verified Seed-party state, prior runs, and NES v0.7.
- The current corpus builder creates 11 fresh scene-confirmed start views from
  one official overworld state. Train is seven cardinal route starts,
  validation is east_north/west_north, and final test is east_south/west_south.
  These are route-family splits on the same map, not new maps, seeds, or tile
  coordinates. Scene-anchor reference images have been visually reviewed and
  may classify scene only; never pass anchor labels, route provenance, or start
  IDs into a navigation policy.
- Use `scripts/gba_navigation_imitation.py --max-decisions 48 --replicates 2
  --publish` for matched baseline/BC comparisons. The current verified run is
  `20261028_153708_566851Z`: 7/7 cardinal scripted demonstrations, 127 action
  labels; validation candidates all 0/4; on the two frozen south-diagonal
  families, expanded frame 3-NN, temporal 3-NN, and balanced linear each
  succeeded 4/4, while always-UP, idle, random, and the provenance script were
  0/4. This is narrow supervised turn generalization, not broad autonomous
  exploration. A validation set with no successes selects no model.
- `scripts/gba_navigation_recovery.py --source-run-id
  20261028_153708_566851Z --horizon 48 --repeats 2` performs perturb-and-recover
  tests. Run `20261028_160355_758132Z` succeeded 4/4 at zero perturbation and
  4/4 after one wrong move, 0/4 after two, and 2/4 after four (east-south 2/2,
  west-south 0/2; the latter collapsed to LEFT). This is not DAgger because
  learner-visited states receive no expert labels. Do not infer robustness
  beyond the tested states. The same run confirmed the compound Castle-entry →
  DOWN-to-field return on 10/10 Castle-successful episodes by requiring two
  consecutive reviewed exterior-anchor matches. These reused final-test route
  families and are not fresh generalization evidence.
- The exterior classifier now tries exact matches against reviewed camera-view
  anchors before a narrow vertical-settling fallback. Keep menu, town, Castle,
  title, transition and fade negatives. The Castle gate remains a separate
  two-frame signal. The successful navigation evaluation had zero menu false
  Castle events; no RAM signal or reward was used. The field classifier was
  `unknown` on about 94% of emulator frames during movement, so do not use it
  as a broad field/visit tracker yet; the held reset scene and narrow Castle
  target gate are what were verified.
- Future experiment reports must include train/validation/final-test splits,
  per-policy held-out results and repeated resets, turn action counts, action
  collapse/stalls, and the validation-selected model (or explicitly `none`).
  A post-hoc best test candidate must not be presented as validation-selected.
  Preserve raw data and checkpoints locally; publish only compact sanitized
  evidence. Keep PPO timesteps at zero until fresh route-family and recovery
  tests remain reliable.
