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

### GBA recovery and route-discovery update (2026-10-29)

- PR #13 is merged into `main` at `afb7cf0`. Current recovery work uses the
  separate `codex/gba-recovery-generalization` branch. Preserve all earlier
  run artifacts and the official Seed-party checkpoint.
- The old north-diagonal validation states were reachable. A bounded route
  feasibility probe found `DOWN×4 → inward lateral×6 → UP` and repeated Castle
  confirmation 4/4 (east and west, two resets each); the previous 4+4 route
  failed 0/4. This is a fixed correction route from exposed validation states,
  not new training data or position inference.
- `scripts/gba_route_discovery.py` now tries a stable, bounded variety of
  DOWN-first, horizontal-first, and single-direction plans. Preserve the
  `candidate_index` order for reproducibility and limit live searches to 40
  candidates per start. Route search is feasibility diagnosis only.
- When serializing a learner mistake state, Stable-Retro needs two no-op
  frames after loading to refresh the GBA framebuffer. Capture the expected
  frame after that same two-frame sync and keep the rolling observation history
  (shift it through those frames; do not replace the stack with repeated copies
  of one frame). Reset twice with two priming frames for deterministic replay.
  A zero-priming reset returns a stale framebuffer and is not valid evidence.
- Live recovery run `20261029_014031_526568Z` saved four states after two
  opposite-policy actions, found repeatable correction routes for both
  decision-0 states, and collected 59 labels. Original 3-NN succeeded from
  those saved states 4/4. Augmented 1-NN also succeeded 4/4 but took longer on
  east; augmented 3-NN collapsed to DOWN and failed 0/2 east. At decision-6
  timing-heldout states all models succeeded 4/4 using UP only. Do not promote
  the augmented model or call this a measured improvement.
- The online perturb test was repeated in run `20261029_013908_600790Z`:
  0 errors 4/4, 1 error 4/4, 2 errors 0/4, 4 errors 2/4 (east 2/2, west
  0/2). The difference from saved-state replay remains unresolved. Keep the
  online stress test and exact-state replay as separate measurements.
- Moving-state diagnostics show `no_match` dominates unknown results. Reviewed
  scene anchors intentionally recognize only a few fixed camera compositions;
  scrolling and sprite/animation changes invalidate their map crops. Keep this
  classifier sparse and conservative until a broader detector is tested
  against idle controls and title, transition, menu, town, Castle, and fade
  negatives. No field reward or coordinate tracking is enabled.
- No new route-family final test has yet been completed. Do not use old
  east_south/west_south or north-diagonal results as fresh held-out evidence.
  Do not start PPO, add speculative RAM rewards, or label the Castle exit route
  as learned; existing compound exit evidence is scripted DOWN only.
- `scripts/gba_navigation_fresh_benchmark.py` defines a disjoint four-leg
  train/validation/final split and freezes validation selection before final
  policy evaluation. Run it only when every generated start passes the
  two-frame exterior gate after reset. Run `--dry-run` first. It must stop
  before policy evaluation when any required start is unverified; see the
  incomplete start-audit package under
  `docs/experiments/gba_navigation_fresh_benchmark/20261029_015018_287848Z/`.
- Final route start views are created before validation because their reset and
  scene labels must be checked. Do not inspect policy outcomes, tune the policy,
  or use final routes for training until validation selection is frozen. The
  earlier `*_loop` candidate route families were visually previewed during
  scene-anchor debugging and are exposed; only the newer reversal-loop IDs in
  the current script may be considered for a future final set.
- New field references under `docs/experiments/gba_scene_classifier/` were
  visually reviewed against map terrain, town buildings, shore, and player
  sprites. They classify only the specific reviewed views. Some route creation
  attempts subsequently produced a battle screen or other unrecognized frames;
  never treat one field-looking candidate screenshot as a reproducible reset.
  Preserve the title, town, menu, and transition negative-control regression.
- The fresh benchmark was not scored: three of four route-derived held-out
  starts could not be independently confirmed. This is a scene/route-start
  verification blocker, not evidence that a policy failed. Do not adapt the
  tested families into training or validation and then continue calling them
  held out; define another frozen split after the scene gate is reliable.

### Autonomous GBA gameplay recording

- Use `run_gba_autoplay.cmd` for the one-click bounded autonomous gameplay demo; the default caps at 6,000 decisions or four battles, and `--dry-run` validates without launching gameplay. The script starts only from the discovered verified Seed-party checkpoint and uses unique run folders.
- The `scripted_baseline` is the measured UP-to-Castle / DOWN-to-exterior route. Report it as scripted. Only two-frame reviewed Castle and exterior scene confirmations may trigger route phase changes.
- `autonomous_session` combines seeded-random cardinal field navigation with bounded scripted Attack/target inputs; it is not a learned policy. Battle menu evidence still covers Goblins only (visually observed counts of three and four), not other enemy species or menu layouts. A victory is counted once only after two consecutive reviewed `Earned <amount> EXP.` phrase matches; recognized numeric templates currently cover 4 and 6 EXP, and Gil templates cover 18 and 24. The phrase and amount are separate claims. Other outcomes remain unknown unless independently observed; never infer victory from a screen change.
- After a confirmed victory and Gil message, `CorneliaPostBattleMapDetector` may recognize only reviewed narrow camera-view anchors. Two-frame confirmation plus matched NOOP / direction / opposite-direction controls support a screen-space movement signal; they do not establish world coordinates. The controller resumes only on that evidence and stops safely on an unrecognized result view. Keep unrecognized scenes unknown; do not generalize these anchors into a broad field classifier.
- Full GIFs, action traces, emulator states, ROMs, and raw captures remain local. `--publish` copies compact sanitized metrics, a report, contact sheet, and selected proof frames to `docs/experiments/gba_autoplay/<run-id>/`.
- GBA rewards remain neutral and PPO training remains off. Earlier evidence of two of three Goblin victories is historical; the Issue #17 report below supersedes it for continuous sessions.

### Continuous autoplay update (2026-10-29)

- The corrected autonomous session mode has been exercised in multiple real WSL resets. The five summarized runs under `docs/experiments/gba_autoplay/issue17_continuous/` recorded 11 distinct Goblin victories and no unknown outcomes; two visible Goblin counts (3 and 4) were manually read from target-panel screenshots. This is count variation within one species, not evidence for other formations.
- The stable seed-77 run confirmed 3/3 victories, result amounts 6/6/4 EXP and 24/24/18 Gil, two post-result matched movement resumptions, and Sif's level-up banner. Two seed-1729 runs each confirmed two victories (4/18 then 6/24), resumed movement after the first, then safely stopped on a different unrecognized result view. A separate seed-1729 attempt captured the candidate view but could not resume until that view was added as an anchor.
- Current run recordings and complete raw traces/states are local under `videos/gba_autoplay/<run-id>/`. Only compact, reviewed screenshots and safe metrics are published. The latest sessions' unsupported postbattle camera views remain the blocker to longer continuation.
- No imitation or RL was trained: available positive samples are repetitive Goblin attack sequences from a narrow layout and do not support a credible held-out learned combat policy. Optimizer timesteps remain zero and rewards neutral. Do not claim learning.

### GBA combat imitation update (2026-10-29)

- The Issue #19 continuation trained a local-only supervised combat policy from
  six encounter-parent groups and deployed it in live gameplay. The model learned
  only the narrow reviewed Goblin interaction: B to back out of a verified Magic
  panel, then A for Attack and target confirmation. It is behavior cloning, not
  PPO/RL, and did not outperform the scripted expert.
- The three-encounter learned run matched the same-seed scripted run: 3/3
  confirmed victories (6/6/4 EXP), 2 movement resumptions, then a safe stop on
  the third unrecognized post-battle map view. Do not infer general combat,
  navigation, survival, or species handling from this result.
- The local trainer now accepts multiple `--dataset` directories, deduplicates
  exact replay samples, and keeps encounter-parent groups disjoint. If labels
  come from a derived battle-menu state, map its state SHA to the source
  battle-command-state SHA with `--group-map`; never let sibling replay states
  split across train/test. Public evidence is under
  `docs/experiments/gba_combat_learning/20261029_issue19_live/`. The model,
  source captures, state files and full recordings remain local/ignored.
- The test set contained only A labels, so its 100% action accuracy is weak
  evidence. Keep the held-out Magic-state closed-loop result separate: the model
  chose B and won; scripted also won; seeded random safely stopped. Continue
  collecting independent formations and decision alternatives before describing
  improvement or starting PPO.
- The optional one-click learned combat launcher is `run_gba_learned_autoplay.cmd`. Its `--dry-run` validates the local model and working WSL runtime without opening the emulator. This entrypoint was exercised end to end through CMD, PowerShell, and WSL.

### Issue #23 storage, menu and result-pacing update (2026-10-30)

- Latest local full audit snapshot: 9,608,440,231 bytes / 57,599 files. See
  `docs/project_inventory.md`; full local reports are ignored under
  `reports/storage_audit/<run-id>/`. Run `scripts/audit_project_storage.ps1`
  for read-only audit; `-SafeClean` may remove only the script's explicit
  regenerable source-tree Python caches. Never delete/relocate ignored game
  data, runtime files, recordings, models, checkpoints, ROMs, or states based
  on size, age, or hash matches alone.
- Only source-tree `__pycache__` files were cleaned in this milestone
  (116 files / 1,673,287 bytes). No uncertain artifacts were archived or
  quarantined because no safe replacement/recovery pair was established.
- `scripts/gba_menu_smoke.py` now records completed and failed menu operations
  in one partial report. Inventory is visibly empty at the official checkpoint.
  The equipment path produced an unknown view resembling Sif's Magic list; do
  not claim Equipment, spell, item-use, purchase, or gear-change capability.
  The controller is read-only and fail-closed until fresh row-specific screen
  anchors and observable effects are validated.
- `scripts/gba_autoplay.py` separates post-victory map wait from the longer
  battle-resolution budget via `--postvictory-wait-frames` (default 48). It
  retries only if the exact known Gil banner remains, and unknown views still
  stop safely. On Issue #23 seed-77 live run the existing supervised model
  supplied 24/24 narrow Goblin combat choices; navigation was seeded-random,
  PPO steps zero, rewards neutral. Never present this as learned navigation
  or a broader combat policy. Full recording and local raw traces stay under
  unique ignored `videos/gba_autoplay/<run-id>/` paths.
- WSL suite now includes 122 tests. The new public Issue #23 summaries are in
  `docs/experiments/gba_menu_controller/20261030_issue23/` and
  `docs/experiments/gba_autoplay/issue23_result_ack/`.
