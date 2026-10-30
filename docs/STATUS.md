# FF1-Dawn-AI status

**Updated:** 2026-10-30
**Current milestone:** GBA Dawn of Souls is the primary development target.
The latest owner-approved GBA roster is Sif/Warrior, Myth/Thief, Milo/Monk,
Seed/Red Mage. Earlier Seed and Mustar checkpoints, reports, and Gymnasium
smokes remain historical and unmodified. PR #10 is merged. The latest GBA
scene evaluation independently confirmed exterior-to-Castle-to-exterior on
3/3 fresh checkpoint restores, and confirmed the Cornelia town title in 1/3
seeded routes and the party/status menu in its control. The single-frame
matcher now reports candidates only; a two-consecutive-frame gate emits
events only after an exterior was confirmed. An optional one-time Castle-entry
reward wrapper remains separate from the neutral base Gymnasium environment.
A bounded PPO smoke (2,048 CPU timesteps, 23.5 seconds) observed zero training
goal events and reached 0/3 post-training episodes from the same start.
Scripted UP reached 3/3, while random and idle reached 0/3. This is not
evidence of learned navigation.
NES remains secondary; v0.7 and its experiments are unchanged, and RAM
signals remain provisional. Issue #21's latest WSL gameplay experiment entered
five Goblin encounters, confirmed four wins, and verified four post-victory
movement resumptions. The fifth encounter stopped without a confirmed target.
The local v2 combat behavior-cloning model passed one held-out Magic-menu
branch, matching v1 and scripted control; it has not shown a gameplay advantage.
No PPO training or reward changes were made.

## 2026-10-30 — Issue #21 learned-agent v2 and continuous recovery

- In live WSL run `20261030_002302_471698Z`, the learned-combat / seeded-random-field controller entered five encounters, confirmed four Goblin victories (6/6/4/4 EXP; 24/24/18/18 Gil), and resumed movement four times without resetting. The fifth battle stopped safely on an unconfirmed target transition; its outcome is unknown. The full local video is `videos/gba_autoplay/20261030_002302_471698Z/autoplay.gif`.
- Fixed the fourth-battle postbattle blockage with a reviewed camera-map anchor and direction-aware matched control: if one cardinal probe matches idle but its opposite is stable and distinct, the controller resumes using the independently moving direction. One run's RIGHT probe was 0.43 MAE from idle while LEFT was 44.05; the selected LEFT movement was repeated and the session resumed. This is screen-space evidence, not a coordinate or blocked-tile claim.
- Added a one-time Attack retry gated by a two-frame match of the exact Attack panel after an A input. The fresh run recorded one retry and resolved battle four. A restored fifth-battle state showed the Flee row rather than Attack after synchronization (score 0.9485 versus the strict 0.95 threshold); the branch comparator safely refused to act. Keep this threshold and report the state mismatch as an unresolved restoration/timing edge.
- A bounded matched-state branch from a confirmed Goblin Attack panel produced the same 4-EXP victory under learned and scripted control. Random's B proposal was safety-rejected. A second held-out grouped test-state branch began on the reviewed Magic panel: updated v2 chose B then A and won with exact 6 EXP; v1 and scripted also won, while random stopped safely. This shows the narrow B menu-recovery behavior, not a win-rate improvement or new-species generalization.
- Trained a local-only v2 linear behavior-cloning model for 150 epochs from 69 deduplicated real observations / eight encounter-parent groups. Group splits: train 34 (A 33/B 1), validation 17 (A 16/B 1), test 18 (A 17/B 1). Per-action, per-stage, and confusion-matrix diagnostics are now reported; test accuracy was 18/18, including only one B example. v2 did not beat v1 or scripted performance. Model weights and datasets remain local; PPO timesteps are 0 and rewards remain neutral.
- Safe evidence and selected small frames: [`Issue #21 experiment`](experiments/gba_combat_learning/20261030_issue21/REPORT.md) and [`autoplay run`](experiments/gba_autoplay/20261030_002302_471698Z/REPORT.md). The official party/checkpoint hash and NES v0.7 were preserved.
- Validation in the existing WSL virtual environment: **115 tests passed**, compileall passed, default and learned-policy dry-runs passed, and several bounded live state-branch sessions were run. No dependencies were installed.

#### Next Issue #21 milestone

Resolve why a saved fifth encounter restores to the Flee menu after capture, then gather a genuinely new successful non-A combat choice or independently confirmed enemy species/layout. Keep the strict safety gates, neutral rewards, zero PPO timesteps, local-only models/raw data, and NES v0.7 unchanged.

## 2026-10-28 — Repeatable GBA evaluation and reset audit

- Added `scripts/gba_eval.py` to run no-op, seeded-random, all four cardinal,
  and A/B/START/SELECT/L/R controls from the same official Seed-party native
  checkpoint. It stores unique local runs with per-action trace, timings,
  frame samples, reset hashes, and no-op comparisons. Added a separate
  24-action RIGHT-vs-NOOP edge probe; it reports only a candidate plateau,
  never confirmed collision.
- Added deterministic reset/step diagnostics to `FF1GbaEnv` and made the
  restore count/priming frames explicit. Gymnasium checker coverage caught and
  fixed nondeterministic timing/cumulative-frame fields in step `info`.
- Real WSL evaluations `20261028_004926_473980Z` and
  `20261028_005025_703640Z`: each ran 12 controls x 3 repeats x 8 discrete
  actions, plus two 24-action controls; 38 trials, 336 actions, 38 identical
  reset observation hashes, unchanged checkpoint, and only zero rewards.
  Latest mean reset was 0.00477 sec; mean discrete step was 0.00732 sec.
- Three one-restore resets produced two distinct observation hashes. Three
  two-restore resets matched exactly. The evaluator's 36 ordinary resets also
  matched. Keep the double-restore/two-priming-frame behavior; removing the
  first restore is not currently reproducible.
- Directional screenshots and action-vs-idle comparisons show clear
  screen-space map/avatar changes: mean changed-pixel fraction 0.642-0.755
  for cardinal actions versus 0.0287 for no-op. Visual review confirms
  directional movement in Cornelia. Phase correlation independently matched
  scene shifts with the opposite cardinal input sign in 3/3 trials across both
  runs: UP (0.002,+62.042 px), DOWN (-0.008,-47.890 px), LEFT (+61.987,-0.052
  px), RIGHT (-62.010,+0.033 px). The no-op scene shift was (-0.008,-0.002
  px). These are rendered background shifts, not player coordinates or tile
  units; camera transitions and scenery can change the estimates. The
  heuristic gold/red avatar centroid misses the reset frame and is rejected
  as a progress signal. The 24-action RIGHT probe did not reach a stable
  visual plateau; blocked terrain was not confirmed.
- START opened the party/status menu. A, B, SELECT, L, and R produced no obvious
  screen difference from idle over this trial; this only describes these
  controls in the tested field state. Seeded-random produced large scene
  changes but no semantic outcome was inferred.
- Sanitized evidence: `docs/experiments/gba_eval/20261028_005025_703640Z-compact-v2/`.
  It contains a concise report, safe metrics, and a small selected-frame proof
  sheet. Native state, raw traces, full recordings, and private paths remain
  local. The earlier shorter edge-probe attempt remains local and was not
  packaged because its environment truncated it before its advertised horizon.
- Tests after implementation: 49/49 passed in WSL. No PPO training, reward
  changes, or NES changes were made.
- Next: calibrate the scene-translation proxy to map/tile units or derive
  position from independently documented Stable-Retro RAM; confirm a known
  blocked tile and reentry after restore before measuring exploration/progress.

## 2026-10-28 — Official Seed party and GBA Gymnasium smoke

- Read the latest owner correction on PR #6 (`6457061644`): Seed/Red Mage is
  intentional and supersedes Mustard/Mustar for the official GBA profile.
- Updated canonical configuration, tests, evidence validation, checkpoint
  discovery, `AGENTS.md`, and GBA runbook. The NES secondary profile is
  unchanged. Previous Seed and Mustar records are historical and unmodified.
- Fresh independent clean starts `20261028_001756_354424Z` and
  `20261028_001816_066107Z` each entered and visibly displayed Sif/Warrior,
  Myth/Thief, Milo/Monk, Seed/Red Mage; both completed setup and reached the
  Cornelia field. Each saved and restored a unique native state and passed
  UP/DOWN/LEFT/RIGHT movement after restore against idle controls. The on-screen
  roster image confirms the four spellings/classes. Sanitized evidence is at
  `docs/experiments/gba_feasibility/20261028_001756_354424Z/`.
- Gymnasium smoke `20261028_002105_827756Z` loaded the second Seed checkpoint
  and ran two 12-action seeded-random direction episodes. Both reset
  observations and complete observation hashes matched; action counts were
  6/6/6/6 across UP/DOWN/LEFT/RIGHT (24 total); all rewards were 0.0; both
  episodes truncated at step 12; checkpoint remained unchanged; longest exact
  repeated-frame streak was 1. This is a pixel-repeat diagnostic, not a
  semantic stuck/progress detector. Sanitized package:
  `docs/experiments/gba_gymnasium/20261028_002105_827756Z/`.
- Gymnasium `check_env` also passed against the actual Seed checkpoint and
  Stable-Retro core (only the standard alternate-render-mode warning appeared).
- Stable-Baselines3's checker passed on that same checkpoint. A CPU PPO
  `CnnPolicy` initialized and predicted/executed one `LEFT` action with an
  84x84x3 observation and 0.0 reward; no `learn()` call or training step ran.
- The first smoke attempt exposed an outer-loop break on time-limit truncation;
  it ran only one episode and correctly failed the publication gate. The loop
  now treats truncation as an episode boundary and continues to the next clean
  reset. The succeeding two-episode run passed.
- GBA has a reward-neutral AI-compatible environment and repeatable action
  smoke, but no battle, coordinate, HP/EXP, or story reward is validated. No
  PPO training ran. Next build a small evaluation interface and independently
  validate field position/progress signals before introducing rewards.

## What the recordings show

The local v0.9 probe has 901 samples (10,240 RAM bytes per sample): 718
pixel-classified battle-like and 183 field-like. Index `0x0014` is 23 for
99.45% of its field-like samples and 0 for all battle-like samples. Its two
edges are step 110 (`23 -> 0`) and step 829 (`0 -> 23`). Several adjacent or
otherwise distinct candidate bytes show the same two-edge pattern. These are
strong correlations within one rollout, labeled by an imperfect visual
heuristic, not semantic validation.

The successful v1.0 manual recording contains 1,787 action samples: 487 labeled
field, 1,251 battle, and 49 unlabeled, from one encounter. The user labeled
field at step 0 and battle at step 180; the step-180 screenshot is visibly a
battle. Saved frames show field at step 20, a black transition at step 40,
battle at step 60, battle results at step 1,440, and field again at step 1,480.
The label change at step 180 therefore came well after the scene had begun to
change, so action samples before that label switch must not be treated as
verified field examples.

Index `0x0014` changes from 23 to 0 at step 39 during the transition, and from
0 back to 23 at step 1,457 while the label is cleared. After returning to the
map it pulses to 0 at steps 1,512, 1,578, 1,661, and 1,763, then returns to
nonzero within a few samples. Nearby proof frames show Cornelia/town and Inn
menu activity, not a sustained battle. This is direct counterevidence to using
`0x0014 == 0` as a stable battle detector. It may be correlated with scene
transition or other changing state; its meaning is unknown.

Other candidates also split field/battle modes in these captures. For example,
`0x00D2` is usually 73 in field-labeled samples and 201 in battle-labeled
samples; `0x0270` is usually 248 and 240 respectively. Under those modal rules,
the pooled analyzer reports manual-label disagreements of 140/487 field and
35/1,251 battle for `0x00D2`, and 141/487 field and 0/1,251 battle for
`0x0270`. These counts include the known late label change and are not audited
false-positive rates. The bytes remain candidate signals only; the data has
one manually recorded encounter and one historical proxy-labeled rollout.

Across the two recorded sources, `0x00D2` repeats the same distinct modal pair
(field 73, battle 201), while `0x0270` repeats (field 248, battle 240). Their
pooled balanced accuracies are 0.886 and 0.893 respectively, but these
descriptive values still include delayed/manual and pixel-label errors.
`0x0014` does not repeat a distinct field/battle pair: its manually labeled
field mode equals its battle mode (0). The agreement of neighboring bytes
could reflect common scene data, so none is yet a verified semantic flag.

## Tooling added

- `ff1_dawn_ai/ram_analysis.py` and `scripts/analyze_nes_ram.py` load v0.9 NPZ
  captures and v1 manual CSVs, group sessions separately, summarize
  label-conditioned values, record candidate transitions, report label
  disagreements and ambiguous samples, and write a concise report plus JSON/CSV.
- `scripts/nes_ram_capture.py` records multiple independent Cornelia resets
  headlessly using the existing v0.7 PPO checkpoint. It does not train or alter
  weights. Labels are generated by the independent pixel-only heuristic, and
  periodic frames are saved for targeted human review.
- `scripts/nes_ram_verify_v1_0.py` now creates a unique timestamped session
  directory and refuses to overwrite one. The original root-level v1.0 output
  remains intact.
- Root `AGENTS.md` preserves the project's engineering, WSL, GitHub, data
  hygiene, and reporting instructions.

## Validation and limitations

Seven tests pass in the available bundled Windows Python runtime, including
synthetic v0.9/v1 loaders, session-separated transitions, and the actual local
v0.9/v1 recordings. The analyzer completed a two-session run over 2,688
samples and wrote a local report under the ignored `videos/nes_ram_analysis/`
directory. Python syntax compilation and Git whitespace checks also pass.

The emulator capture itself was not run from this execution environment:
Windows Python here lacks Stable-Retro/Pygame and this session cannot access
WSL. The existing v0.7 model was left unchanged. No ROM, save state, model, or
raw capture bundle is added to Git. Automated pixel labels still need selective
human review because transitions, town/menu screens, and visual ambiguity can
confuse the heuristic.

## Next step

Run a new 3-session headless capture in the working WSL environment with the
v0.7 baseline, then analyze the resulting session tree and compare whether
`0x00D2`, `0x0270`, and nearby bytes repeat the entry/exit edges without
conflicting during field, town/menu, and result screens. Review only frames
around candidate/visual transitions. Do not use `0x0014` in rewards or launch a
new training run yet.

## 2026-10-27 — WSL one-click capture and three-session result

- Found and analyzed local automated capture `20261027_003906_8f01341e`:
  three independent resets, 1,801 RAM samples each (5,403 total), 10,240 RAM
  bytes per sample, and 30 pixel-heuristic transitions. The run metadata records
  the trusted v0.7 model checksum and says it was not modified.
- Combined offline analysis of v0.9, manual v1.0, and the three automated
  sessions covered 8,091 samples in five recorded sessions. With a pooled
  modal-value classifier, `0x0014` was 23 for 1,527/1,982 field-labeled
  samples and 0 for 6,058/6,060 battle-labeled samples; its aggregate balanced
  accuracy was 0.953. The new auto-capture alone gave 23 for 1,280/1,312 field
  proxy samples and 0 for 4,089/4,091 battle proxy samples. These labels are
  screen-pixel heuristics, not independently confirmed battle states. Across
  the combined sources the same `0x0014` mode pair did not hold across all
  sessions, and the manual recording included 49 unlabeled samples. It remains
  provisional and must not drive rewards or state decisions.
- In the combined proxy-labeled set, `0x0270` used modes 248 (field) and 240
  (battle), with 0.927 balanced accuracy; `0x00D2` used 73 and 201, with
  0.926. Their false-negative counts were 387 and 450 respectively under the
  pooled modal rule. They remain candidates only; pixel label errors can
  account for some apparent disagreements.
- Added `run_nes_ram_capture.cmd` and its WSL runner. The default action checks
  installed dependencies and local model/ROM/state, runs three 1,800-decision
  resets, verifies capture completeness, then analyzes only that new run.
  Logs and a concise summary are saved under a unique ID. `--dry-run` validates
  setup without launching the emulator. Capture failures preserve partial
  evidence and skip analysis; the v0.7 model and reward code are untouched.
- Current offline workspace could not reach WSL (`Wsl/E_ACCESSDENIED`), so the
  new WSL dry-run and live capture were not executed here. The existing local
  three-session capture was analyzed successfully; Python/synthetic validation
  tests are run as part of this change.

## 2026-10-27 — Windows launcher forwarding fix

- Fixed the CMD-to-WSL call to invoke `bash` with the runner script as a
  separate argument, removing the `bash -lc` command string that led Windows
  to interpret `exec` as a CMD command. The default `run` and explicit
  `--dry-run` arguments are forwarded unchanged.
- Added a Windows PowerShell regression test with a mock `wsl.exe` wrapper. It
  checks WSL path conversion, both modes, invalid arguments, and preservation
  of the runner's exit code. The mock verifies command construction without
  starting WSL or the emulator.
- No capture or analysis was run for this fix. A live WSL invocation still
  needs confirmation on the project PC; the test here covers Windows argument
  handling through the simulated WSL boundary.

## 2026-10-27 — Launcher process-boundary fix and runtime gate

- Replaced the CMD-to-WSL argument handoff with a PowerShell entrypoint that
  passes `--exec`, `bash`, the Linux runner path, and `run`/`--dry-run` as
  distinct native arguments. The CMD file no longer uses `call` for WSL.
- The normal run now performs a short NES core, input, RAM, and in-memory
  native-state round-trip smoke before it starts the longer capture. A smoke
  failure stops capture and leaves a diagnostic JSON report under that run's
  unique log directory. Dry-run remains emulator-free.
- Windows `cmd.exe` regression test passed for default mode, dry-run, invalid
  arguments, and WSL exit-code propagation. It records the argument vector
  received by a simulated WSL wrapper; it is not an end-to-end WSL test.
- A real `wsl.exe --exec bash -c ...` check from this Codex environment was
  denied with `Wsl/E_ACCESSDENIED`. No live NES smoke, capture, or analysis was
  run. The project owner's prior recording and all model/ROM/state assets were
  preserved.

## 2026-10-27 — Two-track plan and GBA feasibility spike

- Project scope now has two tracks: continue the NES FF1 pipeline with v0.7
  protected as the trusted baseline, and investigate Dawn of Souls GBA on the
  isolated `codex/gba-feasibility` branch. Neither track has started training.
- Read GitHub Issue #1 and reviewed the local integration, historical scripts,
  and available state artifacts. The old battery probe copied a save under
  several guessed filenames into the integration directory, so it is not
  treated as reliable proof of battery-save support.
- A new read-only format inspector examined the local artifacts: both mGBA
  `.ss0` samples are PNG containers with `gbAs`/`gbAx` chunks; the Stable-Retro
  `.state` sample is 397,312 bytes with a different binary header; the battery
  save is 32,768 bytes. The old external-state screenshots show the Square
  Enix boot logo after import. This establishes an observed format mismatch
  and failed import result, but not the exact technical cause.
- Added `scripts/gba_feasibility_probe.py`. Its optional runtime path boots the
  registered integration, exercises START/A, can replay the previously tried
  party-setup macro, creates a checkpoint using the active Stable-Retro core,
  resets, reloads, and checks deterministic resumed frames. It writes to a
  unique ignored `videos/gba_feasibility/` run directory and never edits the
  integration. `--automate-party-setup` produces a candidate screenshot/state;
  visual review is still required before calling it playable.
- Upstream Stable-Retro's current `emulator.cpp` handles system RAM and native
  serialization but contains no `GET_SAVE_DIRECTORY` or `SAVE_RAM` handling;
  upstream mGBA documentation expects battery saves in the frontend's save
  directory. This points to a frontend persistence gap as a likely reason
  guessed files did not load, but the installed Stable-Retro binary and local
  battery-load behavior were not run here.
- Offline unit tests pass (14 total), Python syntax compilation passes, and the
  static inspector completed against four local artifacts. WSL still returns
  `Wsl/E_ACCESSDENIED`, so GBA boot, the new native-core round trip, party
  setup, and battery-save loading remain unverified. No ROM/save/state/model
  assets were staged or changed.

### GBA assessment and next gate

**No-go for GBA training yet.** Native-core state generation and deterministic
resume are the most promising next experiment because they avoid cross-emulator
state conversion. Battery-save persistence remains unresolved. Run
`python scripts/gba_feasibility_probe.py --runtime --automate-party-setup` in
the existing WSL environment once, then inspect only the generated candidate
screen and report. Require a visually confirmed party/overworld screen and a
passing reset/reload/resume check before approving the checkpoint. Test
battery loading later in a copied integration under the unique output folder;
never copy candidate files into the working integration.

The NES launcher fix is pushed to `main` as `ec7c86e`. Real WSL invocation was
denied from this Codex environment, so the launcher still needs a one-time
double-click check on the project PC before a new NES capture.

## 2026-10-27 — GBA launcher CRLF failure and recovery fix

- The owner reported the GBA launcher failed before emulator startup with Bash
  parsing `set -Eeuo pipefail` as `set: pipefail`. This checkout's raw
  `scripts/run_gba_feasibility.sh` and `scripts/run_nes_ram_experiment.sh`
  bytes had CRLF on every line and no UTF-8 BOM, which is consistent with the
  reported WSL Bash error. Git Bash in this Windows environment accepts a raw
  CRLF fixture, and WSL could not be launched here, so the exact failure is not
  reproduced locally; CRLF is the leading cause, with that platform difference
  still unverified.
- Added `.gitattributes` to enforce LF for Bash and CRLF for CMD and PowerShell
  files. Added a Windows line-ending normalizer to the GBA
  launcher preflight; it repairs only the two Bash runner scripts before WSL
  starts, including pre-existing CRLF/BOM copies.
- Strengthened the Windows launcher test to create an isolated deliberately
  CRLF/BOM-encoded runner, invoke the real CMD/PowerShell launch path with a
  mock WSL executable, and verify the runner is normalized before dispatch.
  It also checks default/dry-run forwarding and error-code propagation, then
  validates syntax and execution with Git Bash where available.
- Expanded GBA diagnostics: every action trace carries an exact-image screen
  ID, changed-pixel fraction, and screenshot; the probe creates a compact
  contact sheet and exact-duplicate summary. Reports now list the seven
  requested milestones independently. Exact frame matching is not presented
  as proof of a playable scene or working resumed gameplay.
- The local GBA launcher run that triggered this task stopped before the
  emulator. No new GBA runtime output exists, and this Codex session still
  cannot access WSL (`Wsl/E_ACCESSDENIED`). The party-setup macro is unchanged
  because there is no new gameplay trace to justify an input-sequence change.
- The repaired launcher was invoked locally in `--dry-run` mode. WSL again
  returned `Wsl/E_ACCESSDENIED` before Bash started; the wrapper reported that
  no experiment log exists and returned Windows exit code 1. Mocked-Wsl tests
  cover argument forwarding and this negative startup-error path.
- Validation completed: 24 Python unit tests passed, both Windows launcher
  regression scripts passed (including a deliberately CRLF/BOM GBA runner),
  Git Bash `-n` passed for both production shell runners, and `git diff
  --check` passed. This validates the repair mechanism, not actual WSL or GBA
  gameplay.
- NES state/reward logic and the v0.7 model are unchanged. No training or
  runtime capture was started. The NES runner receives the same LF protection.

### GBA evidence status

Boot, in-game control effectiveness, party creation, playable scene, native
state serialization, reset restoration, and gameplay resume must be assessed
separately. The current static inspection and old boot-logo screenshots do not
pass any GBA gameplay milestone. The repaired one-click launcher is the next
runtime check; it should preserve a failed trace/report under unique run IDs.

## 2026-10-27 — Independent NES scene review and GBA runtime status

- The local checkout was synchronized to merged PR #3 (`adb1fe2`); prior
  uncommitted work was preserved in a local stash before the fast-forward.
- Inspected `videos/gba_feasibility`: that runtime output directory does not
  exist in this checkout. The latest available GBA output is static artifact
  inspection only (`runtime_requested: false`), listing two PNG-container mGBA
  savestates, a 397,312-byte Stable-Retro state, and a 32,768-byte battery
  save. Thus no current boot, party-creation, input, or native reload/resume
  result can be confirmed here. Older state-import probe screenshots show the
  Square Enix logo, not a playable post-party scene.
- Updated the GBA probe to save a boot-idle image and a pressed/released image
  for each macro input, plus a flushed CSV with step/button and changed-pixel
  fraction. This should pinpoint the first incorrect party-setup transition
  on the next owner-PC run. No runtime was launched here, so it remains
  untested against Stable-Retro.
- Added a RAM-independent targeted NES visual audit and compact evidence under
  `docs/experiments/nes_state_audit/20261027_070449_19924_934/`. It reviewed 44
  transition-near frames from all three sessions. Session 1 step 1680 is a
  title/New Game screen mislabeled battle-like; session 2 step 1360 is a second
  Continue/New Game title screen also mislabeled battle-like. Session 2 step
  1760 has an ambiguous party-panel-only layout. Session 3 steps 1360 and 1440
  show character name entry, which the old classifier also needs to separate
  from active battles.
- These observations further disqualify the pixel proxy as independent truth
  and reinforce that `0x0014`/`0x03CD` are not verified battle flags. The
  visual families in the audit are cautious layout candidates only; victory,
  defeat, escape, game-over, HP, EXP, and location remain unverified. See
  `docs/nes_state_verification.md` for method and next checks.
- No PPO reward, model, training run, ROM, or save-state was modified. The
  GBA go/no-go remains **no-go for training** until the one-click runtime run
  demonstrates repeatable controls, an independently confirmed playable
  post-party checkpoint, and successful native reset/reload/resume.

### Immediate next decision

Run the existing `run_gba_feasibility.cmd` once from the project PC. Its new
per-input trace should identify whether the candidate party macro reaches a
playable scene; inspect that screenshot alongside the native state resume
result. For NES, collect transition-near independent screenshots through a
real battle outcome and field return, then validate candidate HP/EXP/location
signals against those observations. Do not promote RAM candidates or alter
rewards before that evidence exists.

## 2026-10-27 — GBA custom party and repeatable Cornelia checkpoint

- The initial GBA profile used the unauthorized Seed (Red Mage) substitution;
  the secondary NES profile remains Sif/Warrior, Mustard/White Mage,
  Lock/Black Mage, Milo/Monk.
- Replaced automatic naming with character-by-character keyboard navigation.
  Each requested letter is checked against a tolerant visual keyboard cursor
  anchor before it is entered. The selected class is checked from the white
  hand cursor position in the actual 2x2 class picker; each class is then
  confirmed in the roster. The final roster screenshot visibly shows every
  requested name and class.
- Two fresh Stable-Retro runs (`20261027_222448_322832Z` and
  `20261027_222507_734422Z`) completed the same party setup, reached the
  Cornelia field, saved a 397,312-byte native state into run-specific
  `states/gba_custom_party/<run-id>/` directories, and restored it after reset.
  For each run, two equal-length idle trials matched, and UP/DOWN/LEFT/RIGHT
  each caused broad frame changes. The accepted roster and after-UP map were
  visually inspected.
- The earlier generic checkpoints and all previous runs remain unchanged. The
  ROM, full traces, screenshots, and native states remain local and ignored.
  Sanitized findings and a small proof contact sheet are published at
  `docs/experiments/gba_feasibility/20261027_222507_734422Z/`.
- Added a safe evidence packager that refuses unverified milestones, wrong
  roster profiles, missing proof frames, and output overwrite. It emits only a
  report, compact JSON and small contact sheet.
- Review correction: replace Seed with canonical Mustard / in-game Mustar;
  historical Seed states/evidence remain preserved and superseded. Two new
  independent runs verified the corrected party and four-direction restore.
  Evidence: docs/experiments/gba_feasibility/20261027_224615_320655Z/.
  Native checkpoints remain local and ignored.

## 2026-10-27 — Reward-neutral GBA Gymnasium environment

- Added FF1GbaEnv with a verified native checkpoint loaded at every reset,
  discrete NOOP/direction/A/B/START/SELECT/L/R controls, 84x84 grayscale frame
  stacks, neutral rewards, core termination forwarding, and step-limit
  truncation.
- Full WSL unittest suite passed (42 tests). Gymnasium's check_env also passed
  against the actual Stable-Retro core and corrected native checkpoint.
- Two 12-step smokes from the same corrected checkpoint passed with
  deterministic reset observations, zero rewards, truncation at step 12, and
  an unchanged checkpoint hash. Each recorded local traces/metrics/GIFs and
  contact sheets. Sanitized summaries are published at
  docs/experiments/gba_gymnasium/20261027_225340_868798Z/ and
  docs/experiments/gba_gymnasium/20261027_225600_738614Z/.
- Repeatable first-reset output required two identical native state restores,
  each followed by two neutral settling frames; this behavior is encapsulated
  in FF1GbaEnv.reset().
- The smoke verifies control response and Gymnasium mechanics. It does not
  infer coordinates, battle completion, survival, or story progress. No PPO
  training or NES model/reward changes were made.

### Next GBA milestone

Build a short-episode evaluation harness, then validate field movement,
battle outcomes, and party survival against independent evidence before any
training. Keep NES v0.7 untouched.


## 2026-10-27 — Sanitized NES evidence publication and GBA launcher

- Published the successful run `20261027_070449_19924_934` under `docs/experiments/nes_ram/20261027_070449_19924_934/`. It contains a human-readable report, concise metrics JSON, candidate CSV, and a small 11 KB downscaled scene sheet. Raw RAM bundles, full analyzer output, ROMs, save states, model weights, and videos are excluded.
- Across three 1,801-sample sessions, `0x0014` had modal values 23 (field proxy) and 0 (battle proxy) in each session; pooled proxy confusion was 16 false positives and 2 false negatives (balanced accuracy 0.994). All 18 disagreements were within one sample of a pixel-proxy boundary. These are proxy comparisons, not ground truth.
- `0x03CD`-neighbor candidates tied `0x0014`; `0x03D2`'s field mode was inconsistent by session. `0x00D2` had modes 73/201, 7 false positives and 415 false negatives (BA 0.947); `0x0270` had modes 248/240, 20 false positives and 387 false negatives (BA 0.945). Session-level details are in the published JSON.
- Visual review of session 1 step 1680 identified a title/new-game menu mislabeled as battle-like by the pixel classifier; at that frame `0x0014` and `0x03CD` had the purported battle mode. This is evidence against interpreting the pixel labels as truth and keeps every candidate provisional.
- Future successful NES capture pipelines now prepare a whitelisted evidence package automatically. GBA now has a Windows one-click runtime probe launcher with dry-run, WSL argument forwarding, unique logs and summaries; party creation still requires visual review. No training or reward logic changed.
- **Decision:** `0x0014` remains provisional. Next independently review battle-entry, battle-result, field-return, title/menu, and game-over scenes, then compare candidate RAM values with those manual labels. Preserve v0.7 and do not start new training.

## 2026-10-27 — Reviewed successful GBA runtime and guarded menu cycling

- Inspected the complete local run `20261027_203527_887912Z`, including its
  report, metrics, runtime milestones, all 47 action endpoints, both contact
  sheets, candidate screenshot, post-reload screenshot, native state presence,
  and execution log. The screenshots show the boot logo, Dawn of Souls title,
  game selection, FF1 setup, party name/class assignment, and class selection.
  They do not show a completed party or playable overworld.
- The original probe wrote a 397,312-byte native Stable-Retro state and passed
  a deterministic reset/reload/replay frame comparison. The replayed A action
  remained in class selection. State serialization and restore are verified at
  a menu checkpoint; party creation and gameplay resume are not.
- Fixed GBA result reporting so top-level and nested runtime milestone objects
  match and the text report prints the runtime results. `status: completed`
  describes probe execution, while `gameplay_readiness: not_verified` prevents
  treating it as game completion.
- Added an exact-frame pair-boundary cycle guard to the bounded START/A party
  macro. A new one-click WSL run (`20261027_204417_21619_444`) stopped after
  pair 10 repeated pair 8's screenshot, saved 41 trace rows, and again passed
  native state reset/reload. Its candidate screen remains party setup, so the
  guard reduces blind input cycling but does not create a playable checkpoint.
- Published sanitized findings and milestone metrics at
  `docs/experiments/gba_feasibility/20261027_203527_887912Z/`. Local screenshots,
  raw traces, native states, ROMs, and logs remain ignored and uncommitted.
- Full WSL unittest suite: 26 passed. The one-click runtime probe also
  completed successfully. No NES reward/model changes or training were made.

### Immediate next decision

Build a bounded feedback-driven party setup route from the recorded menu
screens. Capture and validate each confirmation/name/class transition, stop on
unrecognized screens, and reach the Cornelia overworld before saving the
candidate checkpoint. After reset/reload, demonstrate an actual field movement.
Keep GBA training disabled; preserve the v0.7 NES baseline and all provisional
NES RAM findings.

## 2026-10-27 — Automated GBA party and Cornelia checkpoint

- The bounded state machine completed a real Stable-Retro run
  (`20261027_213318_29077_440`; report directory
  `20261027_213323_923267Z`). It matched locally observed anchors for Dawn of
  Souls boot logo and title, game selection, FF1 title, new-game prompt, party
  editor, and party confirmation. It advanced the opening story with bounded
  taps rather than repeating a blind setup macro.
- Stable-Retro's default `FILTERED` action mode discarded START because the
  custom integration has no valid-action list. The probe now requests
  `Actions.ALL`; START reached the game's Begin Game confirmation and A
  accepted its preselected Yes option.
- All four roster name regions changed after Auto-Name. The resulting party
  classes were Warrior, Monk, White Mage, and Black Mage. The captured screen
  showed the Cornelia world map/field.
- The probe serialized a 397,312-byte native checkpoint. After reset and
  restore, two equal-length idle observations matched. UP held for 12 frames
  plus 30 frames idle changed 60.4% of pixels, versus 0% between idle
  controls (3% minimum movement threshold). The after-UP screenshot shows the
  map; movement is empirical evidence, while field-screen semantics remain
  based on screenshot review and the movement control comparison.
- Corrected stale readiness fields in JSON, milestones, report, and WSL
  wrapper summary. The fresh run now reports boot, menu/input controls, party
  confirmation, and verified gameplay readiness consistently. Battery-save
  loading remains unverified.
- WSL unittest suite: 33 tests passed. The fresh emulator run completed. A
  sanitized result is published under
  `docs/experiments/gba_feasibility/20261027_213323_923267Z/`; full screenshots,
  trace, native state, ROM, and saves remain local and ignored.

### Next GBA step

Repeat this clean-reset route and compare the milestone summary and movement
proof, then expose the checkpoint as a reproducible evaluation start state and
test multiple field directions. Do not begin large-scale GBA training yet.
NES v0.7 remains untouched.

## 2026-10-28 — Controlled GBA locomotion and RAM differential

- Ran the new headless probe in the existing WSL runtime against the official
  Seed/Red Mage Cornelia checkpoint (`20261028_001816_066107Z`). The native
  checkpoint hash remained unchanged; all 171 reset observations in the run
  matched. The run used three independent resets per condition, one emulator
  frame per control step, and two neutral release frames before RAM sampling.
- UP/DOWN/LEFT/RIGHT pulses at 1, 2, 4, 8, 16, and 32 frames each produced
  repeatable rendered-map shifts with the expected sign on 3/3 repeats. At 64
  frames the DOWN path's scene shift was smaller and phase response weaker,
  consistent with a route/layout boundary or changed scene composition; this
  is not accepted as a blocked-tile result. Contact sheets were visually
  reviewed and show the party moving relative to Cornelia landmarks.
- Reversing 16- and 32-frame vertical and horizontal routes returned map
  registration to within 0.04 pixels of the start across all repeats; matched
  idle controls were within 0.01 pixels. Eight-frame reversals retained about
  14 pixels of screen-map displacement. Image differences after the longer
  reversals were close to idle-control changes, consistent with sprite
  animation/facing rather than continued map displacement. This validates
  repeatable short movement and relative map-view return, not exact player tile
  coordinates.
- The active Stable-Retro `get_ram()` view contains 527,360 bytes. A large
  number of bytes are stable during no-op and respond repeatably to direction;
  this is not a coordinate discovery by itself. Offsets `0x33FC` and `0x3404`
  are provisional examples. At short pulses, their values do not provide two
  clean independent axes (left and right alias at eight frames), and no second
  known origin has been tested. No GBA coordinate signal or visit tracker is
  promoted.
- A 128-frame northbound probe visibly entered Castle Cornelia 1F. Continued
  movement followed a scene transition, so the run did not identify a blocked
  terrain tile. The obstacle milestone remains open.
- Sanitized report, compact metrics, and three small contact sheets are
  published at `docs/experiments/gba_locomotion/20261028_031101_730780Z/`.
  Raw RAM, detailed traces, full captures, ROMs, and native states remain
  ignored locally. Rewards stayed neutral; no training ran. NES v0.7 and its
  experiments were untouched.

### Next GBA decision

Repeat RAM analysis from a second visually established field position, and
target a known wall or water boundary with matched no-op and reverse controls.
Only after coordinate-like values distinguish those origins and survive
reversal should an evaluation-only position/visit tracker be added. Do not
start PPO or add progress rewards yet.

## 2026-10-28 — Castle Cornelia landmark navigation goal

- Read the full PR #9 review (`6459865229`) and confirmed PR #9 is merged at
  `6ea564ab`. Work is based on current `main` in the isolated
  `codex/gba-castle-goal` branch.
- Added `ff1_dawn_ai/gba_landmark.py` and
  `scripts/gba_castle_eval.py`. The conservative detector matches the game's
  static “Castle Cornelia 1F” title region from a previously reviewed frame,
  requires two consecutive matches, ignores RAM and full-screen hashes, and
  leaves every other screen unknown.
- Live WSL evaluation `20261028_052724_624774Z` used the verified
  Seed-party checkpoint `20261028_001816_066107Z`. Three clean scripted UP
  routes reached the detector at the same step: first match 120, two-frame
  confirmation 121 (3/3). Twelve seeded-random 192-action routes, three idle
  trials, and three DOWN controls produced 0 castle detections. The detector
  returned 0 false positives across 3,539 random/idle/DOWN/START observations.
  The START control visually showed a transition at 8/16 frames and the actual
  party/status menu at 32/64 frames; it remained negative throughout.
- A seeded-random counterexample reached the distinct Cornelia town screen at
  step 128. Both provisional RAM bytes were `0x33FC=226` and `0x3404=230`, the
  same pair observed on the Castle screen; this pair occurred on 127
  non-confirmed frames in the full run. Neither byte nor the pair is a unique
  Castle indicator, and none is used by the detector or rewards.
- The optional castle exit route returned visually to the exterior Cornelia
  overworld after a transition (fade at 72 frames, field view at 80). The
  Castle-specific detector intentionally did not automatically certify the
  field screen; the evidence package records this as visual review.
- An 8-frame UP/DOWN reversal showed approximately 14–16 screen pixels of
  residual immediately after reversing. One neutral release frame plus 16
  settling frames reduced residual to about 1 px; two or more release frames
  plus settling returned to within 0.01 px, against near-zero matched idle.
  This supports input/animation timing effects, not a tile-coordinate claim.
  The 64-frame DOWN scene shift plateaued near −47.9 px by frame 56 as the
  player approached/crossed the southern bridge and the camera view reached
  its edge; blocked terrain remains unverified.
- Full WSL unit suite: 59 passed. Python compilation passed. Rewards stayed
  neutral; no PPO run, NES code, NES model, or old experiment was changed.
- Sanitized evidence is published under
  `docs/experiments/gba_castle_goal/20261028_052724_624774Z/` with a compact
  report, metrics, contact sheet, and selected transition/control frames.
  Raw trace, full capture directory, ROM, and native checkpoint remain local.
  Run instructions are in `docs/gba_castle_goal.md`.

### Next GBA decision

Add separately validated field/town/castle screen labels, then use the Castle
title landmark in a short held-out Gymnasium evaluation. Test entry and exit
under more routes and restores, including battle, menu, fade, and unrelated
scene controls. Keep rewards neutral until those labels remain distinct. Do
not begin large PPO training yet.

## 2026-10-28 — GBA scene gate and bounded reward feasibility

- Corrected `CastleCorneliaDetector`: a single frame is now a `candidate`,
  never a confirmed result. `SceneConfirmationGate` needs two consecutive
  matching emulator observations, clears streaks on unknown/other labels, and
  emits a first Castle entry only after a confirmed exterior. Resetting from an
  interior cannot emit that entry event.
- Added reviewed visual anchors for Cornelia exterior field, Cornelia town,
  Castle Cornelia 1F, and the party/status menu. Exterior matching uses three
  independent map details plus a bounded vertical-shift search. Unknown and
  fade frames remain unknown. No RAM values, coordinates, or frame hashes are
  used as scene labels.
- Live WSL run `20261028_060050_480805Z` used the unchanged Seed-party
  checkpoint `20261028_001816_066107Z`. Exterior reset recognition, two-frame
  Castle entry, and two-frame exterior exit all passed 3/3 fresh restores.
  Confirmed Castle entry occurred at action 122; exterior return at action
  205 using one emulator frame per action. A transient one-frame `Cornelia`
  title during travel was not promoted to a town transition. The town title
  was confirmed in 1/3 seeded-random routes; two did not visit it. The
  party/status menu was confirmed and produced zero Castle entry events.
  Historical title-screen, transition, and fade controls remain unknown in
  saved-image regression checks. These counts are episode-level; adjacent
  frames are not treated as independent trials.
- Added `FirstCastleEntryReward`, an opt-in Gymnasium wrapper. The base
  `FF1GbaEnv` remains reward-neutral. It can emit only one reward after two
  confirmed Castle title frames following confirmed exterior; it truncates
  that optional reward episode safely. Idle/menu/town/single-frame tests award
  zero. No speculative RAM or exploration reward was introduced.
- Bounded PPO run `20261028_055426_707460Z`: 2,048 actual CPU timesteps,
  23.496 seconds, 8 emulator frames per decision, and zero goal events during
  collection. Three held-out episodes per policy: scripted UP 3/3 at 16
  decisions, random 0/3, idle 0/3, PPO 0/3. PPO selected A in all 96 decisions.
  This is a failed sparse-reward learning feasibility smoke, not learned
  navigation; the baseline's north route is scripted. The local PPO archive
  and raw traces are excluded from Git.
- Sanitized reports, aggregate metrics, and a small proof set are under
  `docs/experiments/gba_scene_classifier/20261028_060050_480805Z/` and
  `docs/experiments/gba_castle_learning/20261028_055426_707460Z/`.
- Full WSL regression suite: 76 tests passed. `compileall` and dry-run checks
  for the Castle, scene-evaluation, and learning-smoke scripts passed.
  NES v0.7, checkpoints, ROMs, and previous experiment history were not
  changed.

### Next GBA milestone

Improve discovery efficiency with a transparent demonstration/curriculum or
action abstraction, then test on held-out starts/routes. Expand scene
negatives to battles, result screens, game-over, and additional towns before
using these labels for general progress rewards. Do not call the PPO attempt
successful and do not increase the training budget until it receives reward
events and beats seeded-random/idle baselines on more than this one checkpoint.

## 2026-10-28 — GBA navigation action abstraction and imitation feasibility

- Added the opt-in `NavigationActionEnv`, which narrows the unchanged full GBA
  control space to NOOP/UP/DOWN/LEFT/RIGHT macros. A move is held for eight
  emulator frames, then released for two frames; base rewards remain neutral.
- Added `scripts/gba_navigation_imitation.py` to derive new local native starts
  from bounded cardinal routes, confirm Cornelia exterior after reload, record
  successful demonstrations, fit a transparent nearest-neighbor behavior
  cloner, and compare it with idle, seeded random, always-UP, and scripted
  navigation. Each run gets unique ignored output. The official
  `20261028_001816_066107Z` party checkpoint remains unchanged.
- Live run `20261028_062140_608666Z` saved four distinct, reproducible exterior
  starts: center, west, east, and south offsets. Every start passed the
  two-observation exterior gate after native restore. The directionally
  generated views showed about 48 px screen-space map shift; this is not a
  coordinate or tile-position measurement.
- Across one evaluation per start with a 32-decision limit: scripted route
  4/4; always-UP 2/4 (failed west/east); behavior cloning 3/4; seeded random
  0/4; idle 0/4. The clone reproduced the successful center and west
  demonstrations, and reached the held-out south start with UP. It failed the
  held-out east turn challenge, choosing RIGHT for all 32 decisions. Thus this
  is useful behavior cloning on demonstrations, not a generalized turning
  policy. The held-out south success matches always-UP.
- The status-menu control confirmed exterior then menu with zero false Castle
  events. Total runtime was 54.5 seconds, with 31 supervised demonstration
  decisions and zero RL timesteps. No RAM address or progress reward was used.
- Full WSL regression suite: 84 tests passed. Python syntax compilation passed;
  `--dry-run` validated the official checkpoint and action configuration. The
  live emulator comparison completed and published sanitized metrics/report
  and a small contact sheet under
  `docs/experiments/gba_navigation_imitation/20261028_062140_608666Z/`.
  Detailed data and derived native states remain local/ignored.

### Superseding GBA navigation milestone — held-out turning imitation

PR #12 was merged into `main` at `c3d8509`. A multi-start follow-up experiment
and recovery stress test are published under
`docs/experiments/gba_navigation_imitation/20261028_153708_566851Z/` and
`docs/experiments/gba_navigation_recovery/20261028_155424_893532Z/`. They use
the unchanged official Seed-party checkpoint and preserve the NES v0.7 model.

- The corpus generator produced 11 unique, reproducible exterior view states
  from the same checkpoint and map. Every state passed the two-frame field
  gate after native reload; none was rejected. No tile coordinates or new maps
  are claimed. The previously failing camera-offset views became classifiable
  after human visual review and addition of seven small field-only anchor
  references. The direct-match scene-classifier path now avoids the prior
  expensive exhaustive offset scan; its sampled offline throughput was about
  3,037 classifications/second.
- Frozen split: seven cardinal starts for training; `east_north` and
  `west_north` for validation; `east_south` and `west_south` for the final
  turning test. Seven successful scripted cardinal demonstrations supplied
  127 labeled decisions. East and west examples were both included. No final
  test actions or checkpoint provenance were provided to policies.
- Validation failed: every baseline and learned policy scored 0/4 over the two
  north-diagonal starts and two resets. Thus no model was validation-selected.
- On the untouched final south-diagonal set, the original center/west-only
  3-NN scored 2/4 and selected RIGHT for all east-south decisions. Expanded
  frame 3-NN, three-frame temporal 3-NN, and balanced linear imitation each
  scored 4/4 (2/2 on each start). Always-UP, idle, seeded random, and the
  provenance-based scripted baseline each scored 0/4. Each successful expanded
  frame 3-NN episode used four corrective LEFT actions then 17 UP from
  east-south, or four RIGHT then 17 UP from west-south; Castle was confirmed
  at decision 21. This meets the narrow turning-generalization criterion for
  these two route families, not broad exploration or map generalization.
- Menu negative control produced zero false Castle-entry events. The main
  benchmark ran 120 evaluation episodes in 370.5 seconds; supervised dataset:
  127 decisions; RL timesteps: 0. No progress reward or RAM position signal
  was used. Official checkpoint SHA-256 remained
  `2BAD13AA73FDEA33661D39F9D1B83E56D2A0936964703EC4195493E1D4607CFF`.
- The field-anchor classifier returned `unknown` on about 94% of emulator
  frames in moving final-test rollouts. It retained the reset's confirmed
  exterior through uncertain frames and the separate two-frame Castle title
  detector had zero false entries, but this is not adequate coverage for a
  general scene/visit tracker. The script baseline is a provenance-based
  comparator, not an optimized expert on diagonal starts.
- Perturb-and-recover repeated the seven cardinal demonstrations and tested
  two diagonal starts twice at each disturbance level: 0 wrong moves 4/4; 1
  wrong move 4/4; 2 wrong moves 0/4; 4 wrong moves 2/4 (east-south 2/2,
  west-south 0/2). Four wrong west-south moves produced an all-LEFT collapse.
- The updated run `20261028_160355_758132Z` repeated those 16 episodes in 50.7
  seconds and added a bounded compound objective: after each confirmed Castle
  entry, press DOWN and require two consecutive reviewed exterior-anchor
  matches. All 10/10 Castle-successful episodes confirmed field return. These
  are the same final-test route families, not new generalization evidence.
- Published follow-up: `docs/experiments/gba_navigation_recovery/20261028_160355_758132Z/`.
- Full WSL suite: 92 tests passed. Python compilation and checkpoint dry-run
passed. The latest recovery/compound run used 16 episodes and 50.7 seconds. All ROMs, save states,
  native checkpoints, model weights, raw observations and detailed traces stay
  local; sanitized reports and small screenshots are published.

### Next GBA milestone

Collect expert correction traces from fresh perturbed states, then test recovery
on new route families and a blocked-terrain challenge with the two-frame Castle
gate and scene negatives intact. Do not start PPO yet: no policy succeeded on
validation, robustness failed after two wrong actions, and dynamic field
classification remains sparse even though Castle entry and field return were
confirmed for these bounded routes. After repeatable
recovery on unseen starts, extend to Castle entry → Castle exit → exterior, then
Cornelia town entry. The next project priority is still actual FF1 gameplay;
keep NES v0.7 and all prior evidence unchanged.

### GBA recovery audit and north-route feasibility (2026-10-29)

PR #13 is merged at `afb7cf0`; this follow-up is on
`codex/gba-recovery-generalization`. The official party/checkpoint and NES v0.7
were not modified.

- The previous north-diagonal validation failures were route-comparator failures,
  not unreachable goals. A bounded live probe found a repeatable
  `DOWN×4 → inward×6 → UP` route: 4/4 Castle confirmations from the old east and
  west north-diagonal starts (2 resets each). The old 4+4 route failed 0/4.
  These states are exposed and remain validation diagnosis, not new held-out
  evidence. The sanitized route-by-route results are under
  `docs/experiments/gba_route_discovery/20261028_220745_289874Z/`.
- Repeated the online perturb benchmark (`20261029_013908_600790Z`): 0 errors
  4/4, 1 error 4/4, 2 errors 0/4, and 4 errors 2/4 (east 2/2, west 0/2).
  The nonmonotonic 4-error result repeated. Detailed online traces remain local;
  the compact metrics and contact sheet are published under
  `docs/experiments/gba_navigation_recovery/20261029_013908_600790Z/`.
- Recovery run `20261029_014031_526568Z` captured four mistake states after two
  opposite-policy actions, synchronized state restores, and collected 59
  repeatably successful correction labels (two fit states). On those fit
  states, original 3-NN and augmented 1-NN each reached Castle 4/4; augmented
  3-NN regressed to 2/4 and collapsed to DOWN on east (0/2). On decision-6
  timing-heldout states all three scored 4/4 by pressing UP only. Thus the
  augmented policy has not shown measurable recovery improvement and is not
  selected. These are reused route families, not fresh generalization.
- The exact-state harness bug was traced to the GBA framebuffer needing the
  normal two-frame no-op sync after loading. A zero-priming reset gave a stale
  screen; hashing before sync rejected otherwise usable saved states. The
  corrected capture syncs a saved state through two no-op frames, retains the
  rolling observation history, and verifies the resulting frame across repeat
  restores before scoring policy behavior.
- Moving-frame unknowns were mostly classifier `no_match` with fewer
  `uncertain`/transition statuses. This supports the current design: reviewed
  outdoor anchors are sparse fixed camera views; scrolling and sprite/animation
  changes invalidate the crops. Do not convert unknown to field or add a field
  reward.
- Added eight visually reviewed exterior screenshots from novel three/four-leg
  routes as scene-only anchors under `docs/experiments/gba_scene_classifier/`.
  Unit regression preserves title, town, menu, and transition negatives.
  These anchors classify their exact reference views; they have not made the
  moving-frame classifier broad.
- The fresh four-direction benchmark reached start-creation only and stopped
  safely when the two-frame scene gate could not confirm every generated reset
  state. Some candidate screenshots visibly showed the exterior, but subsequent
  observations remained unknown; some routes also varied into unrecognized
  screens. No policy evaluation, model selection, or final-test outcome was
  collected. The failure report is under
  `docs/experiments/gba_navigation_fresh_benchmark/20261029_015018_287848Z/`.
  Those candidate route screens were visually reviewed for scene-anchor work,
  so they are exposed and removed from the fresh benchmark's future final set.
  The script now reserves separate, unpreviewed reversal-loop route IDs for a
  later final run; they must first pass the same reset gate.
- No new frozen route-family test set was evaluated, and no learned
  Castle-entry/exit sequence was established. Existing DOWN-to-field exit is a
  scripted control only. No PPO training, reward changes, RAM position signal,
  or checkpoint modification occurred. Recovery imitation remains supervised
  diagnosis, not RL.

### Next GBA milestone

Resolve why online 2-error episodes fail although policy replay from saved,
synchronized mistake states succeeds; preserve both as separate measurements.
Then improve the moving-scene observer with independent field/battle/transition
evidence. The predeclared fresh benchmark is currently blocked at start
verification, so do not score its unverified starts or relax the two-frame
gate. Once new route starts can be reset and confirmed repeatably, compare
policies on a frozen final set. Keep PPO at zero timesteps until those gates
pass.

### Autonomous gameplay recording push (Issue #15)

- Added `ff1_dawn_ai/gba_autoplay.py`, `scripts/gba_autoplay.py`, and the `run_gba_autoplay.cmd` WSL launcher. The scripted baseline repeats the measured UP-to-Castle / DOWN-to-field excursion. The seeded-random scout drives movement independently and uses a bounded Attack/target routine only after strict two-frame visual menu matches. Both start from and preserve the official Seed-party checkpoint.
- Fresh-reset seed 77 run `20261029_031018_621251Z` reached a Goblin battle at decision 471, confirmed four Attack/target menu sequences (eight combat A presses), and displayed the exact `Earned 6 EXP.` banner after 742 neutral frames. It then acknowledged the `Obtained 24 gil.` message and confirmed a two-frame Cornelia-area map anchor. The result showed Sif 31/35 HP, Myth 30/30, Milo 32/33, Seed 30/30. Local GIF: `videos/gba_autoplay/20261029_031018_621251Z/autoplay.gif`.
- Across three fresh seeds, 77 and 1729 confirmed the exact 6-EXP and 24-gil banners; 314 safely stopped after one A input when the resulting Magic submenu did not match the strict Goblin target panel. This is 2/3 narrow Goblin battle outcomes, not a general win rate. The two-frame Cornelia-area map view matched after rewards in seed 77 but not 1729; this anchor is a single-view resume check, not the broad field detector or a coordinate signal.
- The original small target crop falsely matched a Magic submenu. Battle menus now compare a 100×50 left panel, distinguishing Attack selection, the reviewed Goblin target list, and Magic. The exact result templates require two consecutive matches. Tests include several party-slot positives and field, menu, Castle, and Magic negatives. Other formations, escapes, defeats, HP/EXP parsing, and general field movement remain unverified.
- GIF capture cadence now uses actual action counts rather than fractional trace IDs; event screenshots for EXP/Gil and the map return are inserted directly, and the overlay stays below the top reward banner. Compact sanitized evidence is published in `docs/experiments/gba_autoplay/20261029_031018_621251Z/` and the two additional seed folders.
- Navigation and combat remain scripted/stochastic controls, not learned behavior. No PPO timesteps ran; GBA rewards remain neutral and no RAM reward was enabled. NES v0.7 and all checkpoints remain unchanged.

#### Next gameplay action

Repeat battle-control validation on a different enemy formation, then improve the party-slot command selector and verify movement after the narrow map return against matched idle controls. Only enable a sparse battle-result reward after outcome detection repeats beyond this Goblin formation; keep large PPO training off until then.

### Issue #17 continuous autonomous gameplay (2026-10-29)

- Fixed the encounter confirmation lifecycle to require a current two-frame streak, count battle outcomes per distinct encounter, and avoid duplicate reward inflation. The bounded `autonomous_session` policy now clears per-battle gates only after a two-frame reviewed postbattle map view and repeatable idle / cardinal / opposite-direction screen comparison. It replays the selected movement on the same emulator state and continues field exploration without resetting.
- Across five corrected real WSL sessions (`20261029_152615_993483Z`, `20261029_153036_754453Z`, `20261029_153422_852363Z`, `20261029_153654_875430Z`, `20261029_153823_991272Z`), 11 encounters were entered and finished, with 11 exact EXP-banner victories and no unknown outcomes. Numeric glyph recognition distinguished 4 vs. 6 EXP and 18 vs. 24 Gil. A two-frame Sif level-up banner was confirmed in run `20261029_153036_754453Z`.
- The longest uninterrupted run entered 3 battles and won 3 (seed 77): 6/6/4 EXP, 24/24/18 Gil, 2 confirmed movement resumptions, 3,674 policy decisions, 4,014 emulator frames, and 16.521 s runner runtime. Its local full GIF and highlights are under `videos/gba_autoplay/20261029_153036_754453Z/`.
- Seed-1729 session `20261029_153654_875430Z` won two battles in one reset (4 EXP/18 Gil, then 6 EXP/24 Gil), verified movement after the first, and safely stopped at the second unrecognized return view. A repeat (`20261029_153823_991272Z`) reproduced the same event counts. The aggregate 11/11 is from bounded resets using the same official checkpoint and Goblin species, not broad encounter coverage.
- Proof screenshots show Goblin counts 4/4/3 in the three-battle run; this establishes count variation within the same species only. It does not verify another species or a general combat menu. HUD screenshots visually show party HP, but no automatic HP/MP/alive parser has been implemented. No escape/defeat screen signal is available yet.
- The updated EXP/Gil detector separates phrase confirmation from amount recognition and reports only reviewed glyph templates (4/6 EXP, 18/24 Gil). The narrow map matcher now has additional candidate camera views; these support a few repeated postbattle compositions, not generalized field classification. A different unrecognized camera still safely stops continuation.
- Local emulator outputs and full GIFs remain ignored. Sanitized report and selected proof frames are published under `docs/experiments/gba_autoplay/issue17_continuous/`; per-run packages are under their individual run IDs. Video output is local only.
- Validation: 108 repository unit tests passed; Python compile and `scripts/gba_autoplay.py --dry-run` passed. The Windows one-click launcher's `--dry-run` also passed through its PowerShell/WSL path. Five bounded emulator sessions were run in the existing WSL environment; no packages were installed. PPO/training steps: 0; rewards remain neutral; no learned behavior is claimed.

#### Next GBA gameplay milestone

Extend safe postbattle continuation beyond the few reviewed camera anchors, then test at least one genuinely different enemy species/target-menu layout and add a separately validated HP/survival signal. Keep policy behavior scripted until there are independent formations and held-out states for meaningful imitation evaluation. Do not train PPO or activate event rewards yet.


## 2026-10-29 — First GBA supervised combat learning foundation (experimental)

- Added an opt-in pre-action expert decision recorder to the autoplay runner; only
  actions from independently EXP-confirmed victories are committed to a local
  training manifest. Screen images, native states, trained models and raw data
  remain ignored. Unverified fights cannot silently become positive labels.
- Added a grouped train/validation/test splitter, balanced linear supervised
  imitation trainer, local model serializer and opt-in learned Attack/target
  inference with source-attributed action traces.
- **No GBA model was trained or evaluated live by this change.** Existing
  Goblin actions are almost entirely Attack/target A and do not yet supply
  meaningful diverse action labels. The trainer refuses single-class learning.
  Real WSL training, held-out combat scenarios and learned-vs-scripted video
  comparisons are still required before marking Issue #19 complete.
- Offline standalone contract tests ran on a local prototype (5 passed);
  the complete repo unittest suite and WSL emulator were **not available**
  in this execution environment. The PR must be validated in the owner's
  functioning WSL setup by Codex before merge.
- See \`docs/gba_combat_learning.md\` and Issue #19 for the next live work.

## 2026-10-29 — Issue #19 live combat-learning continuation

- Reviewed and extended PR #20's combat-learning path. The multi-dataset trainer now deduplicates replay examples, supports explicit replay-state-to-original encounter-parent remapping, reports group split IDs, and still refuses single-action training. Battle branch metrics now distinguish policy choices attempted from inputs actually executed; a random choice rejected by the safety gate is counted and reported as rejected.
- Ran the full test suite in the existing WSL environment: **114 tests passed**. Python compile checks and the GBA autoplay dry-run passed. Stable-Retro, Gymnasium, SB3, and the project integration were available; no packages or virtual environment were changed. The optional `run_gba_learned_autoplay.cmd --dry-run` also completed through CMD → PowerShell → WSL, confirming the local model without starting the emulator.
- Real scripted capture `20261029_182321_611128Z`: three distinct battle parent states produced three confirmed victories (6, 6, 4 EXP; 24, 24, 18 Gil), 24 pre-action Attack/target labels, and movement resumed after the first two. After the third win, the controller stopped safely because its postbattle map view did not match reviewed anchors.
- A replay-parent audit found three menu-state branch IDs descended from prior encounter states. They were remapped to their original battle-command parent hashes before the final split. The local dataset had 53 deduplicated observations across six encounter parents: 35 train, 10 validation, 8 test; training labels were 33 A and 2 B. A balanced linear behavior-cloning model was trained locally (150-epoch cap, validation-selected epoch 150; zero RL timesteps). The eight-sample test group contained only A labels, so its 100% accuracy is not a useful estimate of broad action generalization.
- Held-out live branch comparison began at a reviewed Magic-menu state from an encounter parent excluded from optimizer training. Learned chose B to return to the target panel, then A for subsequent legal combat decisions; it won with exact 6 EXP evidence (10/10 learned choices, 812 emulator decisions). Scripted also won from the same state with 10/10 expert choices. Seeded random attempted three choices, executed two, and had one B-on-Attack choice rejected by the safety gate; it stopped without victory. The model reproduced the expert behavior but did not outperform it.
- Full learned autoplay `20261029_182917_704300Z` matched same-seed scripted capture `20261029_182321_611128Z`: each entered and won three battles (6/6/4 EXP), and resumed movement twice. Learned attribution covered all 24 combat choices. Both stopped after the third victory on the same unrecognized map view. Full GIFs and the model/state files remain ignored; compact evidence is under `docs/experiments/gba_combat_learning/20261029_issue19_live/`.
- **Scope:** this is a real trained supervised imitation policy integrated in live play, but it only copies the narrow reviewed Goblin menu behavior. Field exploration remains seeded-random; battle/menu/result recognition remains limited to reviewed Goblin layouts and EXP/Gil templates. No HP/survival parser, other species, general result detector, PPO, reward shaping, or advantage over scripted play has been demonstrated. Keep the model local and preserve neutral rewards and NES v0.7.

#### Next Issue #19 milestone

Collect independent successful demonstrations from a second enemy species and distinct target layouts; validate HP/survival observations; then compare learned and scripted policies on held-out encounter parents before considering online learning or PPO.
