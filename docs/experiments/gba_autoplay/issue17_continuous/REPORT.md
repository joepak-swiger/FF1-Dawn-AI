# Issue #17 — continuous GBA autoplay sessions

## Result

Five bounded sessions from the official Sif / Myth / Milo / Seed checkpoint entered and finished 11 distinct Goblin encounters. All 11 displayed a two-frame-confirmed EXP reward; the controller recognized 4 or 6 EXP and 18 or 24 Gil using reviewed digit templates. No encounter was counted twice, and no unknown outcome was counted as a win. The party HUD showed living party members in the reviewed battle frames. One run displayed a two-frame-confirmed “Sif gained a level!” banner.

Six of the eleven victories had a reviewed postbattle map view and a matched NOOP / direction / opposite-direction image probe. In those cases idle and repeated controls were stable, directional views differed from idle and from each other, and the controller executed movement on the same emulator branch before resuming random exploration. This is screen-space movement evidence, not tile coordinates. The other five encounters stopped after their result or at the next unrecognized return view; the controller preserved a proof frame rather than guessing.

The longest session (`20261029_153036_754453Z`, seed 77) recorded three separate victories in one reset, two matched movement resumptions, 6/6/4 EXP, 24/24/18 Gil, and Sif’s level-up banner. It used 3,674 navigation decisions and 4,014 emulator frames, with 16.521 seconds of runner time. Its complete local gameplay GIF and highlights remain at `videos/gba_autoplay/20261029_153036_754453Z/autoplay.gif` and `videos/gba_autoplay/20261029_153036_754453Z/highlights.gif`.

The seed-1729 run `20261029_153654_875430Z` gives a clear two-battle continuation example: battle 1 awarded 4 EXP and 18 Gil; the controller recognized the return view, verified movement with UP versus idle and DOWN, resumed exploration without resetting, then entered and won battle 2 for 6 EXP and 24 Gil. It safely stopped when the second postbattle camera view did not match reviewed anchors. A repeat (`20261029_153823_991272Z`) reproduced the same battle count and result sequence.

## Formation and progress limits

The target-panel screenshots show Goblin groups with counts 3 and 4. This is a verified enemy-count difference, but every encounter was the same Goblin species and the same conservative Attack/target controller. No other enemy type, Magic menu, defeat, or escape was completed. The 11/11 result is therefore a bounded result on a narrow formation family, not a general combat success rate.

Numeric result templates cover only 4 and 6 EXP and 18 and 24 Gil. The result phrase must also match for two consecutive observations. A one-off level-up banner was confirmed, but automated HP, MP, alive/KO, and survival thresholds are not implemented; party HP has only been visually checked in screenshots. Scene recognition remains sparse. The added postbattle map anchors identify selected camera views only, and the controller still stops safely when an unfamiliar view appears.

## Learning and validation

Field movement was seeded-random; combat inputs were scripted. This was not imitation or reinforcement learning. No model was trained, PPO timesteps were zero, and rewards remained neutral. The collected actions are too repetitive and formation-specific for a credible held-out learned policy comparison.

The complete repository test suite passed (107 tests). Python compile checks and `scripts/gba_autoplay.py --dry-run` passed. Five live bounded sessions ran in the existing WSL environment without reinstalling packages. Full GIFs, raw action/event logs, checkpoints, ROMs, and states remain local; only the compact metrics and selected screenshots in this folder are published.

## Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| G0: autonomous recording and repeated known encounter | Pass | Local full GIFs; 11 confirmed results across five runs |
| G1: postbattle map return, movement, and same-session resume | Pass, narrow | Six matched-control movement signals and six resumptions; no tile coordinates |
| G2: second battle attempted in same reset | Pass | Three-battle seed-77 session and two-battle seed-1729 sessions |
| G3: formation/count variation | Partial | Goblin counts 3 and 4 visible; no different species/menu family |
| G4: variable reward/progression signal | Pass, narrow | 4/6 EXP, 18/24 Gil; one Sif level-up banner |
| G5: learned behavior | Not attempted | Insufficient independent battle/menu data; zero training steps |
| G6: watchable local highlights and sanitized GitHub evidence | Pass | Local GIFs plus this report, JSON metrics, and selected proof frames |

## Next gameplay milestone

Make postbattle recovery work across additional camera views without overfitting to static screenshots, then verify a genuinely different enemy species and target-menu layout. Add an independently validated HP/survival signal before attempting a learned combat policy or event-based rewards.

## Selected proof images

- [Three-battle run contact sheet](proof/seed77_three_battle_contact_sheet.png)
- Goblin count screenshots: [four enemies](proof/seed77_goblin_count_4.png), [three enemies](proof/seed77_goblin_count_3.png)
- [Matched movement resumed after battle 1](proof/seed77_postbattle_movement.png)
- [Sif level-up banner](proof/seed77_sif_level_up.png)
- [Seed-1729 two-battle contact sheet](proof/seed1729_two_battle_contact_sheet.png)
- [4 EXP result](proof/seed1729_battle1_earned_4_exp.png), [movement resumed](proof/seed1729_postbattle_movement.png), [second 6 EXP result](proof/seed1729_battle2_earned_6_exp.png)
- Unrecognized camera views that triggered safe stops: [after run-one win](proof/seed1729_postbattle_camera_candidate_1.png), [after run-two win](proof/seed1729_postbattle_camera_candidate_2.png)
