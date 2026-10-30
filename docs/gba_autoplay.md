# Bounded GBA autoplay demo

## Issue #17 continuous-session results (2026-10-29)

The updated `autonomous_session` loop resumes seeded-random cardinal movement after a confirmed reward, a two-frame reviewed postbattle map view, and a matched idle / chosen-direction / opposite-direction control probe. The combat policy remains a fixed Attack/target sequence, not learned behavior.

Across five corrected fresh-reset runs (IDs and per-run metrics in [`issue17_continuous/metrics.json`](experiments/gba_autoplay/issue17_continuous/metrics.json)), the controller entered and finished 11 distinct Goblin encounters, confirmed 11 victories, and recorded zero unknown outcomes. It recognized 4 or 6 EXP and 18 or 24 Gil from reviewed numeric glyph templates. Two manually reviewed target-panel screenshots show Goblin counts of 3 and 4; no other species has been tested. The seed-77 three-battle run verified two movement resumptions; each two-battle seed-1729 run verified one.

The most complete local recording is `videos/gba_autoplay/20261029_153036_754453Z/autoplay.gif` with a 3.9 MB `highlights.gif`: three victories, two movement resumptions, variable 6/6/4 EXP and 24/24/18 Gil, plus a confirmed Sif level-up banner. The repeatable seed-1729 recording at `videos/gba_autoplay/20261029_153654_875430Z/autoplay.gif` shows a confirmed 4 EXP/18 Gil result, matched movement resumption, then a second confirmed 6 EXP/24 Gil victory in the same reset.

Map return anchors are deliberately narrow static compositions. A third reviewed view enabled movement on the first return in the seed-1729 rerun; the next camera view still did not match, so the controller stopped safely after battle two. This is not a broad field detector, location tracker, or coordinate signal. The field classifier still reports most frames unknown. We visually checked the party HUD in proof frames, but did not implement or validate automated HP/MP/KO parsing.

No imitation or PPO training ran (0 optimizer steps); current combat examples all come from the same Goblin family and same scripted action pattern, without held-out formation data. Rewards remain neutral. Full tests: 107 passed; GBA dry-run passed. Runtime experiment detail and test scope are in the linked report. Continue with new visually verified enemy families and robust postbattle field-state recognition before adding event rewards or training.

The default demo starts from the verified Sif / Myth / Milo / Seed checkpoint. The scripted baseline repeats the measured UP-to-Castle / DOWN-to-field route. `autonomous_session` drives seeded-random cardinal movement and can use either bounded scripted Attack/target inputs or the separately selected learned combat model. The navigation controller remains seeded-random.

## One-click run

Double-click `run_gba_autoplay.cmd`. It uses the existing WSL `.venv` and runs one autonomous-session capture bounded at 6,000 decisions or four battles. It writes unique local folders below `videos/gba_autoplay/<run-id>/` and compact safe evidence under `docs/experiments/gba_autoplay/<run-id>/`. Local outputs include a real emulator GIF, contact sheet, report, metrics, action trace, and battle candidate states. The launcher supports `--dry-run`; it never installs packages. Native states, ROMs, raw traces, and full GIFs remain local.

## Earlier single-battle baseline (historical)

Run `20261029_031018_621251Z` was the first single-battle baseline. Its limitations are superseded by the multi-session Issue #17 results above; its original output remains preserved locally.

The battle menu detectors compare the complete 100×50 left panel, including the menu label. Attack-selected and Goblin-target panels were seen across party slots. The reviewed Magic submenu is separately detected; B recovery is allowed only after that menu is confirmed, and the Attack/target panels must be confirmed before combat continues. The multi-session report covers only Goblins; other species and target layouts remain untested.

The reward phrase detectors require two consecutive matches. Separate reviewed digit templates currently identify 4 or 6 EXP and 18 or 24 Gil; unrecognized amounts remain unset. Exact map-view anchors plus matched controls support limited movement resumption, but do not provide a general field classifier or coordinate estimate. Escape, defeat, game-over, HP, and MP are not parsed.

## Current boundaries

- The scripted Castle/exterior route remains independently measured. The random battle scout did not confirm Castle entry.
- Map-view matching is limited to narrow reviewed camera compositions. Matched movement has been demonstrated after six results, but subsequent unfamiliar views can still stop a session.
- The GIF overlay sits below the game's top EXP banner. Event frames are inserted explicitly and capture cadence uses actual action count so the battle, reward messages, and return view are retained.
- Rewards remain neutral and optimizer/PPO timesteps remain zero. Repetitive single-species combat data is not sufficient for a credible held-out learned policy or event reward shaping.
- If `wsl.exe` is inaccessible to Codex, use the one-click launcher on the owner's Windows desktop. Dry-run or synthetic tests are not gameplay proof.

## Issue #19 supervised combat run (2026-10-29)

The first local supervised combat model has now been trained from real EXP-confirmed expert demonstrations and evaluated live. It is behavior cloning, not reinforcement learning. It learned the narrow sequence B from a confirmed Magic panel, then A for Attack and target confirmation. A held-out Magic-menu branch produced one confirmed 6 EXP win under the learned policy and one under the scripted expert; seeded random safely stopped after an unsupported Attack-panel B choice. The learner reproduced the expert without an outcome advantage.

The learned model was also loaded into the full autonomous loop. On the same seed as the scripted run, it entered and won three encounters (6/6/4 EXP), resumed movement twice, and stopped after the third confirmed win on an unrecognized postbattle map view. Its 24 combat choices were attributed to the learned model; the field policy remained seeded-random. Full local GIFs are listed in [`issue19_live/REPORT.md`](experiments/gba_combat_learning/20261029_issue19_live/REPORT.md). The held-out action test had only A labels, so its 100% accuracy is weak evidence. No other species, HP/survival parsing, PPO, or progress reward is validated.

Once `models/gba_combat_bc_issue19.npz` exists locally, double-click `run_gba_learned_autoplay.cmd` to run the learned combat variant. Use `--dry-run` to validate the model and WSL runtime without starting gameplay. The model, full recordings and raw observations remain local.

## Issue #21 — continuous recovery and combat-policy v2 (2026-10-30)

The main learned session `20261030_002302_471698Z` ran for 76.1 seconds from the official checkpoint and entered five encounters. Four Goblin victories were confirmed from exact EXP banners (6/6/4/4 EXP; 24/24/18/18 Gil). The party resumed movement after all four victories, including after the fourth, by selecting the direction that actually differed from matched idle controls. On battle four, the initial direction produced only 0.43 mean pixel difference from idle while its opposite differed by 44.05; the controller safely replayed the moving opposite direction. These are screen-space movement measurements, not tile coordinates.

The first live failure at the fourth battle was isolated using a saved state: the learned and scripted branches both won for 4 EXP, while random was safety-stopped after proposing B on the Attack panel. A bounded retry now occurs only if the exact reviewed Attack panel remains confirmed after the first A; it is attempted once. The next full run recorded one such retry and won battle four. The fifth encounter stopped safely when its target panel was not confirmed. A restored branch of that fifth state settled on the Flee row (attack-template score 0.9485, below the 0.95 gate) after sync, so the safety gate correctly refused to act. Do not loosen the threshold based on this negative case.

The fourth post-victory camera is now a small reviewed map anchor, guarded against battle/result panels and still requiring two consecutive matches plus matched controls. A follow-up 12,000-decision run confirmed four movement resumptions and reached a fifth encounter. The fifth battle outcome remains unknown. Its full local GIF is `videos/gba_autoplay/20261030_002302_471698Z/autoplay.gif`; compact proof is in [`20261030_002302_471698Z`](experiments/gba_autoplay/20261030_002302_471698Z/REPORT.md).

A local v2 supervised model was trained for 150 epochs from 69 deduplicated observations across eight encounter-parent groups; its weights remain ignored. Training contained 33 A and one B example. Grouped splits were train 34, validation 17, test 18; each validation/test split contained one B. The new report records confusion matrices, per-action and per-stage counts/accuracy, and mean max-probability. The test set scored 18/18, including its single B menu-recovery sample; this tiny Goblin-only B sample is weak evidence and is not broad generalization.

On the frozen test-group Magic-menu state, v2 chose B to leave the reviewed Magic panel and then A for Attack/target, winning with exact 6 EXP (10/10 choices attributed to the model). Scripted also won 6 EXP with 10 expert choices; seeded random stopped after one unsupported choice. The previous model also won the same state. Thus v2 makes the verified menu-recovery decision, but it has not outperformed v1 or the scripted controller. On a separate saved Attack-panel state, learned and scripted each confirmed a 4-EXP win; random's B was rejected safely. These comparisons are one-state Goblin branches, not independent species tests. The one updated model's local SHA-256 is `6ba53b1d29573461cab9d71a5e8efaddb3af16ad76de1c4cf3b5df6c61f63f54`.

The safe report and selected proof images are in [`20261030_issue21`](experiments/gba_combat_learning/20261030_issue21/REPORT.md). Learned combat control remains narrow; field movement is seeded-random, enemy species support remains unverified, and no HP/KO/survival parser or trustworthy defeat/escape signal exists. PPO timesteps remain zero and the base Gymnasium reward is neutral. The full WSL suite passed 115 tests, Python compile checks passed, and default plus learned-policy dry-runs passed without starting another emulator.
