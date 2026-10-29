# Bounded GBA autoplay demo

The default demo starts from the verified Sif / Myth / Milo / Seed checkpoint. The scripted baseline repeats the measured UP-to-Castle / DOWN-to-field route. The seeded-random policy drives cardinal movement and can use a bounded Attack and target routine after two-frame battle-menu confirmation. The navigation policy and combat button sequences are scripted/stochastic controls, not a learned policy.

## One-click run

Double-click `run_gba_autoplay.cmd`. It uses the existing WSL `.venv`, records a scripted Castle excursion and seeded encounter scout in unique local folders below `videos/gba_autoplay/<run-id>/`, and writes compact safe evidence under `docs/experiments/gba_autoplay/<run-id>/`. Local outputs include a real emulator GIF, contact sheet, report, metrics, action trace, and battle candidate states. The launcher supports `--dry-run`; it never installs packages. Native states, ROMs, raw traces, and full GIFs remain local.

## Verified gameplay evidence

Run `20261029_031018_621251Z` loaded the official checkpoint reproducibly, reached a Goblin battle at decision 471, confirmed four Attack/target menu sequences (eight combat A presses), and displayed the exact `Earned 6 EXP.` banner after 742 neutral frames. It then acknowledged the `Obtained 24 gil.` message and matched a two-frame Cornelia-area map anchor. The result frame showed Sif at 31/35 HP, Myth at 30/30, Milo at 32/33, and Seed at 30/30. The full local GIF includes the battle, reward banners, and map view at `videos/gba_autoplay/20261029_031018_621251Z/autoplay.gif`.

The battle menu detectors compare the complete 100×50 left panel, including the menu label. Attack-selected and Goblin-target panels were seen across party slots. A Magic submenu from another seed is a negative control; the narrowed detectors do not treat it as Attack or Goblin targeting. Two of three fresh seeds (77 and 1729) confirmed the exact 6-EXP and 24-gil banners. Seed 314 stopped after one A input when its Magic submenu did not match the Goblin target panel. This is not a general encounter success rate: all three runs only cover the observed Goblin formation.

The reward detectors require two consecutive matches to reviewed 240×16 `Earned 6 EXP.` and `Obtained 24 gil.` banners. A separate two-frame image anchor recognized one Cornelia-area map composition after reward messages were acknowledged in seed 77, but not seed 1729. This is not a general field classifier, coordinate estimate, or movement signal. These detectors do not parse other EXP/gil amounts, escape, defeat, game-over, HP, or level.

## Current boundaries

- The scripted Castle/exterior route remains independently measured. The random battle scout did not confirm Castle entry.
- The one map-view return after a battle is a narrow composition match; it confirms only that view after rewards. General field movement after that return still needs a controlled repeatable check.
- The GIF overlay sits below the game's top EXP banner. Event frames are inserted explicitly and capture cadence uses actual action count so the battle, reward messages, and return view are retained.
- Rewards remain neutral and optimizer/PPO timesteps remain zero. One Goblin result signal is insufficient for training or event reward shaping.
- If `wsl.exe` is inaccessible to Codex, use the one-click launcher on the owner's Windows desktop. Dry-run or synthetic tests are not gameplay proof.