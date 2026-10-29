# FF1 Dawn AI

An experiment to train an AI agent to play Final Fantasy I & II: Dawn of Souls for Game Boy Advance using Stable-Retro and Stable-Baselines3.

## Goal

Start after party creation in Final Fantasy I and train an agent to:

1. Move around safely
2. Leave Cornelia
3. Survive basic battles
4. Reach the Chaos Shrine
5. Eventually fight and defeat Garland

## GBA autoplay demonstration

Double-click [`run_gba_autoplay.cmd`](run_gba_autoplay.cmd) to run the bounded
scripted Cornelia excursion and seeded autonomous session from the official
local Seed-party checkpoint. Run GIFs, traces, and native states stay under the
ignored `videos/gba_autoplay/`; sanitized summaries and selected proof frames
are published under `docs/experiments/gba_autoplay/`. The continuous controller
has completed repeated Goblin encounters and resumed movement after verified
postbattle map views. Combat remains scripted and formation-specific, field
movement is seeded-random, and neither is learned. It currently stops safely on
unrecognized postbattle camera views. No PPO training or non-neutral rewards
are enabled. See [`docs/gba_autoplay.md`](docs/gba_autoplay.md) and the
[Issue #17 session summary](docs/experiments/gba_autoplay/issue17_continuous/REPORT.md).

## Experimental learned combat

The first local supervised combat policy is a narrow imitation baseline for
reviewed Goblin menus. Once its local model exists at
`models/gba_combat_bc_issue19.npz`, double-click
[`run_gba_learned_autoplay.cmd`](run_gba_learned_autoplay.cmd) to run it in the
bounded autonomous session. Use `--dry-run` to validate the WSL runtime and
local model without opening the emulator. The full learned recording and model
stay local; sanitized results are published by the runner. The model has
matched, but not outperformed, the scripted combat baseline and has not been
validated on other enemy species. See
[`docs/gba_combat_learning.md`](docs/gba_combat_learning.md) and the
[Issue #19 live report](docs/experiments/gba_combat_learning/20261029_issue19_live/REPORT.md).

## Important

ROMs, save states, videos, training runs, and model checkpoints are intentionally not committed to GitHub.

## Public release / ROM notice

This repository does not include game ROMs, save files, save states, trained model files, checkpoints, generated videos, or GIFs.

You must provide your own legally obtained ROM. Local ROMs, save files, model outputs, checkpoints, and generated media are intentionally ignored by Git.
