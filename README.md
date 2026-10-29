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

## Important

ROMs, save states, videos, training runs, and model checkpoints are intentionally not committed to GitHub.

## Public release / ROM notice

This repository does not include game ROMs, save files, save states, trained model files, checkpoints, generated videos, or GIFs.

You must provide your own legally obtained ROM. Local ROMs, save files, model outputs, checkpoints, and generated media are intentionally ignored by Git.
