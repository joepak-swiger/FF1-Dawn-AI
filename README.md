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
scripted Cornelia Castle excursion plus a seeded-random movement scout from the
official local Seed-party checkpoint. The generated GIF and complete logs stay
under the ignored `videos/gba_autoplay/` directory; compact sanitized evidence
is prepared under `docs/experiments/gba_autoplay/`. The scripted route is a
reference controller, not a learned policy. Two of three bounded seeded
encounter trials confirmed a Goblin victory through the explicit
`Earned 6 EXP.` result, and one reached a two-frame Cornelia-area map view
after acknowledging EXP and Gil messages. These are narrow references from
one formation; general battle menus and outcomes remain unverified. No PPO
training or non-neutral rewards are enabled.
See [`docs/gba_autoplay.md`](docs/gba_autoplay.md).

## Important

ROMs, save states, videos, training runs, and model checkpoints are intentionally not committed to GitHub.

## Public release / ROM notice

This repository does not include game ROMs, save files, save states, trained model files, checkpoints, generated videos, or GIFs.

You must provide your own legally obtained ROM. Local ROMs, save files, model outputs, checkpoints, and generated media are intentionally ignored by Git.
