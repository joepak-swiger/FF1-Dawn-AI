# Public Release Notes

This project is an experimental reinforcement-learning project for Final Fantasy I on NES using Stable-Retro and Stable-Baselines3.

## What is included

- Python environment wrappers
- Stable-Retro custom integration metadata
- Training scripts
- Evaluation scripts
- Reward-shaping experiments
- Documentation for the project direction

## What is not included

This repository does not include:

- NES, GBA, GB, or GBC ROM files
- Battery save files
- Emulator save states
- Trained model files
- Checkpoints
- Generated videos or GIFs

You must provide your own legally obtained game ROM.

## Current status

Current milestone:

- v0.1: PPO smoke test works
- v0.2: anti-stuck reward shaping works
- v0.3: CNN PPO experiment added

This is not yet a finished game-playing AI. The agent can load the saved Cornelia start state, move around, enter battles, interact with menus, and train with PPO, but it does not yet complete story objectives like reaching Garland.
