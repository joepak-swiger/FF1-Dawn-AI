# Controlled GBA checkpoint evaluation

Run `20261028_005025_703640Z` tested the official Seed-party native checkpoint from `20261028_001816_066107Z`.

- Repeated checkpoint resets: 38; identical observation hashes: True.
- Policies: NOOP, RANDOM, UP, DOWN, LEFT, RIGHT, A, B, START, SELECT, L, R; 3 repeats, 8 actions per trial.
- Completed trials/actions: 38 / 336.
- Validation: {'checkpoint_unchanged': True, 'repeatable_reset': True, 'all_trials_completed': True, 'all_steps_completed': True, 'neutral_rewards_only': True}.
- Pixel-change fractions are only screen-change diagnostics. Phase correlation yields repeatable screen-space map translation in the expected opposite direction to cardinal input, which is a useful movement proxy but not player/world coordinates or tile units.
- NOOP: mean rendered-scene shift (-0.01, -0.00) px.
- UP: mean rendered-scene shift (0.00, 62.04) px; cardinal sign match 100%.
- DOWN: mean rendered-scene shift (-0.01, -47.89) px; cardinal sign match 100%.
- LEFT: mean rendered-scene shift (61.99, -0.05) px; cardinal sign match 100%.
- RIGHT: mean rendered-scene shift (-62.01, 0.03) px; cardinal sign match 100%.
- The gold/red avatar color-centroid marker misses reset frames and is rejected as a progress signal.
- Controlled idle, cardinal, button, and seeded-random trials start from native checkpoint reloads. A prolonged RIGHT-vs-NOOP map-edge probe is reported as a candidate visual plateau only, not confirmed collision evidence.
- Rewards remained zero; no PPO training ran. Local native state, raw action trace, and full screenshots are excluded.

The compact proof sheet samples idle, cardinal movement, the START menu, and the 24-action RIGHT-vs-NOOP edge probe.
