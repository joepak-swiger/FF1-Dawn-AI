# Bounded GBA Castle-entry learning feasibility

Run `20261028_055426_707460Z` used the verified checkpoint `20261028_001816_066107Z`.

PPO (CnnPolicy, CPU) trained from scratch for 2048 timesteps in 23.5 seconds. Training observed 0 confirmed goal events.

## Three-episode same-start comparison

- SCRIPTED_UP: 3/3 first-entry goals; decision steps [16, 16, 16].
- SEEDED_RANDOM: 0/3 first-entry goals; decision steps [].
- IDLE: 0/3 first-entry goals; decision steps [].
- PPO_TRAINED: 0/3 first-entry goals; decision steps [].

Only the scripted north baseline reached the goal (3/3 at 16 decisions with eight emulator frames per action). Random, idle, and the PPO policy reached 0/3. PPO selected A in all 96 held-out decisions; sparse first-entry reward supplied no training successes.

This was a bounded learning-feasibility smoke, not learned navigation. All evaluations use the same fixed checkpoint and short horizon. The agent did not learn the hardcoded north route; the route is reported only as a scripted reference.

The base Gymnasium environment remains reward-neutral. The optional wrapper rewards only a two-frame confirmed field-to-Castle transition. No unverified RAM-based progress rewards were added.

The compact contact sheet is included. PPO weights, raw traces, ROMs, and native states remain local.
