# GBA perturb-and-recover imitation stress test

- Source benchmark: 20261028_153708_566851Z
- Successful cardinal training demonstrations: 7/7; 127 action labels.
- Held-out family stress set: east_south, west_south; episodes: 16; runtime: 51.12 sec.
- The perturbation reverses the model's own current directional suggestion for 0, 1, 2, or 4 decisions; the same fixed model then resumes control. This is not DAgger because no expert relabels learner-visited states.
- 0_wrong_moves: 4/4; east_south=2/2 actions={'NOOP': 0, 'UP': 34, 'DOWN': 0, 'LEFT': 8, 'RIGHT': 0}; west_south=2/2 actions={'NOOP': 0, 'UP': 34, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 8}
-   mean dominant-action fraction=0.8095; oscillation events=4; repeated-observation stalls=0; unknown scene frames=1888
- 1_wrong_moves: 4/4; east_south=2/2 actions={'NOOP': 0, 'UP': 34, 'DOWN': 0, 'LEFT': 14, 'RIGHT': 2}; west_south=2/2 actions={'NOOP': 0, 'UP': 34, 'DOWN': 0, 'LEFT': 2, 'RIGHT': 14}
-   mean dominant-action fraction=0.68; oscillation events=0; repeated-observation stalls=0; unknown scene frames=1996
- 2_wrong_moves: 0/4; east_south=0/2 actions={'NOOP': 0, 'UP': 70, 'DOWN': 0, 'LEFT': 24, 'RIGHT': 6}; west_south=0/2 actions={'NOOP': 0, 'UP': 68, 'DOWN': 0, 'LEFT': 4, 'RIGHT': 28}
-   mean dominant-action fraction=0.69; oscillation events=4; repeated-observation stalls=0; unknown scene frames=1992
- 4_wrong_moves: 2/4; east_south=2/2 actions={'NOOP': 0, 'UP': 32, 'DOWN': 0, 'LEFT': 16, 'RIGHT': 8}; west_south=0/2 actions={'NOOP': 0, 'UP': 0, 'DOWN': 0, 'LEFT': 104, 'RIGHT': 0}
-   mean dominant-action fraction=0.7857; oscillation events=0; repeated-observation stalls=0; unknown scene frames=2108
- Compound Castle-entry → field-return: 10/10 Castle-successful episodes independently confirmed two consecutive exterior anchor matches after DOWN actions (mean 35.6 DOWN macros; attempted across 16 total episodes).
- No PPO/RL updates or rewards were used. The stress test revisits the same two final-test route families and must not be counted as another untouched generalization set.

Detailed per-decision traces, observations, emulator states, ROM data, and model weights remain local and are not included.
