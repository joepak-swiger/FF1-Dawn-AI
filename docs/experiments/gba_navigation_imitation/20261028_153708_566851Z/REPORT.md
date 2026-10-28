# Multi-start GBA navigation imitation benchmark

Run `20261028_153708_566851Z` from the unchanged local Seed-party checkpoint. The verified party is Sif/Warrior, Myth/Thief, Milo/Monk, Seed/Red Mage.

All 11 starts were confirmed as Cornelia exterior by two-frame scene gate and reproduced after native reload. They were generated from one source checkpoint; different start views are not separate maps or coordinate measurements.

The train split has 7 cardinal route starts; validation has two north-diagonal route families; final test has two south-diagonal route families. Each validation and test start was reset twice. The final test was untouched for training and validation selection.

Successful demonstrator rollouts: 7/7, covering 127 labeled decisions. Training is supervised behavior cloning, not RL; RL timesteps = 0.

## Results by split

| Policy | Train | Validation | Frozen final test |
|---|---:|---:|---:|
| IDLE | 0/7 | 0/4 | 0/4 |
| SEEDED_RANDOM | 0/7 | 0/4 | 0/4 |
| ALWAYS_UP | 3/7 | 0/4 | 0/4 |
| SCRIPTED | 7/7 | 0/4 | 0/4 |
| BC_3NN_CENTER_WEST_ABLATION | 5/7 | 0/4 | 2/4 |
| BC_3NN_MULTI_START | 7/7 | 0/4 | 4/4 |
| BC_TEMPORAL_3NN | 7/7 | 0/4 | 4/4 |
| BC_BALANCED_LINEAR | 7/7 | 0/4 | 4/4 |

Final-test detail (two independent resets per start):

| Policy | East-south | West-south | Turn actions across all repeats |
|---|---:|---:|---|
| IDLE | 0/2 | 0/2 | left=0, right=0 |
| SEEDED_RANDOM | 0/2 | 0/2 | left=26, right=27 |
| ALWAYS_UP | 0/2 | 0/2 | left=0, right=0 |
| SCRIPTED | 0/2 | 0/2 | left=8, right=8 |
| BC_3NN_CENTER_WEST_ABLATION | 0/2 | 2/2 | left=0, right=10 |
| BC_3NN_MULTI_START | 2/2 | 2/2 | left=8, right=8 |
| BC_TEMPORAL_3NN | 2/2 | 2/2 | left=8, right=8 |
| BC_BALANCED_LINEAR | 2/2 | 2/2 | left=10, right=8 |

Final-test scene/loop diagnostics (episode means or totals):

- IDLE: unknown-frame fraction 0.790; mean dominant-action fraction 1.000; oscillations 0; worst identical-observation streak 1; false Castle events 0.
- SEEDED_RANDOM: unknown-frame fraction 0.976; mean dominant-action fraction 0.307; oscillations 26; worst identical-observation streak 7; false Castle events 0.
- ALWAYS_UP: unknown-frame fraction 0.958; mean dominant-action fraction 1.000; oscillations 0; worst identical-observation streak 2; false Castle events 0.
- SCRIPTED: unknown-frame fraction 0.958; mean dominant-action fraction 0.917; oscillations 0; worst identical-observation streak 1; false Castle events 0.
- BC_3NN_CENTER_WEST_ABLATION: unknown-frame fraction 0.967; mean dominant-action fraction 0.881; oscillations 0; worst identical-observation streak 1; false Castle events 0.
- BC_3NN_MULTI_START: unknown-frame fraction 0.938; mean dominant-action fraction 0.809; oscillations 4; worst identical-observation streak 1; false Castle events 0.
- BC_TEMPORAL_3NN: unknown-frame fraction 0.938; mean dominant-action fraction 0.809; oscillations 0; worst identical-observation streak 1; false Castle events 0.
- BC_BALANCED_LINEAR: unknown-frame fraction 0.938; mean dominant-action fraction 0.786; oscillations 0; worst identical-observation streak 1; false Castle events 0.

Validation selection: **none** (no_learned_policy_succeeded_on_validation). No policy reached Castle on either validation route.
Held-out challenge: always-UP failed both starts (0/4). The predeclared multi-start frame 3-NN, three-observation temporal 3-NN, and balanced linear imitator each reached Castle on both starts in both resets (4/4). Successful model action totals were 8 LEFT + 34 UP for east-south and 8 RIGHT + 34 UP for west-south across two repeats. This meets the bounded turn-generalization criterion for these two route families, but the sample is small.

The center+west-only 3-NN ablation succeeded 2/4 and repeated all-RIGHT on east-south; the expanded model's added east examples address this prior action-collapse pattern. Idle, seeded-random, always-UP, and provenance-scripted baselines each scored 0/4 on the final set. The scripted baseline failed on diagonal starts despite reaching 7/7 training starts.

Castle detector negative control: menu transition produced 0 false Castle entries. No RAM coordinates, game rewards, or PPO training were used.

## Interpretation and next step

Claim: `learned_policy_succeeded_on_multiple_held_out_turning_start_families`. The BC policies use only rendered observations and actions; they did not receive start IDs, route provenance, or oracle actions at inference. The final candidate comparison is reported separately from validation selection; validation had no successful policy, so no model was chosen from it.
Follow-up run `20261028_160355_758132Z` recollected the seven training demonstrations and perturbed the same two final-test families. Recovery succeeded 4/4 with one wrong move, 0/4 with two wrong moves, and 2/4 with four wrong moves; the west-south policy collapsed to LEFT in both four-move trials. In the 10 episodes reaching Castle, the same run continued with DOWN and confirmed return to the exterior in 10/10, using a mean 35.6 macros. This is a compound-task result on the reused families, not a new held-out result.

Next: collect expert corrections on perturbed states and evaluate recovery on fresh route families, then attempt the compound Castle-entry/exit objective. Dynamic field-scene classification is still mostly unknown during movement (~94%), so do not start PPO or use a field-visit reward yet.

Published package excludes ROMs, save states, raw traces, native checkpoints, and model weights.
