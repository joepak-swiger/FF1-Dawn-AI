# Castle Cornelia landmark evaluation

Run `20261028_052724_624774Z` used the verified Seed-party checkpoint `20261028_001816_066107Z`.

## Verified destination

The target is Castle Cornelia 1F, confirmed from the game's own floor-title UI. The detector compares the fixed title-banner text region to a reference extracted from a previously reviewed gameplay frame. It requires two consecutive high-confidence matches, ignores RAM and full-frame hashes, and returns `unknown` for all other scenes.

The first 1-frame-per-action north route produced a transition/fade at step 119 (uncertain), matched at step 120, and met the two-frame confirmation at step 121. Three clean restores reached the target at that same confirmation step (3/3). The source checkpoint hash and all reset observations remained unchanged/reproducible.

## Controlled policy comparison

| Policy | Result | Entry cost |
|---|---:|---:|
| Straight north | 3/3 | 121.0 one-frame actions |
| Seeded random | 0/12 | — one-frame actions |
| Idle | 0/3 | — one-frame actions |
| Straight south control | 0/3 | — one-frame actions |

The detector returned no castle hits in 3539 random, idle, southbound, and START-menu control observations (0 false positives). Visual review confirmed the START control showed a white transition at frames 8/16 and the party/status menu at frames 32/64; none was classified as Castle Cornelia.

## Castle exit and movement diagnostics

After the castle-entry route, sustained DOWN input faded from the titled interior around frame 64. Frame 72 shows the scene transition, and frame 80 shows the exterior Cornelia overworld around the castle; visual review confirms a field return. The runtime detector intentionally reports this as unknown because it is a castle-title detector, not an exterior-field classifier.

An 8-frame out-and-back retains about 14–16 screen pixels of apparent map offset immediately after reversing. With one neutral frame and 16 settling frames, residual averaged about 1 px; with at least two neutral release frames and 16 settling frames, the rendered map returned to within about 0.01 px of its start. Matched idle remained near 0 px. This points to movement/input timing and delayed camera settling; it does not establish a tile coordinate.

The 64-frame DOWN probe's scene shift reached about −46 px by frame 48 and plateaued near −47.9 px by frame 56. Screenshots show the party approaching/crossing the southern bridge and the map view reaching its edge. This is not evidence of a blocked tile; collision remains unverified.

## RAM candidates

Across the three visually confirmed castle entries, `0x33FC/0x3404` read `226/230`. A seeded random replay (seed 110, step 128) displayed the separate `Cornelia` town screen and read the same `226/230` pair while the castle-title detector returned `unknown`. The pair also occurred on 127 non-confirmed frames in the main run. Neither offset, nor the pair, identifies Castle Cornelia uniquely. Their memory meaning remains unknown and neither is used by the detector or rewards.

## Limits and next step

The castle landmark is reliable for this named floor screen, but this is not a generalized scene classifier. The random comparison used 12 fixed seeds and a 192-action horizon, so its 0/12 result is a bounded baseline rather than a universal success rate. No PPO training or reward changes were made; NES v0.7 and its experiments were untouched.

Next, add separately validated field/town/castle screen labels and use the title landmark in a short, held-out Gymnasium evaluation. Keep rewards neutral until labels survive negative controls across scenes and resets.

Evidence includes the transition, menu false-positive control, same-value Cornelia-town counterexample, castle exit, and the DOWN camera-edge frames. Raw traces, ROMs, native states, and full capture directories remain local.
