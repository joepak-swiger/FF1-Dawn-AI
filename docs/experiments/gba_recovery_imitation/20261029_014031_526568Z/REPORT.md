# GBA recovery and route-feasibility update — 2026-10-29

## North-diagonal reachability

The old north-diagonal validation starts were reachable. A bounded route search found `DOWN×4 → inward×6 → UP`, confirmed by the independent two-frame Castle title gate on both starts in two resets each (4/4). The old `DOWN×4 → inward×4 → UP` comparator failed all four attempts. This diagnoses an under-corrected scripted comparator; these exposed validation starts are not new held-out evidence.

## Recovery experiment

Run `20261029_014031_526568Z` captured four learner-visited states after two actions opposite the current policy and collected 59 correction labels from two repeatably successful routes. The route search used no RAM or position labels and no PPO updates.

| Policy | East mistake state | West mistake state | Decision-6 timing states |
|---|---:|---:|---:|
| Original 3-NN | 2/2 | 2/2 | 4/4 |
| Correction-augmented 1-NN | 2/2 | 2/2 | 4/4 |
| Correction-augmented 3-NN | 0/2 | 2/2 | 4/4 |

The augmented 1-NN reproduced the searched east correction (`DOWN×8 → LEFT×8 → UP`) but took 33 decisions versus 26 for the original policy, which used `LEFT×9 → UP`. West used the original inward-then-UP pattern. The augmented 3-NN collapsed to DOWN on east. At the timing-held-out states, every policy succeeded by repeating UP×16, so this does not show improved recovery or generalization. No correction policy is promoted.

The separate online perturb benchmark repeated the earlier failure: 0 wrong moves 4/4; 1 wrong move 4/4; 2 wrong moves 0/4; 4 wrong moves 2/4 (east 2/2, west 0/2). Notably, the original policy succeeded from the saved synchronized two-error states. Why replay succeeds while the online sequence fails remains unresolved; the two protocols are reported separately.

## Scene coverage and limits

In the saved-state evaluation, most moving frames stayed `unknown`, overwhelmingly with classifier status `no_match`. The current classifier compares three fixed outdoor image crops against reviewed camera views. Camera scrolling and sprite/animation changes move those pixels away from the anchors. It is a sparse scene-transition detector, not a general field tracker; unknown frames remain unknown. Castle negative controls were not repeated in this recovery run.

No new frozen route-family benchmark was evaluated. Existing south-diagonal starts are already exposed. No learned Castle-entry/exit sequence was established; the known DOWN exit is scripted. No PPO, reward shaping, RAM reward, checkpoint changes, or NES changes were made.

## Evidence

- [Compact metrics](metrics.json)
- `recovery_route_castle_proof.png` shows one successful recovery trajectory reaching Castle Cornelia 1F.

Native states, ROMs, training datasets, raw traces and full recordings remain local.
