# Issue #23 result acknowledgment and continuous autoplay

## Result-screen wait change

The previous live seed-77 run (`20261030_002302_471698Z`) entered five Goblin
battles, confirmed four wins, and spent **3,981 emulator frames** in the
`wait_for_postbattle_map` phase (about 995 frames per confirmed victory).
Investigation found this path could inherit the much longer battle-resolution
window after Gil acknowledgment. The controller now has a separate
`--postvictory-wait-frames` cap (default 48) and retries only when the exact,
reviewed Gil banner remains visible. Unknown screens still stop safely; EXP,
Gil, and level-up acknowledgments remain distinct.

The same seed/checkpoint with the changed controller (`20261030_031529_229206Z`)
ran for 26.778 seconds, entered three Goblin encounters, confirmed **3/3
victories** (6/6/4 EXP; 24/24/18 Gil), and resumed screen-space movement after
the first two. It spent 184 emulator frames in post-victory map waits (about
61 per confirmed win), roughly **93.8% fewer wait-phase frames per win** than
the earlier run average. The third result included Sif's level-up and was
acknowledged; the controller then stopped because the next map view was not in
the reviewed postbattle anchor set. This comparison is a run-level average,
not a matched per-battle timing study. These are real emulator-frame counts,
not GIF playback compression.

The GIF remains local at
`videos/gba_autoplay/20261030_031529_229206Z/autoplay.gif`. Compact proof and
the full sanitized run metrics are published in
[`20261030_031529_229206Z`](../20261030_031529_229206Z/REPORT.md).

A same-seed scripted-combat comparator (`20261030_031938_119289Z`) also
confirmed 3/3 victories with the same 6/6/4 EXP and 24/24/18 Gil, resumed
movement 2/3 times, used 184 post-victory map-wait frames, and stopped at the
same third unrecognized result map. Both used the same seeded-random field
controller. The learned model therefore matched the scripted result; it did
not improve it. The model run attributed 24/24 combat choices to the model;
the comparator attributed 24/24 to scripted control.

## Learned policy and limits

The existing local supervised v2 model made 24/24 meaningful combat choices
in the run, attributed to `learned`; the model hash is recorded in the
sanitized run evidence. The choices remain narrow A/target menu selections in
the same Goblin family. Field exploration remains seeded-random, optimizer
steps were zero, and rewards stayed neutral. No new model was trained because
this run did not provide diverse expert labels or a held-out alternative
action. It is not evidence of navigation learning or combat superiority over
scripted control.

No non-Goblin enemy was independently confirmed. The fifth-battle restore-to-
Flee discrepancy was not exercised: this run stopped after its third win on
an unrecognized postbattle map view. It remains unresolved.
