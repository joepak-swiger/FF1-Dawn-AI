# Issue #21 — GBA learned-agent v2

## What the live agent did

The learned-combat / seeded-random-field session `20261030_002302_471698Z` ran for 76.1 seconds (9,334 decisions; 10,014 emulator frames). It entered five Goblin encounters, confirmed four victories, and resumed movement after each of those wins without resetting the emulator. Confirmed rewards were 6/6/4/4 EXP and 24/24/18/18 Gil. There was one Attack-panel retry, permitted only after the exact Attack panel persisted. The fifth encounter stopped safely when the target panel was not confirmed; its result is unknown.

On battle four's new map camera, the first cardinal probe matched idle (0.43 mean pixel difference), while the opposite probe differed (44.05). Replaying the opposite direction produced verified screen-space movement, allowing the same session to continue. This establishes visible movement after the battle; it does not estimate coordinates or a blocked tile.

Local gameplay recording: `videos/gba_autoplay/20261030_002302_471698Z/autoplay.gif`. The full recording and native states remain outside GitHub. The compact autoplay evidence is published at [`20261030_002302_471698Z`](../../gba_autoplay/20261030_002302_471698Z/REPORT.md).

Same-state comparison GIFs are local: learned v2 `videos/gba_combat_branch_eval/20261030_002442_912301Z/combat_branch.gif`, scripted `videos/gba_combat_branch_eval/20261030_002447_557483Z/combat_branch.gif`, seeded random `videos/gba_combat_branch_eval/20261030_002452_061994Z/combat_branch.gif`, and learned v1 `videos/gba_combat_branch_eval/20261030_002506_424158Z/combat_branch.gif`.

## Policy comparison and what v2 learned

The new linear behavior-cloning model was trained locally for 150 epochs from 69 deduplicated examples across eight encounter-parent groups. The split contained train 34 (A 33/B 1), validation 17 (A 16/B 1), and test 18 (A 17/B 1), with group separation. The held-out split scored 18/18, including just one B example on the `menu_other` stage. Its confusion matrix and per-action/per-stage metrics are in [`metrics.json`](metrics.json). This small Goblin-only set does not establish generalization.

From the same held-out Magic-menu state, v2 chose B to leave Magic, then A for Attack and target selection, and confirmed a 6-EXP win (10 learned choices, 9 A and 1 B). The previous v1 model and the scripted controller also won for 6 EXP with the same decisions. Seeded random was stopped after an unsupported choice. From a separate Attack-menu branch, learned v1 and scripted both won for 4 EXP; random's B proposal was rejected. These results show the model can reproduce the reviewed B recovery action, but v2 did not beat v1 or the script.

The new successful demonstration branch added eight A labels for one encounter parent; it did not add a new combat action class. The B examples in v2's training data came from earlier verified Magic-menu recovery branches. No expert correction was collected from a learner failure, and no model weights or datasets are published. Local v2 model SHA-256: `6ba53b1d29573461cab9d71a5e8efaddb3af16ad76de1c4cf3b5df6c61f63f54`.

## Failures and limits

- The fifth full-session battle's target screen was never confirmed, so no outcome or training label was inferred.
- Replaying its saved state showed a Flee command row after framebuffer synchronization (Attack-anchor score 0.9485, under the 0.95 threshold). Learned, scripted, and random branch evaluators all stopped before acting. Keep the threshold strict; the capture/restore timing mismatch needs investigation.
- The four wins were the already supported Goblin family. A distinct enemy species, HP/KO/survival detector, defeat/escape recognizer, and meaningful learned navigation are not established.
- Field movement remains seeded random. PPO timesteps are zero, Gymnasium rewards remain neutral, and no RAM reward was enabled.

## Validation

- Full WSL unittest suite: **115 passed**.
- `compileall` passed; default and learned-policy GBA dry-runs passed without starting an emulator.
- Four bounded live branch evaluations compared learned, scripted, and random controllers on identical states; one held-out Magic-menu comparison and the five-encounter autonomous session generated local GIF recordings.
- Existing WSL runtime and official checkpoint were reused. No package installation, checkpoint replacement, or NES v0.7 changes occurred.

Selected proof frames: [held-out Magic-menu branch](heldout_magic_contact_sheet.png) and [fourth postbattle field view](fourth_postbattle_field.png).
