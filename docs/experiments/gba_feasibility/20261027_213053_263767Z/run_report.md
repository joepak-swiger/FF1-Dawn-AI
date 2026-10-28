# GBA party creation and playable checkpoint

**Result:** automated party creation and a playable Cornelia field checkpoint
passed in the existing WSL Stable-Retro runtime.

## Observed run

- Run ID: `20261027_213047_28057_328`
- State-machine report ID: `20261027_213053_263767Z`
- Menu sequence: Dawn of Souls title → game selection → Final Fantasy I title
  → new-game prompt → party editor → Begin Game confirmation.
- The raw START control had to be enabled with Stable-Retro `Actions.ALL`;
  default filtered actions omit START for this integration. START opened party
  confirmation and A accepted its preselected Yes option.
- Four roster name regions changed after Auto-Name. Party classes were
  Warrior, Monk, White Mage, and Black Mage.
- The captured candidate and post-UP screenshots show the Cornelia world map.
  The state machine advanced 15 opening-story prompts before testing a field
  candidate.
- Native Stable-Retro checkpoint: 397,312 bytes, captured at the field-map
  candidate. After reset/reload, two 42-frame idle controls matched exactly.
  Restoring again and holding UP for 12 frames followed by 30 idle frames
  changed 60.4% of pixels; the idle-to-idle difference was 0%, against a 3%
  minimum threshold. The saved after-UP frame visibly shows northward party
  movement on the same map.
- The Windows/WSL wrapper summary reported `playable checkpoint verified` and
  four-character party confirmed. Its dry-run path also passed separately.

## Milestones

| Milestone | Result |
|---|---|
| Stable-Retro boot/title and mapped controls | Passed: initial boot-logo anchor, menu transitions, and post-restore UP movement |
| Party editor reached | Passed using a recorded panel anchor |
| Four roster rows named | Passed |
| Party submitted and confirmed | Passed |
| Cornelia field reached | Passed by visual review and a controlled movement trial |
| Native checkpoint saved | Passed |
| Reset and native restore | Passed |
| Actual movement after restore | Passed: UP movement exceeded idle controls |
| Battery-save persistence | Not tested |

## Interpretation and limits

This is the first reproducible post-party playable checkpoint. The field cue
is a candidate filter, not a general scene classifier; the movement-control
test and screenshot review provide the independent evidence in this run. The
full local trace, screenshots, native state, ROM, and battery save are not
published. No PPO training, NES reward change, or model modification occurred.

## Next step

Repeat the route from a clean emulator reset and compare the normalized
milestones. Then test multiple movement directions from the restored checkpoint
and make the checkpoint a reproducible GBA evaluation start state. Keep large
scale GBA training disabled until repeated starts and mapped controls are
confirmed. Battery-save loading needs a separate isolated-copy experiment.
