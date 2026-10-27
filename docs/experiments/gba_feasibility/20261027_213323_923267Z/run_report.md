# GBA party creation and playable Cornelia checkpoint

**Result:** verified by a fresh automated WSL Stable-Retro run.

## Run evidence

- Run ID: `20261027_213318_29077_440`
- Probe report ID: `20261027_213323_923267Z`
- Initial frame matched the recorded Dawn of Souls boot-logo anchor. The
  state machine then recognized the Dawn of Souls title, game selection,
  FF1 title, new-game prompt, party editor, and party confirmation.
- Stable-Retro `Actions.ALL` was required because default filtered actions
  drop START for this integration. START opened Begin Game confirmation; A
  accepted its preselected Yes choice.
- All four name regions changed after Auto-Name. The confirmed classes were
  Warrior, Monk, White Mage, and Black Mage.
- The opening sequence advanced with bounded A taps to the Cornelia world
  map. The candidate screenshot was inspected and showed the playable map.
- Native Stable-Retro checkpoint: 397,312 bytes. After reset and restore,
  two 42-frame idle controls matched. After another restore, UP held for 12
  frames plus 30 idle frames changed 60.4% of pixels, versus 0% idle-to-idle
  change and a 3% threshold. The final resume screenshot shows the party's
  northward movement on the same map.
- The launcher summary reported `playable checkpoint verified`; top-level,
  nested runtime, runtime progress, and human-readable milestones agree.

## Milestones

| Milestone | Result |
|---|---|
| Emulator boot anchor | Passed |
| Mapped controls | Passed through recognized menu transitions and field movement |
| Party setup and four characters | Passed |
| Party submit/confirmation | Passed |
| Cornelia playable field | Passed by visual review and movement controls |
| Native checkpoint save/reset/restore | Passed |
| Gameplay movement after restore | Passed |
| Battery-save loading | Not tested |

The green/blue field cue is only a candidate filter. The screenshot and
post-restore movement comparison provide the evidence; this is not a generic
scene classifier. Local raw traces, screenshots, state files, ROM, and saves
are not included. No training or NES code/model/reward changes were made.

## Next step

Repeat the route from a clean reset and test multiple movement directions from
the restored checkpoint. Then make the checkpoint a reusable GBA evaluation
start state. Keep large-scale training disabled until those repeatability
checks pass; test battery-save loading separately in an isolated copy.
