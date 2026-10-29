# Issue #15 autonomous gameplay evidence

## End-to-end run

- Run ID: `20261029_031018_621251Z`
- Official Seed-party checkpoint SHA-256 matched; repeated reset observations were identical.
- Policy: seeded-random cardinal exploration plus scripted Attack/target selection. No learned policy or optimizer training.
- Runtime: 5.342 seconds; 1,359 actions / emulator frames.
- Goblin menu first confirmed at decision 471. Four Attack/target command sequences were confirmed using the strict 100×50 left-panel matches (eight combat A presses).
- After 742 neutral battle frames, two consecutive matches confirmed `Earned 6 EXP.`. The exact result still showed Sif 31/35 HP, Myth 30/30, Milo 32/33, Seed 30/30.
- The controller acknowledged that message, confirmed the specific `Obtained 24 gil.` banner, acknowledged it with bounded retries, and confirmed a second two-frame anchor for one Cornelia-area world-map view.
- The generic scene classifier did not call that frame `cornelia_exterior_field`; `field_return_confirmed` remains false. Report this only as a narrow observed map-view return.
- Watch the full local GIF at `E:\JOEPAK\FF1-Dawn-AI\videos\gba_autoplay\20261029_031018_621251Z\autoplay.gif`. The GIF contains the real emulator frames, action/policy overlay, battle, reward messages, and map-view return. Full GIF remains local.

## Independent seeds and failure handling

| Seed | Run ID | Result |
|---:|---|---|
| 77 | `20261029_031018_621251Z` | Exact 6 EXP + 24 gil confirmed; narrow map view confirmed |
| 1729 | `20261029_031111_485204Z` | Exact 6 EXP + 24 gil confirmed; map anchor did not match |
| 314 | `20261029_030055_546675Z` | Stopped after one A when Magic submenu failed strict target-panel match; no outcome claimed |

These trials repeatedly used the observed Goblin formation. They do not establish reliability on other enemies. The exact reward banners are specific to this displayed amount; no general EXP parser, escape/defeat/game-over detector, HP parser, or verified RAM signal exists.

## Testing and boundaries

- Seven focused autoplay tests passed, including Attack/target/Magic menu positives and negatives, two-frame reward/map gates, overlay preservation of the top EXP banner, and no false victory from unknown screen changes.
- Full repository suite: 102 tests passed.
- `py_compile` and real WSL runtime/checkpoint dry-run passed.
- A fresh 204-action scripted baseline also confirmed the Castle and returned exterior anchors.
- PPO timesteps: 0. Rewards remain neutral. No change to the official party/checkpoint or NES v0.7.
- Compact sanitized evidence and selected small proof frames are in the run-specific directories. ROMs, native states, raw traces, and full videos were not published.

## Next

Validate command selection and outcome handling on a different enemy formation, then repeat controlled movement after the narrow map return against idle. Do not enable battle rewards or large PPO training until those signals generalize beyond this Goblin example.