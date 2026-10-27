# GBA custom party and playable checkpoint

Run `20261027_222507_734422Z` completed the custom party and playable-field checks from a clean game reset.

Independent matching clean-start run: `20261027_222448_322832Z`.

| Slot | Display name | Class |
|---:|---|---|
| 1 | Sif | Warrior |
| 2 | Myth | Thief |
| 3 | Milo | Monk |
| 4 | Seed | Red Mage |

The game accepted the party, reached the Cornelia field map, serialized a fresh native Stable-Retro state in a run-specific local directory, and restored it after reset. UP, DOWN, LEFT, and RIGHT each produced broad scene changes; two no-input controls were identical in this deterministic run.

The screenshot contact sheet is the compact visual evidence. Character letters were selected only after the on-screen keyboard cursor matched the requested letter anchor. The roster itself was visually inspected. Class choice was read from the visible picker cursor position and checked after each confirmation.

The original generic checkpoint and all earlier experiment directories were left untouched. The native state and full diagnostic trace remain local and are not included in this package. This establishes a repeatable playable setup checkpoint, not long-term party survival, save-RAM persistence, or reinforcement-learning readiness without a Gymnasium adapter.

## Milestones

- `rom_integration_boots`: **passed**
- `party_setup_started`: **passed**
- `four_characters_confirmed`: **passed**
- `party_setup_completed`: **passed**
- `playable_field_reached`: **passed**
- `native_checkpoint_saved`: **passed**
- `native_checkpoint_restored`: **passed**
- `field_movement_after_restore`: **passed**
