# NES independent scene verification

The RAM analysis records correlations with labels from a pixel heuristic. Those
labels are not ground truth. In particular, the heuristic is based on broad
screen appearance and can classify title screens, menus, transitions, and
battle-result layouts as battle-like.

## Current evidence

The RAM-independent audit of capture `20261027_070449_19924_934` selected saved
screenshots around the proxy transitions and four independently reviewed
visual references. The compact package is in
`docs/experiments/nes_state_audit/20261027_070449_19924_934/`.

- Session 1 step 120 visibly has the battle layout with monsters and the party
  panel. Step 280 is the overworld map.
- Session 1 step 1680 is the title/New Game screen. The old pixel proxy called
  it battle-like; RAM `0x0014` and `0x03CD` also held their proposed battle
  values there. This is direct counterevidence to those values being accepted
  as battle truth.
- Session 2 step 1360 is another title screen showing Continue/New Game, also
  flagged battle-like by the old proxy.
- Session 2 step 1760 has party sprites and empty side panels; it is retained as
  an ambiguous battle/transition/result layout, not labeled as a verified
  active battle.
- Session 3 steps 1360 and 1440 show character name-entry UI. These are menu
  states and should be separated from battle and field.

The audit reviewed 44 frames from 3 sessions and produced four downscaled
contact sheets. Of the selected frames, 19 have a battle-party-panel layout
hypothesis, 17 look like the overworld map, 5 are dark menu/transition
candidates, and 3 are unclassified. These counts describe image-layout cues,
not actual gameplay states. The dark candidate rule does not distinguish a
title/menu from a transition or possible game-over screen. A party panel can
persist into results or transitions, so its presence alone does not mean the
agent can issue a battle command.

## Reproduce the compact review package

```bash
python scripts/nes_scene_audit.py \
  videos/nes_ram_auto_capture/20261027_070449_19924_934 \
  docs/experiments/nes_state_audit/<new-run-id> \
  --anchor session_01:120=active_battle_screen \
  --anchor session_01:280=overworld_map \
  --anchor session_01:1680=title_new_game_menu \
  --anchor session_02:1360=title_continue_new_game \
  --anchor session_02:1760=ambiguous_party_panel_screen \
  --anchor session_03:1360=character_name_entry \
  --anchor session_03:1440=character_name_entry
```

The tool reads screenshots and pixel-proxy transition timestamps only. It does
not read RAM, emit rewards, or label outcomes. It refuses to overwrite an
existing output directory. The contact sheets are downscaled supporting
evidence; raw captures and RAM remain local.

## Next verification target

Collect a few transition screenshots at higher temporal resolution from battle
entry through command selection, turn execution, outcome, and field return.
Independently label just those frames. Then investigate candidate evidence for
HP/EXP/location and identify whether the same screen change is a menu or a
game-state transition. Until win, loss, and escape are independently
distinguished, preserve the v0.7 model and do not change rewards or train.
