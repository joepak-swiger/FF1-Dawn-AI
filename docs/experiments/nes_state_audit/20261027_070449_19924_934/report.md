# NES pixel-only scene audit — 20261027_070449_19924_934

This audit reads only saved screenshots and the pixel-proxy transition timestamps. It does not read RAM. Coarse layout families prioritize review and are not validated game-state labels.

- Source capture: 3 sessions, 5,403 samples.
- Selected 44 saved frames around pixel-proxy transitions plus 7 explicit visual anchors.
- Layout hypotheses among selected frames: {'battle_party_panel_layout_candidate': 19, 'dark_menu_or_transition_layout_candidate': 5, 'overworld_map_layout_candidate': 17, 'unclassified_layout': 3}.
- Dark menu/transition layout candidates labeled battle-like by the old proxy: 3; see frame_features.csv for steps.

## Independently reviewed visual anchors

| Session | Step | Human visual label | Pixel-only layout cue |
|---|---:|---|---|
| session_01 | 120 | active_battle_screen | battle_party_panel_layout_candidate |
| session_01 | 280 | overworld_map | overworld_map_layout_candidate |
| session_01 | 1680 | title_new_game_menu | dark_menu_or_transition_layout_candidate |
| session_02 | 1360 | title_continue_new_game | dark_menu_or_transition_layout_candidate |
| session_02 | 1760 | ambiguous_party_panel_screen | dark_menu_or_transition_layout_candidate |
| session_03 | 1360 | character_name_entry | unclassified_layout |
| session_03 | 1440 | character_name_entry | unclassified_layout |

Two manually reviewed title screens (session 1 step 1680 and session 2 step 1360) were flagged battle-like by the old pixel proxy. The dark-menu cue intentionally groups title, menus, transitions, and possible game-over screens; it does not separate them. Likewise, a battle-panel layout does not prove the game is awaiting a player action rather than showing a result or transition.

## Current limitations and decision

No independent labels exist yet for battle victory, defeat, escape, or game-over. The capture does not expose validated HP, EXP, location, or progress signals. Review the small transition sheets and label a handful of true result screens before promoting any visual or RAM cue. Do not change PPO rewards or start training from these candidates.

Only aggregate visual features, selected frame metadata, and downscaled contact sheets are included. Original frames and raw RAM remain in the local ignored capture.
