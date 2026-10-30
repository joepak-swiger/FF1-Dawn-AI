# GBA autonomous gameplay recording
Run ID: 20261030_031529_229206Z
Status: stopped_after_confirmed_battle_unrecognized_postbattle_map
Policy: autonomous_session; combat=learned; field navigation remains scripted or seeded-random.
Meaningful combat decisions by provenance: {'learned': 24, 'scripted_expert': 0, 'safety_supervisor': 0, 'random': 0} ({'learned': 100.0, 'scripted_expert': 0.0, 'safety_supervisor': 0.0, 'random': 0.0}%); model SHA-256: 6ba53b1d29573461cab9d71a5e8efaddb3af16ad76de1c4cf3b5df6c61f63f54; verified pre-action expert examples: 0.
Training labels are local-only; no supervised/RL optimizer runs inside autoplay.
Stop reason: specific_encounter_victory; postbattle_screen_not_in_reviewed_map_anchors
Decisions: 3655; emulator frames: 3995
Confirmed scenes: cornelia_exterior_field
Castle reached: False; field return confirmed: False
Distinct battle menu entries: 3; finished: 3; victories: 3; unknown outcomes: 0.
Attack/target command sequences: 12; A presses: 49; historical next-party panels confirmed: 9; terminal next-command flag: False.
Observed EXP amounts: [6, 6, 4]; unrecognized amount count: 0.
Observed Gil amounts: [24, 24, 18]; unrecognized amount count: 0; narrow map views: 2.
Known templates specifically matched 6 EXP=True, 4 EXP=True, 24 Gil=True, 18 Gil=True.
Level-up banners confirmed: [{'battle_id': 3, 'character': 'Sif'}].
Matched postbattle movement probes: 2; resumed movement confirmations: 2.
Next party command panel confirmed: False after 11 neutral frames.
After full party inputs: command panel returned=False; field anchor returned=False; neutral wait=427 frames.
A two-frame fixed Earned-[amount]-EXP phrase confirms each counted victory once; numeric amounts are only reported when a reviewed glyph template matches.
Unknown scene observations: 3588
Runtime: 26.778 seconds
GIF frames are captures from the Stable-Retro emulator with an on-screen policy/scene overlay.
No PPO optimizer training ran; base reward remained neutral.
