# GBA locomotion and position-signal validation

Run `20261028_031101_730780Z` tested the official Seed-party native checkpoint from `20261028_001816_066107Z`.

- Stable-Retro reset frame hash matched for all 171 restores; the native checkpoint file was unchanged.
- Three independent restored starts were tested for each direction at 1, 2, 4, 8, 16, 32, 64 held emulator frames, followed by two neutral release frames.
- At 1–32 frames, the phase-correlation screen-space translation sign matched the requested cardinal direction on all three repeats. This measures map-image displacement; it does not identify tile units or absolute player coordinates.
- For UP/DOWN and LEFT/RIGHT routes of 16 and 32 frames each way, the rendered map returned to within 0.04 pixels of its starting registration across repeats. The matched idle control stayed within 0.01 pixels. Screen pixel changes were close to idle levels after reversal, consistent with animation/facing differences rather than a continued map offset.
- Eight-frame reversals did not return to the starting map registration (about 14 pixels residual), so the reversible pulse is at least 16 frames in this tested checkpoint/route.
- A 128-frame northward route visibly entered Castle Cornelia 1F. It did not establish blocked terrain; the route crossed a map transition.
- RAM reads returned 527,360 bytes. Candidate offsets `0x33FC` and `0x3404` responded repeatably to direction at short horizons, but their direction values do not uniquely separate both axes, many other bytes respond, and the candidates were not validated at a second known world position. They remain provisional; no coordinate/visit tracker is enabled.
- No GBA RAM offset is currently accepted as a player coordinate. The hundreds of repeatable responders and rendering-memory changes show why change detection alone is insufficient.
- Rewards stayed neutral. No PPO training ran. Raw RAM, action traces, ROMs, native checkpoint data, and full-resolution capture frames are not included.

## Evidence

- `movement_sheet.png` compares short and longer cardinal inputs with neutral controls.
- `reversal_sheet.png` compares each route, its reverse, and equal-duration idle controls.
- `north_path_sheet.png` shows the route entering Castle Cornelia 1F, explaining why this did not test an obstacle.

## Next step

Capture a second known field position, repeat the RAM differential and reversal protocol there, and test a visually established wall/water boundary with matched controls. Only then decide whether a map-relative visit tracker can be promoted; absolute tile coordinates remain unverified.
