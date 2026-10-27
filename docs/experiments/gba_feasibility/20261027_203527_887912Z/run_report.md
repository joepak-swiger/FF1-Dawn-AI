# GBA runtime trace review — 2026-10-27

## Decision

This run did **not** create a verified playable checkpoint. The game reached
the FF1 party name/class setup menus, including a “Select a class” dialog. The
candidate checkpoint and the post-reload frame both remain in party setup; no
overworld movement or other playable gameplay was demonstrated.

Native Stable-Retro state serialization and deterministic reset/reload replay
did pass. That verifies the core's state round trip at the captured menu. It
does not verify a party, a playable scene, or gameplay resuming after reload.

## What the screenshots show

| Trace point | Visible screen | Finding |
|---|---|---|
| Boot idle, step 0 | Crystal/logo animation | The core produced the expected game artwork, not a blank frame. |
| Early START/A inputs, steps 1–8 | Dawn of Souls title and game-selection screens | START/A advanced through the splash/title sequence into game setup. |
| Steps 15–18 | Party roster and “Select a class” dialog | The macro entered character setup. The screenshot is still a menu. |
| Candidate checkpoint | “New Game” and character name/class assignment panel | The party setup flow is incomplete; no overworld is present. |
| After native reload and replayed A | “Select a class” dialog | Restored state continued deterministically into the same menu flow. |

The first run executed 20 START/A pairs followed by SELECT/DOWN taps. Contact
sheets show the macro cycling among roster/class-selection panels. The seven
final SELECT/DOWN taps produced no clear visible screen progression in their
saved released frames. Exact-frame repeats are evidence of repeated pixels,
not semantic proof that an input was ignored or that the game was stalled.

## Independently checked milestones

| Milestone | Result | Evidence |
|---|---|---|
| Emulator boot | Partially verified | Captured trace visibly shows the GBA logo/title and game setup screens. |
| Input response | Partially verified | START/A visibly advance the boot and menu flow. Direction/SELECT effects in the later route are unclear. |
| Party creation | Not verified | Captured images remain on name/class assignment menus. |
| Playable overworld | Not reached | No world map or controllable field scene appears in the candidate or resumed screenshots. |
| Native state serialized | Passed | The core wrote 397,312 bytes; the saved state matched the in-memory bytes. |
| State restored after reset | Passed | Replaying the same A/idle continuation after reset and native state load matched the pre-reset RGB frame exactly. |
| Gameplay after restoration | Not verified | The deterministic replay was still in the class-selection menu. |
| Battery-save loading | Not tested | No isolated battery-save experiment was part of this run. |

## Follow-up guard validation

A new one-click run (`20261027_204417_21619_444`, report timestamp
`20261027_204423_053988Z`) exercised the updated macro. It detected that START/A
pair 10 returned an exact frame seen after pair 8 and stopped the route before
the additional SELECT/DOWN sequence. It saved 41 trace frames and completed
the native state round trip. Its screenshot still showed an unfinished party
name/class screen, so the guard prevents repeated menu cycling but does not
solve party creation.

The new run's top-level and nested milestones agree. `status: completed` means
the probe finished; `gameplay_readiness: not_verified` keeps the result scoped
to what was actually established.

## Evidence and limits

The source screenshots, full input trace, raw native state, and runtime logs
remain in the local ignored `videos/gba_feasibility/` directories. This report
publishes only manually reviewed screen descriptions and sanitized milestone
results; it does not include ROM data, save states, local paths, or game image
assets. The trace's pixel deltas and exact image hashes are triage evidence,
not semantic labels.

## Next step

Replace the fixed party macro with a bounded, screen-feedback-driven setup
flow. The next runtime should capture one input at a time around the roster,
name/class editor, and confirmation screens, and halt on any screen not
recognized by reviewed evidence. Confirm the four characters and reach the
Cornelia overworld before calling the checkpoint playable. Then load the
native state, perform and observe a field movement, and separately verify the
resulting frame changed as expected. Keep GBA training on hold until those
checks pass.
