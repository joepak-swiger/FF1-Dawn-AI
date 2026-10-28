# GBA runtime trace review — 2026-10-27

## Decision and correction after full-size trace review

This run did **not** create a verified playable checkpoint. Full-size review of
the final input screenshots corrects the earlier contact-sheet-only reading:
the SELECT/DOWN sequence did auto-name all four roster rows (`Zeza`, `Sarisa`,
`Sherko`, `Topapa`) while retaining the default Warrior, Thief, White Mage,
and Black Mage classes. The sequence never pressed the party editor's visible
START = Done control, so the candidate checkpoint remained in the editor. The
after-reload screenshot entered the class-selection dialog after the replayed
A input. No overworld movement or playable gameplay was demonstrated.

Native Stable-Retro state serialization and deterministic reset/reload replay
did pass. That verifies the core's state round trip at the captured menu. It
does not verify a party, a playable scene, or gameplay resuming after reload.

## What the screenshots show

| Trace point | Visible screen | Finding |
|---|---|---|
| Boot idle, step 0 | Crystal/logo animation | The core produced the expected game artwork, not a blank frame. |
| Early START/A inputs, steps 1–8 | Dawn of Souls title and game-selection screens | START/A advanced through the splash/title sequence into game setup. |
| Steps 15–18 | Party roster and “Select a class” dialog | The roster editor is reached; A opens the class picker. |
| Steps 41–47 | Four roster name areas change to `Zeza`, `Sarisa`, `Sherko`, and `Topapa` | All four rows were auto-named; default classes remain visible. |
| Candidate checkpoint | Party editor with all four names and default classes | The party was configured but not submitted with START = Done; no overworld is present. |
| After native reload and replayed A | “Select a class” dialog | Restored state continued deterministically into the same menu flow. |

The first run executed 20 START/A pairs followed by SELECT/DOWN taps. The
repeated START/A phase cycled among roster/class-selection panels. The final
SELECT/DOWN frames do show name changes in all four rows; the previous
contact-sheet-only conclusion that these inputs made no visible progress was
incorrect. Exact-frame repeats indicate identical saved pixels only.

## Independently checked milestones

| Milestone | Result | Evidence |
|---|---|---|
| Emulator boot | Partially verified | Captured trace visibly shows the GBA logo/title and game setup screens. |
| Input response | Partially verified | START/A visibly advance the boot and menu flow. Direction/SELECT effects in the later route are unclear. |
| Party creation | Partially verified | Four names are visible in the roster editor, but the Done action was not sent. |
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
