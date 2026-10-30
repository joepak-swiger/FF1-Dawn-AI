# GBA menu investigation — Issue #23

Live Stable-Retro run: `20261030_031147_663554Z`. Each operation reset from
the official local Seed-party checkpoint; no save state was overwritten.

| Probe | Result | Evidence |
| --- | --- | --- |
| Open Items | Confirmed empty inventory | [frame](inventory_empty_verified.png), exact reviewed anchor score 1.000 |
| Equipment path | Not verified; stopped on an unknown view | [diagnostic frame](equipment_unclassified.png), similarity 0.84966, below 0.975 gate |

The second view resembles Sif's Magic/spell menu. That is a visual description
of the screenshot, not a registered scene classification. The controller did
not claim that it reached Equipment. No item or spell was used, no gear changed,
and no shop was opened. The starting checkpoint's visible inventory contained
no consumables. The full screenshots and action trace remain in the local run
folder under `videos/gba_menu_smoke/20261030_031147_663554Z/`.

This is a useful negative result: the controller fails closed instead of
interpreting a changed screen as successful equipment access. Next, isolate
the top-menu cursor and character-selection steps with reviewed row-specific
anchors and keep the operation read-only until the equipment view is observed
reliably across fresh resets.
