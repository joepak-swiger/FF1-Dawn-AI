# GBA menu probes and current verified scope

`scripts/gba_menu_smoke.py` forks the verified Seed-party checkpoint through
`FF1GbaEnv`, runs separate fresh resets, and records each input and screenshot
under a unique ignored `videos/gba_menu_smoke/<run-id>/` folder. It only uses
START, directional taps, A, and neutral waits. Menu state changes must match
reviewed anchors twice; an unknown view stops that operation and now produces
a retained partial report rather than discarding successful earlier checks.

## Live Issue #23 probe

Run `20261030_031147_663554Z` independently confirmed the empty Items screen
(`1.000` anchor score; 46 input/release frames). The starting checkpoint has
no visible consumables. No item was consumed.

The equipment attempt did **not** verify an equipment view. It reached a view
that visually resembles Sif's Magic/spell list after 121 bounded input/release
frames; the configured classifier correctly left it `unknown` (`0.84966`).
No gear was changed, no spell/item was used, and no purchase was made. The
exact input trace and screenshots are preserved locally. The sanitized
summary and two small reviewed frames are in
[`20261030_issue23`](experiments/gba_menu_controller/20261030_issue23/REPORT.md).

The top-level party menu is known to show Items, Equipment, Magic, Status,
Formation, Config, Save, and Exit. That visual observation is not sufficient
to claim that character selection, equipment inspection, or gear changes work.
Potion/healing, shops, equipment compatibility, HP/MP extraction, and survival
signals remain unverified. Read-only menu checks stay isolated from autoplay
and the official checkpoint. Do not wire these operations into the gameplay
policy until an independently observable before/after effect has passed a
fresh-reset regression.
