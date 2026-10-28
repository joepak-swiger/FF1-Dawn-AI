# Cornelia scene anchors and transition gate

Live run `20261028_060050_480805Z` used the verified local Seed-party checkpoint `20261028_001816_066107Z`.

## Independently reviewed screens

The anchor set names Cornelia exterior field, Cornelia town title, Castle Cornelia 1F title, and the rendered party/status menu. Unknowns and fades remain unknown; no RAM address or full-frame hash participates. Castle and town are matched from the game's own title strip. The exterior uses distinct map landmarks from two reviewed screen positions and tolerates limited vertical scroll.

- Fresh checkpoint field anchors: 3/3.
- Two-frame Castle entry after a confirmed field: 3/3.
- Two-frame Castle exit back to recognized exterior: 3/3.
- Cornelia town confirmed after pausing on its title candidate: 1/3 seeded routes.
- Status/menu screen confirmed: True with 0 Castle-entry events.
- Saved title/new-game fixture and fade/transition fixture remained unknown in offline regression tests.
- A one-frame Cornelia title candidate appears during some routes; the gate requires a second consecutive matching frame and does not promote the flash.

The three entry/exit trials are repeated restores from one fixed checkpoint and one fixed route, not broad location generalization. The town result is positive in one seeded route; the other routes did not visit town. These are episode-level counts; repeated frames are not treated as independent samples.

## Episode-level confirmation counts

- `cornelia_exterior_field`: 10/10 expected scene visits confirmed; 0 missed.
- `cornelia_town`: 1/1 expected scene visits confirmed; 0 missed.
- `castle_cornelia_1f`: 3/3 expected scene visits confirmed; 0 missed.
- `menu`: 1/1 expected scene visits confirmed; 0 missed.
- Unexpected confirmed scene transitions: 0.

## Decision

The validated first-entry event is suitable for an opt-in sparse-goal wrapper for this fixed-start feasibility experiment. It is not a general scene classifier and does not validate battle/result/game-over screens or progression beyond Castle entry.

Selected proof frames are included. Native states, ROMs, RAM dumps, full traces, and raw run folders remain local.
