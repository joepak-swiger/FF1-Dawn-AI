# Fresh GBA route-start audit — 2026-10-29

Status: **incomplete; no policy evaluation was run**. The independent reset/field-start gate rejected three of the four candidate route starts, so no model was selected and no final policy outcomes were collected.

The accepted generated starts were `val_west_north_loop` and `test_west_south_loop`. The gate did not confirm `val_east_north_loop`, `test_east_south_loop`, or `test_west_south_loop` after reset/settling. On a separate same-route replay, the game entered a battle screen, which the current conservative classifier left `unknown`. The same route can therefore produce field views or unrecognized encounter screens, and a field-looking screenshot alone is insufficient to certify a reproducible starting state.

Candidate images for these route families were visually reviewed to expand the scene-only anchor corpus. Because those route families are now exposed through that review, they are excluded from the next frozen final set. The new benchmark script uses separate four-direction reversal routes for any future final evaluation. Eight visually reviewed reference screenshots classify as exterior, while title/new-game, Cornelia town, party-menu, and Castle-transition negative controls still pass offline regression.

The anchors match specific reviewed views; they do not solve moving-frame coverage. No PPO, policy updates, rewards, or final-test outcomes were produced. Existing native checkpoints were not overwritten; generated states and raw captures remain local.
