# GBA north-diagonal route feasibility — 20261028_220745_289874Z

Source run: `20261028_153708_566851Z`. Validation starts were used only to diagnose reachability; successful routes were not added to training.

Confirmed Castle successes: 4/28 route attempts.

All attempted routes use bounded DOWN → inward lateral → UP macros, with Castle entry confirmed by the independent two-frame title gate.

Per-start outcomes:
- `east_north`: 2/14 attempts reached Castle; successful route candidate(s): 7, 7
- `west_north`: 2/14 attempts reached Castle; successful route candidate(s): 7, 7

This is a feasibility search, not a learned policy, optimal route, map-coordinate estimate, or new benchmark. Native states and action traces remain local.

The previously used inverse-provenance comparator retraces four DOWN and four inward lateral macros before UP; that 4+4 candidate (index 3) failed on both starts and both repeats. Candidate 7 changed only the lateral correction to six macros: DOWN×4, LEFT×6 (east_north) or RIGHT×6 (west_north), then UP. It reached Castle on the 25th macro on all four repeats. This is direct evidence that the old comparator was under-correcting this route; no tile coordinates are inferred.
