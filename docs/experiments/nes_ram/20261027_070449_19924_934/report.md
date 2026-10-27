# NES RAM candidate analysis — 20261027_070449_19924_934

## Run record

- Completed capture and analysis: 3 independent resets, 5,403 samples.
- Pixel-proxy labels: 1312 field-like and 4091 battle-like.
- PPO baseline: v0.7 trusted baseline; unchanged during capture: True.
- The analyzed scenes are proxy labels, not independently verified game states. Candidate agreement is not evidence that an address semantically means “battle.”
- Historical v0.9 data had 901 observations (718 battle-like and 183 field-like by the same pixel-heuristic family). At `0x0014`, field samples were 23 in 99.45% of observations and battle-like samples were 0 throughout; the two recorded edges were 23→0 and 0→23. This supports repeatability of the association, but both runs rely on visual proxy labels and do not establish a battle flag.

| Session | Samples | Field-like proxy | Battle-like proxy | Proxy transitions |
|---|---:|---:|---:|---:|
| session_01 | 1801 | 406 | 1395 | 9 |
| session_02 | 1801 | 358 | 1443 | 11 |
| session_03 | 1801 | 548 | 1253 | 10 |

## Candidate comparison

For this table, predict “battle” when the byte equals the pooled battle mode.
FP/FN and balanced accuracy are disagreements against pixel labels, not
adjudicated gameplay errors. The last column counts candidate value edges that
the analyzer marked as conflicting with those labels.

| Address | Field mode (fraction) | Battle mode (fraction) | FP | FN | BA vs proxy | Mode pair repeats | Conflicting edges |
|---|---:|---:|---:|---:|---:|:---:|---:|
| 0x0014 | 23 (0.977) | 0 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03CD | 26 (0.988) | 0 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03CF | 25 (0.988) | 48 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03D1 | 15 (0.988) | 40 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03D2 | 22 (0.377) | 24 (1.000) | 16 | 2 | 0.994 | False | 11 |
| 0x03D3 | 54 (0.988) | 33 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03D5 | 15 (0.988) | 22 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03D9 | 15 (0.988) | 48 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03DA | 39 (0.988) | 34 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03DB | 48 (0.988) | 18 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03DD | 15 (0.988) | 48 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03DE | 48 (0.988) | 16 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03DF | 26 (0.988) | 0 (1.000) | 16 | 2 | 0.994 | True | 11 |
| 0x03CE | 41 (0.988) | 15 (0.905) | 14 | 387 | 0.947 | True | 12 |
| 0x00D2 | 73 (0.990) | 201 (0.899) | 7 | 415 | 0.947 | True | 3 |
| 0x00D3 | 181 (0.990) | 179 (0.899) | 7 | 415 | 0.947 | True | 3 |
| 0x0270 | 248 (0.985) | 240 (0.905) | 20 | 387 | 0.945 | True | 14 |
| 0x0271 | 248 (0.985) | 240 (0.905) | 20 | 387 | 0.945 | True | 14 |
| 0x0272 | 248 (0.985) | 240 (0.905) | 20 | 387 | 0.945 | True | 14 |
| 0x0273 | 248 (0.985) | 240 (0.905) | 20 | 387 | 0.945 | True | 14 |
| 0x0274 | 248 (0.985) | 240 (0.905) | 20 | 387 | 0.945 | True | 14 |
| 0x0275 | 248 (0.985) | 240 (0.905) | 20 | 387 | 0.945 | True | 14 |

### Session-level watch-list values

Per-session modal values and proxy disagreements for the main candidates are
shown below; all counts remain comparisons against the pixel heuristic.

| Address | Session | Field mode (fraction) | Battle mode (fraction) | FP | FN |
|---|---|---:|---:|---:|---:|
| 0x0014 | session_01 | 23 (0.985) | 0 (1.000) | 6 | 0 |
| 0x0014 | session_02 | 23 (0.955) | 0 (0.999) | 6 | 1 |
| 0x0014 | session_03 | 23 (0.985) | 0 (0.999) | 4 | 1 |
| 0x03CD | session_01 | 26 (0.985) | 0 (1.000) | 6 | 0 |
| 0x03CD | session_02 | 26 (0.983) | 0 (0.999) | 6 | 1 |
| 0x03CD | session_03 | 26 (0.993) | 0 (0.999) | 4 | 1 |
| 0x03D2 | session_01 | 39 (0.574) | 24 (1.000) | 6 | 0 |
| 0x03D2 | session_02 | 48 (0.508) | 24 (0.999) | 6 | 1 |
| 0x03D2 | session_03 | 22 (0.593) | 24 (0.999) | 4 | 1 |
| 0x00D2 | session_01 | 73 (0.993) | 201 (0.903) | 3 | 136 |
| 0x00D2 | session_02 | 73 (0.986) | 201 (0.906) | 2 | 136 |
| 0x00D2 | session_03 | 73 (0.991) | 201 (0.886) | 2 | 143 |
| 0x0270 | session_01 | 248 (0.980) | 240 (0.909) | 8 | 127 |
| 0x0270 | session_02 | 248 (0.983) | 240 (0.913) | 6 | 126 |
| 0x0270 | session_03 | 248 (0.989) | 240 (0.893) | 6 | 134 |

## Interpretation

- `0x0014` repeated a distinct 23/0 modal pair in all three sessions. Against
  the pixel proxy it had 16 field samples predicted as battle and 2 battle
  samples predicted as field (balanced accuracy 0.994). It is more repeatable here than in the prior single manual session, but remains provisional.
- Stronger-scoring nearby candidates include several bytes in the
  `0x03CD`–`0x03DF` neighborhood.
  Several tie `0x0014` on proxy accuracy and share its confusion counts. Their
  close address grouping and identical proxy score do not make them independent
  confirmation; they may be correlated scene data.
- Pooled scores hide per-session mode drift at 0x03D2; inspect each session's distribution in `metrics.json` before ranking these bytes.
- For `0x0014`, 18 of 18 modal-rule disagreements occur within one sample of a pixel-proxy transition; this is consistent with boundary timing/label uncertainty, but still needs independent scene labels.
- `0x00D2` repeated modes 73/201; `0x0270` repeated 248/240. Their exact-mode
  rules miss many pixel-labeled battle samples, so they are less suitable as
  simple binary detectors in this run despite clean mode separation.
- Contact-sheet frames are downscaled samples selected around proxy transitions
  and rapid toggles. Review them before interpreting candidate edges; the sheet
  does not supply independent labels.
- No battle wins, losses, escapes, HP/EXP changes, or story progress were
  independently measured. This experiment does not validate a reward signal.

### Manual review notes

- Session 1 step 1680: title/new-game menu by visual inspection, while the pixel heuristic labels it battle-like. At that sample, 0x0014=0, 0x03CD=0, 0x00D2=73, and 0x0270=248.

## Decision

Keep RAM rewards and state transitions unchanged. Next, independently label
several battle entry/exit/result frames, including menus and title/new-game
screens, then test a small shortlist (`0x0014`, `0x03CD`, `0x00D2`, `0x0270`)
against those observations. Resolve whether `0x03CD` and neighboring bytes are
correlated before selecting an address. All published metrics are sanitized
summaries; raw RAM, full analysis CSVs, ROMs, saves, and models are excluded.
