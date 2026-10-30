# Local project storage audit

Final audit timestamp: `2026-10-30T03:21:23Z` (Windows PowerShell, full E:
checkout). The final inventory is local and ignored under
`reports/storage_audit/20261030_032102_8090Z/`. That report folder was written
after filesystem enumeration and is outside its own snapshot.

## Measured footprint

The final enumeration snapshot contained **9,608,440,231 bytes (about 8.95
GiB) in 57,599 files**. The repository's online size is not a measure of this
local checkout. Git status accounting:

| Category | Files | Bytes | Interpretation |
| --- | ---: | ---: | --- |
| Ignored local data/runtime | 55,315 | 9,539,031,556 | `.venv`, gameplay runs, checkpoints, models, ROMs, and local captures |
| Tracked source/evidence | 323 | 9,437,462 | Project files kept under version control |
| Untracked files | 1,249 | 41,973,810 | Local experiment/documentation history; preserved |
| Git metadata | 712 | 17,997,403 | Local object database and metadata |

Largest top-level consumers were `.venv` **6,451,887,284 B**, `videos`
**1,836,932,561 B**, `checkpoints` **885,399,413 B**, and `models`
**302,422,426 B**. The largest individual files are CUDA/runtime and
Torch/Triton libraries. They are dependencies in the working WSL runtime, so
they were kept. ROMs (17,039,376 B), states (11,794,322 B), and Git data were
also kept. The official party/checkpoint, learned weights, training data,
milestone recordings, and NES v0.7 assets were not changed.

## Cleanup performed

The first safe-clean pass removed **116 regenerable bytecode files / 1,673,287
bytes** under `ff1_dawn_ai/`, `scripts/`, and `tests/`. The full test run
regenerated 99 source-tree bytecode files (1,466,273 B); these were removed
after tests with exact path and directory-name checks. The final audit found
zero remaining cache proposals. Windows free space increased by 1,695,744 B
across the post-test removal. The final audit itself changed free space from
794,114,617,344 B to 794,106,908,672 B because it wrote local inventory
reports; this report-generation cost is not a cleanup saving.

No run, script, ROM, emulator state, model, data set, video, or historical
experiment was deleted, archived, or quarantined. Same-hash and same-size
groups were evidence for review only; duplicates were not removed because
they may be independent proof copies. Hashing read 16,777,116 B within the
configured cap and skipped 5,190 same-size groups. The local manifest records
cleanup proposals with purpose, regeneration, and rollback. `.venv`, `.git`,
`roms/`, `states/`, `models/`, `checkpoints/`, and `videos/` are excluded from
automatic deletion.

## Retention decision

The ignored local artifacts are the working runtime and experiment history,
not proven disposable duplicates. Keep them until a future manifest links a
candidate to a verified replacement and identifies the preserved copy. New
experiments continue to use unique run IDs; full output remains local while
small reviewed summaries are published under `docs/experiments/`.
