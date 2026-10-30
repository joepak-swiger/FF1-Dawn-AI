# Local project storage audit

Final audit timestamp: `2026-10-30T03:25:21Z` (Windows PowerShell, full E:
checkout). The inventory and detailed manifest are local and ignored under
`reports/storage_audit/20261030_032500_4686Z/`. This output directory was
created after enumeration and is not counted in its own snapshot.

## Measured footprint

The final pre-clean enumeration measured **9,617,137,711 bytes (about 8.96
GiB) across 57,095 files**. GitHub's reported repository size does not measure
this local checkout.

| Category | Files | Bytes | Interpretation |
| --- | ---: | ---: | --- |
| Ignored local data/runtime | 55,418 | 9,548,517,853 | `.venv`, gameplay runs, checkpoints, models, ROMs, local captures |
| Tracked source/evidence | 349 | 9,685,219 | Files under version control |
| Untracked files | 1,223 | 41,727,496 | Local experiment/documentation history; preserved |
| Git metadata | 105 | 17,207,143 | `.git` files counted in the checkout |

Largest top-level consumers were `.venv` **6,451,887,284 B**, `videos`
**1,836,932,561 B**, `checkpoints` **885,399,413 B**, and `models`
**302,422,426 B**. `.venv` is the existing WSL runtime with large CUDA,
Torch, and Triton libraries. The largest single file was
`libcublasLt.so.13` (541,607,888 B); `libtriton.so` (505,651,000 B),
`libtorch_cuda.so` (502,842,281 B), and `libtorch_cpu.so` (446,314,593 B)
followed. These dependencies were retained. ROMs (17,039,376 B), states
(11,794,322 B), models, checkpoints, recordings, datasets, and NES v0.7 assets
were also retained.

Largest suffix groups were PNG (15,719), Python bytecode (9,976), Python
source (9,976), C headers (9,948), JSON (3,731), and native state files
(2,613). The oldest files were in the working `.venv` from September 28,
2026. The newest files in the pre-clean snapshot were Python bytecode created
by the just-completed test run at 03:24:55 UTC; they were immediately removed
by the safe-clean pass.

## Cleanup performed

`audit_project_storage.ps1 -SafeClean` removed **99 regenerable bytecode files
/ 1,467,499 bytes** from source-tree `__pycache__` folders. Estimated project
bytes after this removal were **9,615,670,212 B**. No cleanup proposal
remained after the run. The audit wrote several megabytes of local reports;
Windows free space therefore moved from 794,106,052,608 B to
794,100,097,024 B during this pass. This is report-generation cost, not a
cleanup saving.

An earlier safe-clean pass in this task also removed 116 bytecode files
(1,673,287 B); test runs recreated caches, which were removed again after
testing. These are regenerable Python artifacts, not unique data.

No run, source script, ROM, emulator state, model, dataset, video, or historical
experiment was deleted, archived, or quarantined. Same-hash/same-size groups
were review candidates only and were not removed. The final pass hashed
16,774,617 B within its configured budget, found 359 exact duplicate groups,
and skipped 5,143 same-size groups. The local manifest contains the top 30
largest, oldest, and newest files plus suffix and status breakdowns. Re-run
`scripts/audit_project_storage.ps1` without flags for a read-only snapshot;
automatic cleanup is limited to explicit source-tree Python caches. `.venv`,
`.git`, `roms/`, `states/`, `models/`, `checkpoints/`, and `videos/` are never
automatic deletion targets.

## Retention decision

The ignored local artifacts are the working runtime and experiment history,
not proven disposable duplicates. Keep them until a future manifest links a
candidate to a verified replacement and identifies the preserved copy. New
experiments continue to use unique run IDs; full output remains local while
small reviewed summaries are published under `docs/experiments/`.
