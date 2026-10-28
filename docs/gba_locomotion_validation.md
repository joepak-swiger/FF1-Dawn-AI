# GBA locomotion validation runbook

The GBA position experiment is evaluation-only. It uses the official
Sif/Warrior, Myth/Thief, Milo/Monk, Seed/Red Mage native Stable-Retro checkpoint
and does not change the checkpoint, ROM, rewards, or models.

## Running a bounded probe

From the repository's existing WSL virtual environment:

```bash
cd /mnt/e/JOEPAK/FF1-Dawn-AI
source .venv/bin/activate
python scripts/gba_locomotion_probe.py --dry-run
python scripts/gba_locomotion_probe.py --repeats 3 --blocked-frames 128
```

The default resolves the latest locally verified Seed-party checkpoint. A
specific state can be selected with `--checkpoint`. Each run writes to a new
ignored `videos/gba_locomotion/<run-id>/` directory. Output includes per-reset
frames, short held-direction responses, matched no-op samples, opposite
direction return paths, a bounded north route, a JSONL trace, raw local RAM
samples, metrics, and generated contact sheets. Existing run directories are
never overwritten.

After reviewing the frames and metrics, prepare a GitHub-safe package with:

```bash
python scripts/package_gba_locomotion_evidence.py videos/gba_locomotion/<run-id>
```

The packager fails closed when the checkpoint changed, resets were not
repeatable, fewer than three independent trials ran, or proof frames are
missing. It exports only whitelisted metrics and small contact sheets. Review
the resulting `docs/experiments/gba_locomotion/<run-id>/` before staging.

## Interpretation limits

- Phase correlation estimates rendered scene translation. A stable directional
  displacement can support that movement occurred, but it does not directly
  measure a player tile or world coordinate.
- Exact screen hashes are for repeatability only. Sprite animation, facing,
  transitions, camera behavior, or changed scenery can alter or preserve a
  hash independently of logical position.
- RAM candidate offsets must differ from a matched no-op, reproduce across
  resets, distinguish cardinal directions, return after reverse movement, and
  distinguish at least two visually verified starting positions before they
  can be treated as position signals.
- A movement plateau is not proof of blocked terrain. Confirm the obstacle
  visually and compare repeated input, no-op, and reverse controls in the same
  scene.
- Do not use provisional movement/RAM signals in reward shaping or training.
