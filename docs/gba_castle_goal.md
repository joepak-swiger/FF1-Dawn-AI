# GBA Castle Cornelia goal evaluation

This evaluation checks a concrete navigation objective from the official
Sif/Warrior, Myth/Thief, Milo/Monk, Seed/Red Mage native checkpoint. It does
not train an agent or change rewards.

## Run the experiment

From the existing WSL virtual environment:

```bash
cd /mnt/e/JOEPAK/FF1-Dawn-AI
source .venv/bin/activate
python scripts/gba_castle_eval.py --dry-run
python scripts/gba_castle_eval.py --repeats 3 --horizon 192
```

Each run writes to a unique ignored `videos/gba_castle_goal/<run-id>/`
directory. It compares scripted UP, 12 seeded-random routes, idle, and DOWN
controls. It also captures START UI controls, an 8-frame reversal/release
diagnostic, 64-frame DOWN versus idle, and a bounded castle exit route. Each
action advances one emulator frame. Checkpoint state is restored natively for
each independent route.

The destination detector uses the reviewed static `Castle Cornelia 1F` title
banner in `docs/experiments/gba_castle_goal/reference_cornelia_1f.png`. It
requires two consecutive high-confidence matches, does not inspect RAM, and
returns `unknown` for every other screen. A missing title during a fade is not
proof of an overworld return; review the retained screenshots.

## Publish a reviewed result

After inspecting `castle_goal_contact_sheet.png`, transition screenshots,
menu controls, and castle-exit frames, package a compact GitHub-safe summary:

```bash
python scripts/package_gba_castle_evidence.py \
  videos/gba_castle_goal/<run-id> --reviewed-field-return
```

Pass `--reviewed-field-return` only after confirming the retained exterior
frame visually shows the overworld. The package contains selected proof frames
and summary metrics; native states, ROMs, raw RAM, detailed traces, and full
capture runs remain local.

## Interpretation limits

- The title anchor verifies the named Castle Cornelia 1F screen. It does not
  classify battles, other castle floors, game-over/title screens, or general
  field/town screens.
- The 1-frame action route cost is a count of emulator input frames, not a
  count of accepted game tiles.
- Phase-correlation output tracks rendered map/camera displacement in pixels,
  not player coordinates or tile units.
- `0x33FC` and `0x3404` remain unexplained RAM offsets. Their shared `226/230`
  pair on the Castle and Cornelia town screens rules them out as a unique
  castle-entry flag in the observed data.
- Random-route success is only a bounded seeded baseline for the specified
  horizon; it is not a general probability estimate.
- See [docs/gba_scene_learning.md](gba_scene_learning.md) for the later
  multi-scene confirmation gate, fresh Castle entry/exit results, optional
  one-time reward wrapper, and bounded PPO feasibility smoke. The base
  Gymnasium environment remains reward-neutral. Do not generalize this visual
  reward to other scenes or start larger training until the agent receives
  goal events and beats random/idle on held-out routes and starts.
