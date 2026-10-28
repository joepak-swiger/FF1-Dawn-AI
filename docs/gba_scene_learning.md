# GBA scene gate and Castle-entry learning smoke

The GBA track uses reviewed visual anchors for the Cornelia exterior field,
Cornelia title/town screen, Castle Cornelia 1F title, and the party/status
menu. It leaves unsupported scenes as `unknown`. A title-strip match from one
frame is only a candidate. `SceneConfirmationGate` confirms a scene after two
consecutive emulator observations, clears the streak after uncertain or
mismatching frames, and emits a first Castle-entry event only after a
confirmed exterior field.

The base [FF1GbaEnv](../ff1_dawn_ai/gba_env.py) remains neutral-reward. The
optional [FirstCastleEntryReward](../ff1_dawn_ai/gba_goal_reward.py) wrapper is
disabled unless explicitly created with `enabled=True`. It awards one point
and truncates the goal episode only on the first temporally confirmed
field-to-Castle transition. It does not read RAM, infer coordinates, or reward
screen changes.

## Run bounded checks

Use the existing WSL virtual environment and verified local Seed-party
checkpoint. These commands do not start a long training experiment:

```bash
python -m unittest discover -s tests
python scripts/gba_scene_eval.py --dry-run
python scripts/gba_scene_eval.py
python scripts/gba_castle_learning_smoke.py --dry-run
python scripts/gba_castle_learning_smoke.py --timesteps 2048 --episodes 3
```

The live scene evaluator repeats entry and exit from three independent native
checkpoint restores, probes three seeded random routes and opens the party
menu as a negative control. The learning smoke compares scripted UP, seeded
random, idle, and a small CPU PPO agent using identical eight-frame actions and
a 32-decision horizon. The PPO budget is capped at 4,096 timesteps. The
scripted route is a reference only; a matching sequence is not learned
navigation.

Runs and PPO archives stay under ignored `videos/` folders. After reviewing
the results, publish only a compact subset:

```bash
python scripts/package_gba_scene_learning_evidence.py <scene-run-id> <learning-run-id>
```

The packager refuses to overwrite existing evidence. It publishes aggregate
metrics, a short report, selected small proof frames, and a compact contact
sheet. ROMs, native states, raw action traces, and PPO weights remain local.

## Current validated boundary

Run `20261028_055309_034925Z` confirmed exterior reset, Castle entry, and
Castle exit on 3/3 fixed-start repeats. The Cornelia town title was confirmed
in 1/3 seeded routes; the other two did not visit town. The party/status menu
was confirmed without a Castle-entry event. Title/new-game, transition, and
fade fixtures were negative controls. These are specific Cornelia anchors,
not a general classifier for battles, outcomes, game over, or story progress.

The 2,048-step PPO smoke `20261028_055426_707460Z` collected zero goal events.
Scripted UP succeeded 3/3, while random, idle, and PPO succeeded 0/3. PPO
selected A in all 96 evaluation actions. This demonstrates that the sparse
reward and current observation setup do not yet teach the route; do not
increase training time or call it a learned policy. Improve sample efficiency
through a transparent demonstration or action abstraction, then evaluate on
held-out routes and states.
