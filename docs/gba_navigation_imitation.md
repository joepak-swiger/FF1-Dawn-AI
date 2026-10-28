# GBA Cornelia navigation imitation

The navigation layer in `ff1_dawn_ai/gba_navigation.py` keeps the full
eleven-button `FF1GbaEnv` intact and exposes an opt-in five-action interface:
NOOP, UP, DOWN, LEFT, RIGHT. Each directional decision holds the button for
eight emulator frames and then releases all buttons for two frames. The
environment returns the original neutral reward.

## Run

Use the existing WSL virtual environment. No package installation or PPO
training is required:

```bash
python scripts/gba_navigation_imitation.py --dry-run
python scripts/gba_navigation_imitation.py --max-decisions 32 --publish
```

Each live run starts from the official local Seed-party checkpoint, creates
four separate ignored native states after bounded cardinal routes, waits 32
neutral frames, and checks a recognized Cornelia exterior across consecutive
observations. It reloads each new state twice and requires identical reset
frames. The original custom-party checkpoint is read-only. Starts that fail
scene recognition or reproducibility checks are rejected and listed in the
local metrics.

The experiment records successful scripted demonstrations as observation /
action pairs and fits a three-nearest-neighbor behavior-cloning baseline. The
training starts are center and west-offset; east and south are held out. The
runner compares idle, seeded random, always-UP, the explicit scripted route,
and behavior cloning, with a 32-decision ceiling by default. A status-menu
negative control checks for false Castle-entry events. Detailed traces, native
states, and local frames remain under ignored `videos/gba_navigation/<run-id>`.
`--publish` writes only aggregate metrics, a short report, and a small contact
sheet under `docs/experiments/gba_navigation_imitation/<run-id>`.

## Current result

Live run `20261028_062140_608666Z` produced four reproducible exterior starts.
All four passed the two-observation exterior scene gate after native reload.
Directional paths and different stable screenshots distinguish them; phase
correlation showed about 48 screen pixels of map shift for each offset. This
is a rendered camera/map signal, not a tile coordinate.

The scripted policy reached Castle Cornelia from 4/4 starts. Always-UP reached
2/4; it failed from west and east. Idle and seeded random reached 0/4. The
imitation policy reached 3/4: it reproduced the demonstrated west route with
four RIGHT actions followed by fourteen UP actions, reached from center with
UP, and also used UP to reach the held-out south start. It failed on the
held-out east turn challenge, selecting RIGHT for all 32 decisions. Therefore
the result shows useful imitation on demonstrated routes, but no generalization
to the opposite-side turn. The south held-out success matched always-UP and is
not evidence of turn generalization.

There were zero false Castle events in the menu control. The run took 54.5
seconds and used 31 demonstrated decisions. Reinforcement-learning timesteps
were zero; no reward shaping or RAM progress signals were used. This is not a
learned general navigation policy. The next useful experiment is to add a
second successful demonstration from the east side (and a held-out route with
a different turn sequence), then rerun this bounded comparison before trying
curriculum or RL.
