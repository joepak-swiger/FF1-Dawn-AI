# NES RAM verification

`scripts/nes_ram_probe_v0_9.py` ranks RAM bytes correlated with a visual
classifier. Its results are hypotheses. `scripts/nes_ram_verify_v1_0.py`
collects manually labeled gameplay samples to check those hypotheses.

## Run

From the WSL project directory with the existing virtual environment active:

```bash
python scripts/nes_ram_verify_v1_0.py
```

The local NES ROM integration and `states/ff1_nes_cornelia_start.state` must
already be available. The script uses the existing `FF1NesEnv`, but does not
load or modify any PPO model.

## Controls and labeling

- Arrow keys move; hold a key to repeat an action about every 100 ms.
- `Z` or `A` presses the game's A button; `X` or `S` presses B.
- `F1` labels field/overworld, `F2` battle, `F3` menu/town, `F4` other.
- `F5` saves a screenshot. `F6` clears the label while the state is uncertain.
- `Q` closes the environment and writes the reports.

Press a label key whenever the game has visibly entered a new state. A label
key starts a new label epoch and saves a frame plus the current RAM sample.
Clear the label during uncertain transition animations. Samples gathered while
a label remains active are grouped under that epoch. Label epochs help reveal
whether a result repeated over multiple intervals, but are not independent
emulator runs and do not verify that the human label was correct.

For useful evidence, label the initial field state, an encounter and battle,
the battle result/exit, and the returned field state. Repeat for several
battles. For stronger independence, run the verifier again from its normal
Cornelia reset and repeat the sequence. Also label menus/town and any game-over
state encountered. Do not infer that `0x0014` (or any other byte) is a battle
flag from a single run.

## Outputs and interpretation

Generated files are ignored under `videos/nes_ram_verify_v1_0/`:

- `manual_ram_events.csv`: label events and action samples, including label
  epoch, action, and candidate byte values.
- `candidate_address_label_summary.csv`: per-address modal values, fractions,
  and epoch counts by label.
- `metrics.json` and `run_report.txt`: concise run summary and candidate
  signal ranking.
- `frames/`: labeled, manual, and periodic proof screenshots.

The `separation_score` ranks modal-value separation between the manually
labeled field and battle samples. It is not a statistical confidence value,
and repeated samples in one epoch are correlated. Before using a byte in an
environment reward or state detector, confirm it across multiple verifier
runs and check field, battle, menu/town, post-battle, and game-over states.
This milestone intentionally does not change PPO rewards or start training.
