# GBA Cornelia navigation imitation

GBA is the primary project target. The verified party remains Sif/Warrior,
Myth/Thief, Milo/Monk, and Seed/Red Mage. The navigation adapter is an opt-in
layer over the unchanged eleven-button `FF1GbaEnv`: NOOP, UP, DOWN, LEFT, and
RIGHT each hold for eight emulator frames and release for two. Base rewards
remain neutral.

## Reproduce the bounded benchmark

Use the existing WSL environment; do not reinstall packages:

```bash
cd /mnt/e/JOEPAK/FF1-Dawn-AI
source .venv/bin/activate
python scripts/gba_navigation_imitation.py --dry-run
python scripts/gba_navigation_imitation.py --max-decisions 48 --replicates 2 --publish
```

Each run creates unique native states under ignored
`videos/gba_navigation/<run-id>/checkpoints/`. It starts from the canonical
Seed-party checkpoint read-only, then checks scene labels, repeated state
restoration, and two-frame Castle entry. The frozen route split is declared in
`scripts/gba_navigation_imitation.py`:

- Train: `center`, `west_near`, `west_far`, `east_near`, `east_far`,
  `south_near`, `south_far`.
- Validation: `east_north`, `west_north`.
- Final test: `east_south`, `west_south`.

These are distinct reproducible rendered views reached from one checkpoint on
the same overworld, not different maps, RNG seeds, or exact tile coordinates.
The reviewed field-anchor images label scenes only; the policies receive
rendered observations and never receive start IDs, route provenance, RAM, or
oracle actions at inference.

The benchmark compares idle, seeded random, always-UP, a route-provenance
script, the original center/west 3-NN ablation, expanded multi-start frame 3-NN,
three-observation temporal 3-NN, and a class-balanced linear classifier.
Validation alone can choose a policy. If every model fails validation, the
report says `none`; final-test results are reported as a frozen comparison and
must not be used to retune the tested models.

The local output includes `metrics.json`, `run_report.txt`, `start_provenance.json`,
`demonstrations.json`, `demonstration_trace.csv`, `demonstration_data.npz`,
`evaluation.csv`, a partial-evaluation journal, checkpoints, and screenshots.
Only the whitelist-generated report, compact metrics, and contact sheet are
published to `docs/experiments/gba_navigation_imitation/<run-id>/`. Raw
observations, traces, native states, ROMs, saves, and model weights remain local.

## Latest verified result

Run `20261028_153708_566851Z` built and restored 11/11 starts and accepted zero
rejected candidates. All seven cardinal demonstration trajectories succeeded
and supplied 127 supervised decisions, including turn-then-UP trajectories
from both east and west. The validation route families both failed for every
policy in two resets (0/4 each); validation therefore selected no model.

On the two frozen south-diagonal route families, always-UP, idle, seeded random,
and the provenance script each scored 0/4. The center/west-only 3-NN scored 2/4
and collapsed to RIGHT for all east-south decisions. Three predeclared learned
variants—expanded frame 3-NN, temporal 3-NN, and balanced linear—each scored
4/4, reaching Castle in 21 decisions (210 emulator frames) on both starts in
both independent reset repeats. The expanded 3-NN issued four LEFT macros then
17 UP macros from east-south, and four RIGHT then 17 UP from west-south. Its
input contains only rendered frames and the 127 cardinal demonstrations. This
is bounded held-out turning generalization across two route families, not proof
of broad exploration competence or independent map generalization.

The menu negative control yielded zero false Castle-entry events. The evidence
package records scene-anchor provenance, per-policy/per-start results, action
counts, and limitations. Training was supervised imitation only; RL timesteps
were zero and no game reward was introduced.

The scene-anchor classifier labeled about 94% of emulator frames in the
moving final-test rollouts `unknown`. Its two-frame Castle detector still
confirmed the target with zero false Castle events, but field labels do not
yet cover dynamic movement frames broadly enough for a general scene or visit
tracker.

## Perturbation and recovery

After the frozen comparison, `scripts/gba_navigation_recovery.py` recollects
the seven successful cardinal demonstrations, refits the fixed expanded
frame-only 3-NN, and stress-tests the same diagonal states. It reverses the
model's current directional suggestion for 0, 1, 2, or 4 macros, then returns
control to the same policy:

```bash
python scripts/gba_navigation_recovery.py --source-run-id 20261028_153708_566851Z --horizon 48 --repeats 2
```

Run `20261028_155424_893532Z` reached Castle 4/4 with no perturbation and 4/4
after one wrong macro. Two wrong macros succeeded 0/4. Four wrong macros
succeeded 2/4 (east-south 2/2, west-south 0/2); west-south collapsed to LEFT.
This is perturb-and-recover evaluation, not DAgger: no expert relabeled the
states visited by the learner. It revisits the same two route families and is
not a new held-out test.

The immediate next step is to collect expert corrections on perturbed states,
validate recovery on fresh diagonal/obstacle route families, and repeat the
held-out comparison before attempting RL. The learned scene gate is not yet
robust during movement (~94% unknown frames), so compound Castle entry/exit and
PPO remain deferred. Do not use speculative RAM coordinates or progress
rewards. The secondary NES v0.7 baseline remains unchanged.
