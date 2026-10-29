# First supervised combat policy — experimental foundation

**Status (2026-10-29):** The recorder, grouped-split trainer, model format and opt-in inference path have been implemented. **They have NOT yet been executed on the owner's live WSL gameplay. No real GBA combat policy was trained in this change.** This is a partial milestone toward issue #19, not completion of that issue.

## Why these restrictions exist

The existing 11 winning fights use the same Goblin species and scripted **A → A (Attack/target)** decisions. Outcome evidence is genuine, but repeatedly cloning A does not demonstrate learned combat strategy. The supervised trainer deliberately refuses training if all *training* actions are identical or fewer than five encounter-parent groups exist. Groups prevent replays of the same battle state from leaking into validation/test splits.

### Real collection (in the existing working WSL venv)

    python scripts/gba_autoplay.py --policy autonomous_session --combat-policy scripted --attack-on-confirmed-battle --record-combat-demos --decisions 6000 --max-battles 4

The output directory is unique under \`videos/gba_autoplay/<run_id>/\`. Its **local-only** \`expert_decisions/decisions.jsonl\` contains pre-action observations, scene stage, actual expert button, encounter-parent SHA, per-image checksums and action provenance. **Only examples from battles with independently confirmed EXP victory banners are written.** Recordings with no confirmed victory do not become positive examples. The data remains ignored and must not be committed.

The current recorder captures only *meaningful* Attack/target confirmations, **not thousands of NOOP wait frames**. To obtain multiple action classes, Codex must independently verify and collect genuinely successful alternatives such as battle menu recovery, legal target changes or other controlled combat interactions. Do **not** manufacture B/DOWN labels or mix incompatible encounter states solely to make training run.

### Train once sufficient verified decisions exist

    python scripts/gba_train_combat_bc.py --dataset videos/gba_autoplay/<run_id>/expert_decisions --model models/gba_combat_bc.npz

\`models/gba_combat_bc.npz\` and the adjacent \`.metrics.json\` remain local. The trainer uses balanced linear softmax supervised imitation over 20×20 resized RGB observations plus a scene-stage one-hot; it splits by original encounter-parent group, chooses its training epoch by validation accuracy/loss, and reports one held-out action accuracy. It is **not** PPO or reinforcement learning; offline action accuracy is not gameplay success. As currently implemented, this command trains on **one dataset run folder**; future collectors can consolidate verified groups with checksums into a larger dataset under ignored local directories.

### Actual closed-loop inference (after a model exists)

    python scripts/gba_autoplay.py --policy autonomous_session --combat-policy learned --combat-model models/gba_combat_bc.npz --decisions 6000 --max-battles 4

Only the Attack and target **button decisions** use the learned model. Encounter identification, animation waits, phase transitions, result acknowledgement and postbattle safety checks remain scripted **supervision**; field exploration remains seeded-random. The trace's \`source\` column records \`learned\`, \`scripted_expert\`, \`safety_supervisor\` or blank for earlier unrelated frames. The public metrics include counts for combat decision sources. Unexpected learned actions are **not** silently replaced with A: unsupported choices cause a safe stop when expected next menus are not confirmed.

**Next required live milestone:** Codex runs real demo collection with varied legal combat decisions; verifies labels and split independence; trains a local model; runs same-start scripted vs learned vs random closed-loop evaluations; records explicit \`LEARNED\` and \`SCRIPTED\` GIFs and compares encounter wins, incorrect input sequences and source-attributed decisions. Capture actual reports before claiming any real learning. Do not alter official Sif/Myth/Milo/Seed checkpoint or NES v0.7.
