# First supervised combat policy — experimental foundation

**Status (2026-10-29):** The recorder, grouped-split trainer, model format and opt-in inference path have now been exercised in the working WSL emulator. A local supervised model was trained from six encounter-parent groups and deployed in both a held-out Magic-menu branch and a complete three-encounter field session. This is a narrow behavior-cloning result, not RL or a general combat policy; see `docs/experiments/gba_combat_learning/20261029_issue19_live/`.

## Why these restrictions exist

The existing 11 winning fights use the same Goblin species and scripted **A → A (Attack/target)** decisions. Outcome evidence is genuine, but repeatedly cloning A does not demonstrate learned combat strategy. The supervised trainer deliberately refuses training if all *training* actions are identical or fewer than five encounter-parent groups exist. Groups prevent replays of the same battle state from leaking into validation/test splits.

### Real collection (in the existing working WSL venv)

    python scripts/gba_autoplay.py --policy autonomous_session --combat-policy scripted --attack-on-confirmed-battle --record-combat-demos --decisions 6000 --max-battles 4

The output directory is unique under \`videos/gba_autoplay/<run_id>/\`. Its **local-only** \`expert_decisions/decisions.jsonl\` contains pre-action observations, scene stage, actual expert button, encounter-parent SHA, per-image checksums and action provenance. **Only examples from battles with independently confirmed EXP victory banners are written.** Recordings with no confirmed victory do not become positive examples. The data remains ignored and must not be committed.

The current recorder captures only *meaningful* Attack/target confirmations, **not thousands of NOOP wait frames**. To obtain multiple action classes, Codex must independently verify and collect genuinely successful alternatives such as battle menu recovery, legal target changes or other controlled combat interactions. Do **not** manufacture B/DOWN labels or mix incompatible encounter states solely to make training run.

### Train from multiple successful captures

    python scripts/gba_train_combat_bc.py --dataset videos/gba_autoplay/<run_id>/expert_decisions videos/gba_combat_branch_eval/<run_id>/expert_decisions --model models/gba_combat_bc.npz

\`models/gba_combat_bc.npz\` and the adjacent \`.metrics.json\` remain local. The trainer accepts multiple folders, deduplicates exact replay observations, and keeps rows with the same encounter-parent ID in one split. If a later menu-state replay was captured from an existing encounter state, pass a JSON \`--group-map\` mapping that replay-state hash to its original battle-command-state hash before splitting; otherwise sibling replays could leak across train and test. The trainer uses balanced linear softmax imitation over 20×20 resized RGB observations plus a scene-stage one-hot and chooses its epoch by validation accuracy/loss. It is not PPO or reinforcement learning.

### Actual closed-loop inference (after a model exists)

    python scripts/gba_autoplay.py --policy autonomous_session --combat-policy learned --combat-model models/gba_combat_bc.npz --decisions 6000 --max-battles 4

Only the Attack and target **button decisions** use the learned model. Encounter identification, animation waits, phase transitions, result acknowledgement and postbattle safety checks remain scripted **supervision**; field exploration remains seeded-random. The trace's \`source\` column records \`learned\`, \`scripted_expert\`, \`safety_supervisor\` or blank for earlier unrelated frames. The public metrics include counts for combat decision sources. Unexpected learned actions are **not** silently replaced with A: unsupported choices cause a safe stop when expected next menus are not confirmed.

**Current measured scope:** Six encounter-parent groups produced 53 deduplicated samples. The learned model selected B on a held-out Magic-menu view, then selected A for Attack/target confirmations and won the confirmed Goblin battle. From that exact state, the scripted comparator also won and seeded random stopped safely after one rejected B choice on the Attack panel. The model reproduced the expert behavior; it has not shown better outcomes than the script. Three-encounter autonomous runs under the same seed yielded three confirmed wins for both learned and scripted combat, with movement resumed after the first two and a safe stop after the third unrecognized post-battle map view. The held-out accuracy was 100% on an eight-sample test group containing only A labels; treat that score as a weak diagnostic, not evidence of broad class generalization.

The field controller remains seeded-random, battle recognition is validated only for the reviewed Goblin layouts, and result detection covers reviewed EXP/Gil templates. No HP/survival model, other enemy species, general battle result parser, PPO, reward shaping, or model superiority has been established. Keep the trained model local and ignored. Do not alter official Sif/Myth/Milo/Seed checkpoints or NES v0.7.

**Next live milestone:** Expand demonstrations and held-out checks to a second enemy species and distinct target/menu layouts, then add independently validated survival/HP observations. Continue comparing learned and scripted decision attribution before claiming improvement.
