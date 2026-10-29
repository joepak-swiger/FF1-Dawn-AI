"""Frozen, newly routed GBA navigation benchmark with train/validation/final splits.

It derives independent start checkpoints from the official local Seed-party
state, trains only on cardinal route demonstrations plus prior exposed
correction labels, selects only on new validation families, then evaluates a
predeclared final pair once. The final routes are not used for candidate search.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ff1_dawn_ai.gba_navigation import NearestNeighborBehaviorCloner  # noqa: E402
from scripts.gba_navigation_imitation import (  # noqa: E402
    ROUTE_SPECS, create_starts, discover_checkpoint, evaluate, make_classifier,
    collect_demonstration,
)

TRAIN_IDS = ("center", "west_near", "west_far", "east_near", "east_far",
             "south_near", "south_far")
VALIDATION_ROUTES = {
    "val_east_north_loop": ["RIGHT"] * 4 + ["UP"] * 4 + ["LEFT"] * 2 + ["DOWN"] * 2,
    "val_west_north_loop": ["LEFT"] * 4 + ["UP"] * 4 + ["RIGHT"] * 2 + ["DOWN"] * 2,
}
FINAL_ROUTES = {
    "test_east_reversal_loop": ["RIGHT"] * 4 + ["DOWN"] * 4 + ["UP"] * 2 + ["LEFT"] * 2,
    "test_west_reversal_loop": ["LEFT"] * 4 + ["DOWN"] * 4 + ["UP"] * 2 + ["RIGHT"] * 2,
}
POLICIES = ("IDLE", "SEEDED_RANDOM", "ALWAYS_UP", "SCRIPTED",
            "BC_ORIGINAL_3NN", "BC_RECOVERY_AUGMENTED_1NN")


def make_specs() -> dict[str, list[str]]:
    return {key: ROUTE_SPECS[key] for key in TRAIN_IDS} | VALIDATION_ROUTES | FINAL_ROUTES


def validate_split(specs: dict[str, list[str]]) -> None:
    if set(specs) != set(TRAIN_IDS) | set(VALIDATION_ROUTES) | set(FINAL_ROUTES):
        raise ValueError("benchmark route split is incomplete or contains extra route families")
    if set(VALIDATION_ROUTES) & set(FINAL_ROUTES):
        raise ValueError("validation and final route families overlap")
    for route in specs.values():
        if any(action not in ("UP", "DOWN", "LEFT", "RIGHT") for action in route):
            raise ValueError("route contains a non-directional action")


def run(replicates: int = 2, max_decisions: int = 48) -> Path:
    specs = make_specs()
    validate_split(specs)
    source_benchmark = ROOT / "videos/gba_navigation/20261028_153708_566851Z"
    recovery_data = ROOT / "videos/gba_recovery_imitation/20261029_014031_526568Z/training_data_local.npz"
    if not recovery_data.is_file():
        raise FileNotFoundError(f"required local recovery labels unavailable: {recovery_data}")
    out = ROOT / "videos/gba_navigation_fresh_benchmark" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir()
    started = time.perf_counter()
    classifier = make_classifier()
    official = discover_checkpoint()
    starts, rejected = create_starts(official, out, classifier, specs=specs)
    start_map = {row["start_id"]: row for row in starts}
    expected = set(specs)
    if expected - set(start_map):
        partial = {"run_id": out.name, "status": "start_creation_incomplete",
                   "accepted_starts": sorted(start_map), "rejected": rejected,
                   "missing": sorted(expected - set(start_map)), "rl_timesteps": 0}
        (out / "metrics.json").write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
        pubdir = ROOT / "docs/experiments/gba_navigation_fresh_benchmark" / out.name
        pubdir.mkdir(parents=True, exist_ok=False)
        public = {"run_id": out.name, "status": "start_creation_incomplete",
                  "accepted_start_ids": sorted(start_map),
                  "missing_start_ids": sorted(expected - set(start_map)),
                  "rejected_start_scenes": [{"start_id": row["start"],
                    "independent_field_confirmation": row.get("confirmed_field", False),
                    "classification_labels": row.get("labels", [])} for row in rejected],
                  "policy_evaluations_started": False, "validation_selection": None,
                  "final_policy_evaluation_started": False, "rl_timesteps": 0,
                  "limitation": "The sparse scene anchor set could not independently verify every generated start after reset; no policy outcomes were collected."}
        (pubdir / "metrics.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
        report = [f"# Fresh GBA route benchmark start audit — {out.name}", "",
                  "Status: **incomplete; no policy evaluation was run**. The required independent reset/field-start gate rejected some generated routes, so no model was selected and the final test was never opened.", "",
                  f"Accepted starts: {', '.join(sorted(start_map)) or 'none'}.",
                  f"Unverified starts: {', '.join(sorted(expected - set(start_map)))}.", "",
                  "Several screenshots were visually recognizable as the Cornelia exterior, but the conservative two-frame scene gate returned `unknown` on most subsequent observations. Other attempts encountered frames outside the reviewed anchors. This is evidence of sparse scene coverage and possible random encounter variation, not a verified gameplay failure.", "",
                  "No PPO, policy updates, rewards, or final-test outcomes were produced. Native states and raw captures remain local."]
        (pubdir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        raise RuntimeError(f"could not establish all frozen starts; see {out / 'metrics.json'}")
    (out / "frozen_split.json").write_text(json.dumps({"train": list(TRAIN_IDS),
        "validation": list(VALIDATION_ROUTES), "final_test": list(FINAL_ROUTES),
        "route_specs": specs,
        "final_policy_outcomes_evaluated_after_validation_selection": True,
        "final_route_views_prepared_before_selection": True}, indent=2)+"\n", encoding="utf-8")

    # Recollect demonstrations only from the frozen cardinal training split.
    demo_obs, demo_actions, demo_reports = [], [], []
    for start_id in TRAIN_IDS:
        obs, actions, result = collect_demonstration(start_map[start_id], out, classifier,
                                                      max_decisions=max_decisions)
        if result["success"]:
            demo_obs.extend(obs)
            demo_actions.extend(actions)
        demo_reports.append({"start_id": start_id, "success": result["success"],
                             "decisions": result["decisions"]})
    if not demo_actions:
        raise RuntimeError("no successful cardinal training demonstrations")
    np.savez_compressed(out / "training_data_local.npz", observations=np.stack(demo_obs),
                        actions=np.asarray(demo_actions, dtype=np.int64))
    original = NearestNeighborBehaviorCloner(3, 4).fit(demo_obs, demo_actions)

    # Prior south-diagonal states are already exposed and are now training-only.
    # The new final families remain excluded from training and validation.
    correction = np.load(recovery_data)
    augmented_obs = demo_obs + list(correction["correction_observations"])
    augmented_actions = demo_actions + list(correction["correction_actions"])
    augmented = NearestNeighborBehaviorCloner(1, 4).fit(augmented_obs, augmented_actions)

    rows = []
    split_names = [("validation", list(VALIDATION_ROUTES))]
    for split, ids in split_names:
        for repeat in range(replicates):
            for index, start_id in enumerate(ids):
                for policy in POLICIES:
                    model = original if policy == "BC_ORIGINAL_3NN" else augmented if policy == "BC_RECOVERY_AUGMENTED_1NN" else None
                    row = evaluate(start_map[start_id], policy, classifier, model=model,
                                   seed=4100 + repeat * 100 + index * 10 + POLICIES.index(policy),
                                   max_decisions=max_decisions, output_dir=out)
                    row.update(split=split, repeat=repeat + 1)
                    rows.append(row)
                    (out / "evaluation.partial.jsonl").open("a", encoding="utf-8").write(json.dumps(row)+"\n")
                    print(f"{split}/{start_id} r{repeat+1} {policy}: {row['goal_reached']}", flush=True)
    val_rows = [row for row in rows if row["split"] == "validation"]
    learned = ["BC_ORIGINAL_3NN", "BC_RECOVERY_AUGMENTED_1NN"]
    validation_scores = {policy: sum(r["goal_reached"] for r in val_rows if r["policy"] == policy)
                         for policy in learned}
    selected = max(learned, key=lambda p: validation_scores[p]) if max(validation_scores.values(), default=0) else None
    (out / "validation_selection.json").write_text(json.dumps({"selected": selected,
        "scores": validation_scores, "rule": "max repeated validation Castle successes; tie prefers original 3-NN",
        "final_test_used_for_selection": False}, indent=2)+"\n", encoding="utf-8")

    # Reveal the frozen final families only after validation selection is fixed.
    for repeat in range(replicates):
        for index, start_id in enumerate(FINAL_ROUTES):
            for policy in POLICIES:
                model = original if policy == "BC_ORIGINAL_3NN" else augmented if policy == "BC_RECOVERY_AUGMENTED_1NN" else None
                row = evaluate(start_map[start_id], policy, classifier, model=model,
                               seed=8100 + repeat * 100 + index * 10 + POLICIES.index(policy),
                               max_decisions=max_decisions, output_dir=out)
                row.update(split="final_test", repeat=repeat + 1)
                rows.append(row)
                (out / "evaluation.partial.jsonl").open("a", encoding="utf-8").write(json.dumps(row)+"\n")
                print(f"final_test/{start_id} r{repeat+1} {policy}: {row['goal_reached']}", flush=True)
    metrics = {"run_id": out.name, "official_party": "Sif/Warrior, Myth/Thief, Milo/Monk, Seed/Red Mage",
               "source_demo_run_id": source_benchmark.name,
               "recovery_label_source_run_id": "20261029_014031_526568Z",
               "split": {"train": list(TRAIN_IDS), "validation": list(VALIDATION_ROUTES),
                         "final_test": list(FINAL_ROUTES)},
               "start_creation_rejections": rejected, "training_demonstrations": demo_reports,
               "base_action_labels": len(demo_actions), "recovery_action_labels": int(len(correction["correction_actions"])),
               "validation_scores": validation_scores, "validation_selected_policy": selected,
               "final_test_selection_leak": False, "policies": list(POLICIES),
               "results": rows, "rl_timesteps": 0,
               "elapsed_seconds": round(time.perf_counter()-started, 3),
               "limitations": ["New route families are distinct sequences on the same Cornelia map, not new maps or verified coordinates.",
                               "Castle entry is the only goal signal; no field position, battles, or progression are measured.",
                               "The final set was predeclared and scored only after validation selection."]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    pubdir = ROOT / "docs/experiments/gba_navigation_fresh_benchmark" / out.name
    pubdir.mkdir(parents=True, exist_ok=False)
    aggregate = []
    for policy in POLICIES:
        by_split = {}
        for split in ("validation", "final_test"):
            selected_rows = [r for r in rows if r["split"] == split and r["policy"] == policy]
            by_split[split] = {"successes": sum(r["goal_reached"] for r in selected_rows),
                               "episodes": len(selected_rows),
                               "mean_decisions": round(float(np.mean([r["decision_count"] for r in selected_rows])), 2) if selected_rows else None,
                               "action_counts": {name: sum(r["action_counts"][name] for r in selected_rows)
                                                 for name in ("UP", "DOWN", "LEFT", "RIGHT", "NOOP")}}
        aggregate.append({"policy": policy, "by_split": by_split})
    public = {"run_id": out.name, "official_party": metrics["official_party"],
              "source_demo_run_id": metrics["source_demo_run_id"], "split": metrics["split"],
              "training_demonstration_successes": sum(x["success"] for x in demo_reports),
              "training_demonstrations": len(demo_reports), "base_action_labels": len(demo_actions),
              "recovery_action_labels": metrics["recovery_action_labels"],
              "validation_scores": validation_scores, "validation_selected_policy": selected,
              "final_test_selection_leak": False, "results": aggregate,
              "rl_timesteps": 0, "elapsed_seconds": metrics["elapsed_seconds"],
              "limitations": metrics["limitations"]}
    (pubdir / "metrics.json").write_text(json.dumps(public, indent=2)+"\n", encoding="utf-8")
    best = selected or "none (validation had no successful imitation policy)"
    report = [f"# Fresh GBA navigation benchmark — {out.name}", "",
              f"Validation selection: **{best}**. Final route start views are prepared before validation for scene/reset verification, but final policy outcomes are recorded only after selection; no final outcome influences model choice.",
              "", "The route families are new directional sequences and view starts on the Cornelia map. They do not represent new maps or independently measured tile coordinates.", "",
              f"Training demonstrations: {sum(x['success'] for x in demo_reports)}/{len(demo_reports)}; base labels: {len(demo_actions)}.", "",
              "| Policy | Validation | Final test |", "|---|---:|---:|"]
    for item in public["results"]:
        report.append(f"| {item['policy']} | {item['by_split']['validation']['successes']}/{item['by_split']['validation']['episodes']} | {item['by_split']['final_test']['successes']}/{item['by_split']['final_test']['episodes']} |")
    report += ["", f"Training used {len(demo_actions)} cardinal demonstration action labels plus {metrics['recovery_action_labels']} previously collected recovery labels for the augmented comparator. PPO/RL timesteps: 0.",
               "", "Native checkpoints, ROMs, full traces and training arrays remain local. The report publishes aggregate outcomes only."]
    (pubdir / "REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(out)
    print(pubdir)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=48)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.replicates <= 3 or not 24 <= args.max_decisions <= 80:
        parser.error("replicates 1..3 and max-decisions 24..80 are supported")
    specs = make_specs()
    validate_split(specs)
    if args.dry_run:
        print(json.dumps({"runtime_not_started": True, "train": list(TRAIN_IDS),
                          "validation": list(VALIDATION_ROUTES), "final_test": list(FINAL_ROUTES),
                          "unique_direction_sequences": len({tuple(x) for x in specs.values()}),
                          "replicates": args.replicates, "max_decisions": args.max_decisions}, indent=2))
        return 0
    run(args.replicates, args.max_decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
