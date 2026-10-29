"""Collect bounded expert-corrected GBA recovery data from actual policy errors.

The script saves learner-visited native states after two reversed actions,
searches short fixed directional routes from those exact states, and admits
only routes that replay successfully twice under the independent Castle gate.
It fits an augmented supervised 3-NN and evaluates it closed-loop against the
original on a mirrored visited state. These reuse previously exposed starts;
they are not fresh generalization results or formal DAgger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ff1_dawn_ai.gba_landmark import SceneConfirmationGate  # noqa: E402
from ff1_dawn_ai.gba_navigation import (  # noqa: E402
    NAVIGATION_ACTIONS, NearestNeighborBehaviorCloner,
)
from scripts.gba_navigation_imitation import (  # noqa: E402
    FINAL_TEST_START_IDS, collect_demonstration, make_classifier, make_env,
)
from scripts.gba_route_discovery import candidate_routes, route_summary, run_trial  # noqa: E402


def opposite(action: int) -> int:
    pairs = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    name = NAVIGATION_ACTIONS[int(action)]
    return NAVIGATION_ACTIONS.index(pairs[name])


def capture_mistake(start: dict, policy, out: Path, *, decisions_before_error: int) -> tuple[dict, np.ndarray]:
    base, nav = make_env(ROOT / start["checkpoint_local_path"], max_steps=128 * 12)
    try:
        obs, info = nav.reset(seed=1729)
        if not info.get("checkpoint_loaded"):
            raise RuntimeError("source checkpoint was not loaded")
        for _ in range(decisions_before_error):
            action = policy.predict(obs)
            obs, _, terminated, truncated, _ = nav.step(action)
            if terminated or truncated:
                raise RuntimeError("environment ended before scheduled recovery perturbation")
        trace = []
        for index in range(2):
            suggested = policy.predict(obs)
            action = opposite(suggested)
            obs, _, terminated, truncated, step_info = nav.step(action)
            trace.append({"index": index, "policy_suggested": NAVIGATION_ACTIONS[suggested],
                          "executed_wrong_action": NAVIGATION_ACTIONS[action],
                          "emulator_frames": int(step_info.get("macro_emulator_frames", 0))})
            if terminated or truncated:
                raise RuntimeError("environment ended during controlled perturbation")
        state = base.retro_env.em.get_state()
        frame = nav.render()
        tag = start["start_id"] + "_offset" + str(decisions_before_error) + "_two_wrong"
        state_path = out / (tag + ".state")
        state_path.write_bytes(state)
        Image.fromarray(frame).save(out / (tag + "_at_error.png"))
        # Reproduce the exact state-load synchronization used by FF1GbaEnv.
        # GBA animation makes a pre-save frame hash differ after restore even
        # when the native state is sound. Align the saved screenshot and stack
        # to the reset protocol instead of rejecting correct restores.
        base.retro_env.em.set_state(state)
        raw = None
        for _ in range(2):
            raw, done = base._one_step(base._empty_action())
            if done:
                raise RuntimeError("emulator terminated while synchronizing saved mistake state")
        # Preserve temporal context from the mistaken policy state. A reset
        # repeat would replace that history with three copies of one frame.
        obs = base._observe(raw)
        frame = base.render()
        Image.fromarray(frame).save(out / (tag + "_restore_sync.png"))
        stack_path = out / (tag + "_observation.npy")
        np.save(stack_path, obs, allow_pickle=False)
        return ({"identifier": tag, "start_id": start["start_id"], "checkpoint_local_path": str(state_path.relative_to(ROOT)),
                 "observation_local_path": str(stack_path.relative_to(ROOT)),
                 "require_field_on_reset": False, "wrong_macros": 2,
                 "policy_decisions_before_error": decisions_before_error,
                 "route_family": start.get("route_family"),
                 "state_sha256": hashlib.sha256(state).hexdigest(),
                 "visited_rgb_sha256": hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest(),
                 "visited_observation_sha256": hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest(),
                 "state_load_sync_verified": True,
                 "scene_at_visited_state": "unknown; derived from verified exterior start with directional inputs only",
                 "perturb_trace": trace}, obs.copy())
    finally:
        nav.close()


def collect_verified_demo(start: dict, route: dict, classifier, output: Path,
                          name: str, horizon: int):
    base, nav = make_env(ROOT / start["checkpoint_local_path"], max_steps=horizon * 12 + 24)
    gate = SceneConfirmationGate(2)
    hits = []
    observations, actions = [], []

    def observe(frame):
        if gate.update(classifier.classify(frame)).first_castle_entry:
            hits.append(True)

    nav.frame_observer = observe
    try:
        obs, info = nav.reset(seed=1729)
        gate.reset()
        if not info.get("checkpoint_loaded"):
            return [], [], False
        if start.get("observation_local_path"):
            obs = np.load(ROOT / start["observation_local_path"], allow_pickle=False)
            base._frames = obs.copy()
        for action_name in route["route"][:horizon]:
            action = NAVIGATION_ACTIONS.index(action_name)
            observations.append(obs.copy())
            actions.append(action)
            obs, _, terminated, truncated, _ = nav.step(action)
            if hits:
                Image.fromarray(nav.render()).save(output / (name + "_expert_goal.png"))
                break
            if terminated or truncated:
                break
        return observations, actions, bool(hits)
    finally:
        nav.close()


def evaluate(start: dict, policy, classifier, label: str, *, repeats: int, horizon: int):
    rows = []
    unknown = 0
    unknown_statuses: dict[str, int] = {}
    for repeat in range(repeats):
        base, nav = make_env(ROOT / start["checkpoint_local_path"], max_steps=horizon * 12)
        gate = SceneConfirmationGate(2)
        hit = []

        def observe(frame):
            nonlocal unknown
            result = classifier.classify(frame)
            unknown += int(result.label == "unknown")
            if result.label == "unknown":
                unknown_statuses[result.status] = unknown_statuses.get(result.status, 0) + 1
            if gate.update(result).first_castle_entry:
                hit.append(True)

        nav.frame_observer = observe
        try:
            obs, info = nav.reset(seed=1729)
            gate.reset()
            exact_frame = True
            if start.get("visited_rgb_sha256"):
                exact_frame = hashlib.sha256(
                    np.ascontiguousarray(nav.render()).tobytes()).hexdigest() == start["visited_rgb_sha256"]
            if start.get("observation_local_path"):
                obs = np.load(ROOT / start["observation_local_path"], allow_pickle=False)
                base._frames = obs.copy()
            action_counts = {name: 0 for name in NAVIGATION_ACTIONS}
            if exact_frame:
                for decision in range(1, horizon + 1):
                    action = policy.predict(obs)
                    action_counts[NAVIGATION_ACTIONS[action]] += 1
                    obs, _, terminated, truncated, _ = nav.step(action)
                    if hit or terminated or truncated:
                        break
            rows.append({"repeat": repeat + 1, "checkpoint_loaded": bool(info.get("checkpoint_loaded")),
                         "state_restored_exactly": exact_frame,
                         "goal_reached": bool(hit), "decisions": sum(action_counts.values()),
                         "action_counts": action_counts})
        finally:
            nav.close()
    return {"policy": label, "start_id": start["start_id"],
            "successes": sum(row["goal_reached"] for row in rows), "episodes": len(rows),
            "repeats": rows, "unknown_scene_frames": unknown,
            "unknown_status_counts": unknown_statuses}


def public_summary(metrics: dict) -> dict:
    """Return compact review metrics without local paths or per-frame traces."""
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in metrics["closed_loop_evaluation"]:
        key = (row["policy"], row["evaluation_split"], row["start_id"])
        grouped.setdefault(key, []).append(row)
    evaluations = []
    for (policy, split, start_id), episodes in sorted(grouped.items()):
        action_names = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT")
        evaluations.append({
            "policy": policy, "evaluation_split": split, "start_id": start_id,
            "successes": sum(item["successes"] for item in episodes),
            "episodes": sum(item["episodes"] for item in episodes),
            "decisions_per_episode": [trial["decisions"] for item in episodes for trial in item["repeats"]],
            "action_counts": {name: sum(trial["action_counts"][name]
                                         for item in episodes for trial in item["repeats"])
                              for name in action_names},
            "unknown_scene_frames": sum(item["unknown_scene_frames"] for item in episodes),
            "unknown_status_counts": {status: sum(item["unknown_status_counts"].get(status, 0)
                                                  for item in episodes)
                                      for status in ("no_match", "uncertain", "uncertain_transition")},
        })
    return {
        "run_id": metrics["run_id"], "source_run_id": metrics["source_run_id"],
        "base_demonstration_action_labels": metrics["base_demonstration_action_labels"],
        "learner_visited_mistake_states": metrics["learner_visited_mistake_states"],
        "repeatably_successful_training_correction_states": metrics["repeatably_successful_training_correction_states"],
        "expert_correction_action_labels": metrics["expert_correction_action_labels"],
        "route_search": {key: {
            "used_for_training": value["used_for_training"],
            "attempts": len(value["candidate_trials"]),
            "successful_routes": [row["route"] for row in value["candidate_trials"]
                                  if row.get("repeat_success")],
        } for key, value in metrics["route_search"].items()},
        "closed_loop_evaluation": evaluations,
        "false_positive_controls": metrics["false_positive_controls"],
        "rl_timesteps": metrics["rl_timesteps"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "limitations": metrics["limitations"],
    }


def run(source_run: str, *, horizon: int = 48, repeats: int = 2,
        search_limit: int = 10) -> Path:
    source = ROOT / "videos/gba_navigation" / source_run
    starts = {s["start_id"]: s for s in json.loads(
        (source / "start_provenance.json").read_text(encoding="utf-8"))}
    if not all(s in starts for s in FINAL_TEST_START_IDS):
        raise ValueError("source run misses previously exposed diagonal starts")
    out = ROOT / "videos/gba_recovery_imitation" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    classifier = make_classifier()
    t0 = time.perf_counter()

    base_obs, base_actions = [], []
    for start in starts.values():
        if start.get("split") != "train":
            continue
        obs, actions, report = collect_demonstration(start, out, classifier, max_decisions=horizon)
        if report["success"]:
            base_obs.extend(obs)
            base_actions.extend(actions)
    if not base_actions:
        raise RuntimeError("could not recollect original training demonstrations")
    original = NearestNeighborBehaviorCloner(neighbors=3, stride=4).fit(base_obs, base_actions)

    east_id, west_id = FINAL_TEST_START_IDS
    training_records, heldout_records = [], []
    for start_id in (east_id, west_id):
        record, _obs = capture_mistake(starts[start_id], original, out,
                                       decisions_before_error=0)
        training_records.append(record)
        record, _obs = capture_mistake(starts[start_id], original, out,
                                       decisions_before_error=6)
        heldout_records.append(record)
    # Persist the exact visited states and their actual perturbation traces
    # before route search. A failed search must still leave diagnosable evidence.
    (out / "mistake_states_local.json").write_text(
        json.dumps(training_records + heldout_records, indent=2) + "\n", encoding="utf-8")
    correction_obs, correction_actions, searches = [], [], {}
    all_records = [(record, True) for record in training_records] + [
        (record, False) for record in heldout_records]
    for record, use_for_training in all_records:
        candidate_start = dict(record)
        route_reports = []
        for index, route in enumerate(candidate_routes(record["start_id"], horizon)[:search_limit], 1):
            route["candidate_index"] = index
            first = run_trial(candidate_start, route, 1, classifier, out, horizon)
            route_reports.append({"candidate_index": index, "reset_valid": first["reset_valid"],
                                 "success": first["goal_reached"],
                                 "route": route_summary(route)})
            if not first["reset_valid"] or not first["goal_reached"]:
                continue
            second = run_trial(candidate_start, route, 2, classifier, out, horizon)
            if not second["reset_valid"] or not second["goal_reached"]:
                route_reports[-1]["repeat_success"] = False
                continue
            route_reports[-1]["repeat_success"] = True
            if use_for_training:
                obs, actions, demo_success = collect_verified_demo(
                    candidate_start, route, classifier, out, record["identifier"], horizon)
                if demo_success:
                    correction_obs.extend(obs)
                    correction_actions.extend(actions)
            break
        searches[record["identifier"]] = {"used_for_training": use_for_training,
                                          "candidate_trials": route_reports}

    if not correction_actions:
        metrics = {"run_id": out.name, "source_run_id": source_run,
                   "status": "recovery_route_not_found",
                   "approach": "expert-corrected recovery imitation feasibility; no labels admitted without repeated Castle success",
                   "base_demonstration_action_labels": len(base_actions),
                   "learner_visited_mistake_states": len(all_records),
                   "repeatably_successful_training_correction_states": 0,
                   "expert_correction_action_labels": 0, "route_search": searches,
                   "mistake_states": training_records + heldout_records,
                   "rl_timesteps": 0, "elapsed_seconds": round(time.perf_counter()-t0, 3),
                   "limitations": ["No candidate correction route reached the Castle twice from an exact saved state.",
                                   "No recovery policy was fitted or evaluated.",
                                   "Visited route families have prior exposure; no generalization claim is made."]}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
        report = [f"# GBA recovery imitation feasibility — {out.name}", "",
                  "No verified recovery route was found within the bounded search. No correction labels were admitted, and no augmented policy was trained.",
                  f"Saved exact learner-visited states: {len(all_records)}; routes attempted: {sum(len(x['candidate_trials']) for x in searches.values())}; RL timesteps: 0.",
                  "", "Exact native state files, screenshots, and action traces remain local."]
        (out / "run_report.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
        public_dir = ROOT / "docs/experiments/gba_recovery_imitation" / out.name
        public_dir.mkdir(parents=True, exist_ok=False)
        public = {k: v for k, v in metrics.items() if k != "mistake_states"}
        public["mistake_states"] = [{k: item[k] for k in ("start_id", "wrong_macros", "policy_decisions_before_error",
                                      "state_sha256", "visited_rgb_sha256", "visited_observation_sha256",
                                      "scene_at_visited_state", "perturb_trace")}
                                    for item in training_records + heldout_records]
        (public_dir / "metrics.json").write_text(json.dumps(public, indent=2)+"\n", encoding="utf-8")
        (public_dir / "REPORT.md").write_text("\n".join(report)+"\n\nNative states and raw captures are not published.\n", encoding="utf-8")
        print(out)
        print(public_dir)
        return out
    augmented = NearestNeighborBehaviorCloner(neighbors=3, stride=4).fit(
        base_obs + correction_obs, base_actions + correction_actions)
    augmented_1nn = NearestNeighborBehaviorCloner(neighbors=1, stride=4).fit(
        base_obs + correction_obs, base_actions + correction_actions)
    np.savez_compressed(out / "training_data_local.npz",
                        base_observations=np.stack(base_obs),
                        base_actions=np.asarray(base_actions, dtype=np.int64),
                        correction_observations=np.stack(correction_obs),
                        correction_actions=np.asarray(correction_actions, dtype=np.int64))
    evaluation = []
    for record, split in [(r, "correction_fit_state_diagnostic") for r in training_records] + [
            (r, "timing_heldout_state") for r in heldout_records]:
        for label, policy in (("ORIGINAL_3NN", original),
                              ("RECOVERY_AUGMENTED_1NN", augmented_1nn),
                              ("RECOVERY_AUGMENTED_3NN", augmented)):
            item = evaluate(dict(record), policy, classifier, label,
                            repeats=repeats, horizon=horizon)
            item["evaluation_split"] = split
            evaluation.append(item)
    corrected_train_count = sum(
        bool(searches.get(record["identifier"], {}).get("used_for_training"))
        and any(row.get("repeat_success") for row in searches[record["identifier"]]["candidate_trials"])
        for record in training_records)
    metrics = {"run_id": out.name, "source_run_id": source_run,
               "approach": "expert-corrected recovery imitation; two policy-opposite actions at decision 0 (training) or after six original-policy decisions (held out) on exposed east/west south-diagonal rollouts; routes reset from exact saved mistake states; corrections admitted only after two successful checkpoint replays",
               "base_demonstration_action_labels": len(base_actions),
               "learner_visited_mistake_states": len(all_records),
               "repeatably_successful_training_correction_states": corrected_train_count,
               "expert_correction_action_labels": len(correction_actions),
               "training_timing": "two wrong macros immediately at start; mirrored east/west corrections used only after repeatable route validation",
               "heldout_timing": "two wrong macros after six original-policy decisions; held out from imitation labels",
               "mistake_states": training_records + heldout_records,
               "route_search": searches, "closed_loop_evaluation": evaluation,
               "false_positive_controls": "not repeated in this run; prior Castle negative-control results remain historical",
               "rl_timesteps": 0, "elapsed_seconds": round(time.perf_counter()-t0, 3),
               "limitations": ["The exact route families have already been examined; no new generalization claim is made.",
                               "All route families have prior exposure; timing-heldout states are not fresh route-family tests.",
                               "The bounded route search produces a reproducibly successful route, not an optimal or human expert policy.",
                               "Most moving field frames remain unknown; no RAM or position reward was used."]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    public = public_summary(metrics)
    report = ["# GBA recovery-augmented imitation", "",
              f"Run `{out.name}` used 4 native-state-restored learner mistakes and 127 original plus 59 verified correction labels. The two decision-0 correction routes repeated successfully; PPO/RL timesteps: 0.", "",
              "## Closed-loop outcomes", "",
              "These route families were already exposed. Fit-state rows are in-sample diagnostics; decision-6 rows are timing-held-out only, not unseen route-family tests.", ""]
    for row in evaluation:
        per_episode = ", ".join(str(trial["decisions"]) for trial in row["repeats"])
        action_text = "; ".join(", ".join(f"{key} {value}" for key, value in trial["action_counts"].items() if value)
                                   for trial in row["repeats"])
        report.append(f"- `{row['policy']}` / `{row['evaluation_split']}` / `{row['start_id']}`: {row['successes']}/{row['episodes']}; decisions per episode {per_episode}; actions {action_text}.")
    report += ["", "The original 3-NN and recovery 1-NN both reached Castle 4/4 across the two fit states; the augmented 1-NN replayed the searched DOWN×8 → inward×8 → UP correction from east and used the original inward×8 → UP behavior from west. The augmented 3-NN collapsed to DOWN on east and scored 0/2 there. On timing-held-out states, both original and augmented models scored 4/4 by repeating UP×16; this does not show a recovery improvement.", "",
               "## Scene coverage and limits", "",
               "The moving-frame classifier remained unknown on most frames: unknown results were overwhelmingly `no_match`, with fewer `uncertain` and transition frames. Its three fixed outdoor landmark crops only recognize reviewed camera views; movement scrolls map content and changes sprite/animation pixels, so the conservative anchor detector intentionally refuses to call those frames field. This makes it a sparse transition detector, not a general field tracker.", "",
               "The online two-wrong-move benchmark was independently repeated: 0/4; zero- and one-wrong-move controls were 4/4; four wrong moves were nonmonotonic (east 2/2, west 0/2). Replay from synchronized saved states did not reproduce that failure: the original policy itself succeeded from both two-error states. The discrepancy remains unresolved, and neither correction model is promoted.", "",
               "This is bounded supervised recovery imitation, not formal DAgger, reinforcement learning, or fresh route-family generalization. No new held-out route families were used; no RAM/position reward was introduced. Castle false-positive controls were not rerun in this experiment."]
    (out / "run_report.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    public_dir = ROOT / "docs/experiments/gba_recovery_imitation" / out.name
    public_dir.mkdir(parents=True, exist_ok=False)
    (public_dir / "metrics.json").write_text(json.dumps(public, indent=2)+"\n", encoding="utf-8")
    (public_dir / "REPORT.md").write_text("\n".join(report)+"\n\nNative state files, local datasets, full traces, and emulator captures are not published.\n", encoding="utf-8")
    proof = out / "east_south_offset0_two_wrong_expert_goal.png"
    if proof.is_file():
        shutil.copy2(proof, public_dir / "recovery_route_castle_proof.png")
    print(out)
    print(public_dir)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="20261028_153708_566851Z")
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--search-limit", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 16 <= args.horizon <= 96 or not 1 <= args.repeats <= 3 or not 1 <= args.search_limit <= 40:
        parser.error("horizon 16..96; repeats 1..3; search-limit 1..40")
    if args.dry_run:
        print(json.dumps({"runtime_not_started": True, "source_run": args.source_run,
                          "mistake_states": 4, "wrong_macros_per_state": 2,
                          "training_perturb_timing": 0, "heldout_perturb_timing": 6,
                          "search_limit": args.search_limit, "horizon": args.horizon}, indent=2))
        return 0
    run(args.source_run, horizon=args.horizon, repeats=args.repeats, search_limit=args.search_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
