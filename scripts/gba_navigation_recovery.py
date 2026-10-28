"""Perturb-and-recover stress test for the multi-start GBA behavior cloner.

This script recollects only successful cardinal demonstrations from a prior
local benchmark, refits the fixed frame-only 3-NN policy, then injects 0/1/2/4
deliberately wrong legal moves on frozen diagonal challenge starts. Visited
learner states are not expert-labeled; this is not DAgger or RL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    NAVIGATION_ACTIONS, NearestNeighborBehaviorCloner, NavigationActionEnv,
)
from scripts.gba_navigation_imitation import (  # noqa: E402
    FINAL_TEST_START_IDS, collect_demonstration, digest, make_classifier, make_env,
)


def opposite_direction(action: int) -> int:
    """Return the opposite legal cardinal action; reject non-directional input."""
    reverse = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    name = NAVIGATION_ACTIONS[int(action)]
    if name not in reverse:
        raise ValueError("perturbation requires a directional predicted action")
    return NAVIGATION_ACTIONS.index(reverse[name])


def make_run_dir() -> Path:
    parent = ROOT / "videos/gba_navigation_recovery"
    parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    path = parent / run_id
    path.mkdir()
    return path


def evaluate_recovery(start: dict, model: NearestNeighborBehaviorCloner,
                      classifier, *, perturb_moves: int, replicate: int,
                      horizon: int, out: Path) -> dict:
    state_path = ROOT / start["checkpoint_local_path"]
    _base, nav = make_env(state_path, max_steps=(horizon + perturb_moves + 4) * 12)
    gate = SceneConfirmationGate(2)
    castle_steps: list[int] = []
    decision_index = 0
    unknown_frames = 0
    def observe(frame):
        nonlocal unknown_frames
        result = classifier.classify(frame)
        unknown_frames += int(result.label == "unknown")
        update = gate.update(result)
        if update.first_castle_entry:
            castle_steps.append(decision_index + 1)
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                out / f"goal_{start['start_id']}_p{perturb_moves}_r{replicate}.png")
    nav.frame_observer = observe
    model.reset() if hasattr(model, "reset") else None
    trace = []
    action_counts = {name: 0 for name in NAVIGATION_ACTIONS}
    obs_hashes = []
    wall_start = time.perf_counter()
    try:
        obs, reset_info = nav.reset(seed=1729)
        gate.reset()
        gate.update(classifier.classify(nav.render()))
        obs_hashes.append(digest(obs))
        success = False
        executed_frames = 0
        total_limit = horizon + perturb_moves
        for decision_index in range(total_limit):
            policy_action = model.predict(obs)
            perturb = decision_index < perturb_moves
            action = opposite_direction(policy_action) if perturb else policy_action
            before_scene = classifier.classify(nav.render()).label
            obs, _reward, terminated, truncated, info = nav.step(action)
            after_scene = classifier.classify(nav.render()).label
            action_name = NAVIGATION_ACTIONS[action]
            action_counts[action_name] += 1
            executed_frames += int(info.get("macro_emulator_frames", 0))
            obs_hashes.append(digest(obs))
            trace.append({"decision": decision_index + 1,
                          "phase": "perturb" if perturb else "policy_recovery",
                          "policy_suggested_action": NAVIGATION_ACTIONS[policy_action],
                          "executed_action": action_name,
                          "emulator_frames": int(info.get("macro_emulator_frames", 0)),
                          "scene_before": before_scene, "scene_after": after_scene,
                          "castle_entry_confirmed": bool(castle_steps)})
            if castle_steps:
                success = True
                break
            if terminated or truncated:
                break
        max_same = current_same = 1
        for first, second in zip(obs_hashes, obs_hashes[1:]):
            current_same = current_same + 1 if first == second else 1
            max_same = max(max_same, current_same)
        actions = [step["executed_action"] for step in trace]
        oscillations = sum(actions[i] == actions[i - 2] and actions[i] != actions[i - 1]
                           and actions[i] in ("UP", "DOWN", "LEFT", "RIGHT")
                           for i in range(2, len(actions)))
        compound = {"attempted": bool(success), "field_return_confirmed": False,
                    "down_actions_after_castle": 0, "consecutive_field_frames": 0}
        if success:
            # Evaluation-only continuation: leave the confirmed castle using
            # DOWN and require two independent reviewed exterior-frame matches.
            # The Castle title's disappearance alone is not treated as exit.
            field_streak = 0
            for exit_action in range(1, 49):
                _obs, _reward, terminated, truncated, exit_info = nav.step(
                    NAVIGATION_ACTIONS.index("DOWN"))
                frame = nav.render()
                label = classifier.classify(frame).label
                field_streak = field_streak + 1 if label == "cornelia_exterior_field" else 0
                compound["down_actions_after_castle"] = exit_action
                compound["consecutive_field_frames"] = field_streak
                if field_streak >= 2:
                    compound["field_return_confirmed"] = True
                    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                        out / f"field_return_{start['start_id']}_p{perturb_moves}_r{replicate}.png")
                    break
                if terminated or truncated:
                    break
        result = {"start_id": start["start_id"], "route_family": start.get("route_family"),
                  "split": "frozen_final_test", "policy": "BC_3NN_MULTI_START_PERTURB_RECOVERY",
                  "perturb_moves": perturb_moves, "replicate": replicate,
                  "goal_reached": success, "decisions": len(trace),
                  "goal_confirmation_decision": castle_steps[0] if castle_steps else None,
                  "compound_castle_exit": compound,
                  "emulator_frames": executed_frames, "action_counts": action_counts,
                  "dominant_action_fraction": round(max(action_counts.values()) / max(1, len(trace)), 4),
                  "unknown_observation_frames": unknown_frames,
                  "max_identical_macro_observation_streak": max_same,
                  "directional_oscillation_count": oscillations,
                  "checkpoint_reset_loaded": bool(reset_info.get("checkpoint_loaded")),
                  "elapsed_seconds": round(time.perf_counter() - wall_start, 3),
                  "trace": trace}
        return result
    finally:
        nav.close()


def run(source_run_id: str, *, horizon: int = 48, repeats: int = 2) -> Path:
    if Path(source_run_id).name != source_run_id:
        raise ValueError("source run ID must be a single directory name")
    source = ROOT / "videos/gba_navigation" / source_run_id
    manifest_path = source / "start_provenance.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    starts = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not starts or not all(item.get("reload_frame_reproducible") and
                              item.get("field_confirmed_after_reload") for item in starts):
        raise ValueError("source run contains unverified start states")
    start_by_id = {item["start_id"]: item for item in starts}
    train_ids = [item["start_id"] for item in starts if item.get("split") == "train"]
    challenge_ids = [item for item in FINAL_TEST_START_IDS if item in start_by_id]
    if len(challenge_ids) < 2:
        raise ValueError("source run lacks both frozen diagonal final-test starts")
    out = make_run_dir()
    classifier = make_classifier()
    observations, labels, episodes, demo_rows = [], [], [], []
    t0 = time.perf_counter()
    for start_id in train_ids:
        start = start_by_id[start_id]
        episode_obs, episode_actions, report = collect_demonstration(
            start, out, classifier, max_decisions=horizon)
        demo_rows.append(report)
        if report["success"]:
            observations.extend(episode_obs)
            labels.extend(episode_actions)
            episodes.append((episode_obs, episode_actions))
        print(f"recollected demonstration {start_id}: success={report['success']} "
              f"decisions={report['decisions']}", flush=True)
    if not episodes:
        raise RuntimeError("no successful training demonstrations could be replayed")
    model = NearestNeighborBehaviorCloner(neighbors=3, stride=4).fit(observations, labels)
    np.savez_compressed(out / "training_demonstrations.npz",
                        observations=np.stack(observations),
                        actions=np.asarray(labels, dtype=np.int64),
                        episode_lengths=np.asarray([len(a) for _, a in episodes], dtype=np.int32))
    with (out / "demonstration_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["start_id", "decision", "action", "emulator_frames", "scene_before",
                  "scene_after", "castle_entry_confirmed"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for demo in demo_rows:
            for row in demo.get("step_trace", []):
                writer.writerow({"start_id": demo["start_id"], **row})
    rows = []
    for start_id in challenge_ids:
        for perturb in (0, 1, 2, 4):
            for replicate in range(repeats):
                row = evaluate_recovery(start_by_id[start_id], model, classifier,
                                        perturb_moves=perturb, replicate=replicate,
                                        horizon=horizon, out=out)
                rows.append(row)
                print(f"recovery {start_id} perturb={perturb} replicate={replicate + 1}/{repeats}: "
                      f"success={row['goal_reached']} decisions={row['decisions']}", flush=True)
    trace_fields = ["start_id", "perturb_moves", "replicate", "decision", "phase",
                    "policy_suggested_action", "executed_action", "emulator_frames",
                    "scene_before", "scene_after", "castle_entry_confirmed"]
    with (out / "recovery_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=trace_fields)
        writer.writeheader()
        for row in rows:
            for step in row["trace"]:
                writer.writerow({"start_id": row["start_id"], "perturb_moves": row["perturb_moves"],
                                 "replicate": row["replicate"], **step})
    for row in rows:
        row.pop("trace")
    grouped = {}
    for row in rows:
        key = f"{row['perturb_moves']}_wrong_moves"
        item = grouped.setdefault(key, {"successes": 0, "episodes": 0, "per_start": {},
                                        "_dominant": [], "directional_oscillation_total": 0,
                                        "unknown_observation_frames": 0, "stalled_episode_count": 0})
        item["successes"] += int(row["goal_reached"])
        item["episodes"] += 1
        item["_dominant"].append(row["dominant_action_fraction"])
        item["directional_oscillation_total"] += row["directional_oscillation_count"]
        item["unknown_observation_frames"] += row["unknown_observation_frames"]
        item["stalled_episode_count"] += int(row["max_identical_macro_observation_streak"] >= 4)
        start = item["per_start"].setdefault(row["start_id"], {"successes": 0, "episodes": 0,
                                                               "decisions_to_goal": [], "action_counts": {n: 0 for n in NAVIGATION_ACTIONS}})
        start["episodes"] += 1
        start["successes"] += int(row["goal_reached"])
        if row["goal_reached"]:
            start["decisions_to_goal"].append(row["goal_confirmation_decision"])
        for name, count in row["action_counts"].items():
            start["action_counts"][name] += count
    for item in grouped.values():
        item["mean_dominant_action_fraction"] = round(float(np.mean(item["_dominant"])), 4)
        del item["_dominant"]
    metrics = {"run_id": out.name, "source_benchmark_run_id": source_run_id,
               "official_party_checkpoint_run_id": "20261028_001816_066107Z",
               "demonstrations_recollected": len(demo_rows),
               "successful_demonstration_count": sum(bool(d["success"]) for d in demo_rows),
               "supervised_decisions": len(labels), "training_start_ids": train_ids,
               "frozen_test_start_ids": challenge_ids,
               "methods": "3-NN supervised behavior cloning; perturbations apply the opposite of the current policy suggestion, then return control to that same policy; no expert labels on perturbed states, no DAgger, no RL",
               "perturbation_results": grouped, "episodes": rows,
               "elapsed_seconds": round(time.perf_counter() - t0, 3), "rl_timesteps": 0,
               "compound_objective": "from held-out exterior start, imitate to two-frame Castle confirmation, then press DOWN until two consecutive independently reviewed exterior anchor matches; exit not confirmed by title absence alone",
        "compound_exit_results": {"attempted": sum(bool(row["compound_castle_exit"]["attempted"]) for row in rows),
                                         "field_return_confirmed": sum(bool(row["compound_castle_exit"]["field_return_confirmed"]) for row in rows),
                                         "mean_down_actions_to_field_confirmation": round(float(np.mean([
                                             row["compound_castle_exit"]["down_actions_after_castle"]
                                             for row in rows if row["compound_castle_exit"]["attempted"]
                                         ])), 2),
                                         "episodes": len(rows)},
               "limitations": ["Perturbations are a bounded adversarial stress test on two previously tested route families, not new held-out routes.",
                               "Castle entry and field return are confirmed through separate visual anchor gates; no coordinates or comprehensive world state are inferred.",
                               "No coordinates, RAM-derived labels, progress rewards, or model weights are published."]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = [f"GBA navigation perturb-and-recover run {out.name}",
              f"Source benchmark: {source_run_id}",
              f"Successful cardinal training demonstrations: {metrics['successful_demonstration_count']}/{len(demo_rows)}; {len(labels)} action labels.",
              f"Held-out family stress set: {', '.join(challenge_ids)}; episodes: {len(rows)}; runtime: {metrics['elapsed_seconds']} sec.",
              "The perturbation reverses the model's own current directional suggestion for 0, 1, 2, or 4 decisions; the same fixed model then resumes control. This is not DAgger because no expert relabels learner-visited states.", ""]
    for key, data in grouped.items():
        report.append(f"{key}: {data['successes']}/{data['episodes']}; " + "; ".join(
            f"{sid}={v['successes']}/{v['episodes']} actions={v['action_counts']}"
            for sid, v in data["per_start"].items()))
        report.append(f"  mean dominant-action fraction={data['mean_dominant_action_fraction']}; "
                      f"oscillation events={data['directional_oscillation_total']}; "
                      f"repeated-observation stalls={data['stalled_episode_count']}; "
                      f"unknown scene frames={data['unknown_observation_frames']}")
    report += ["", f"Compound Castle-entry → field-return: {metrics['compound_exit_results']['field_return_confirmed']}/{metrics['compound_exit_results']['attempted']} Castle-successful episodes independently confirmed two consecutive exterior anchor matches after DOWN actions (mean {metrics['compound_exit_results']['mean_down_actions_to_field_confirmation']} DOWN macros; attempted across {metrics['compound_exit_results']['episodes']} total episodes).",
               "No PPO/RL updates or rewards were used. The stress test revisits the same two final-test route families and must not be counted as another untouched generalization set."]
    (out / "run_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    target = ROOT / "docs/experiments/gba_navigation_recovery" / out.name
    target.mkdir(parents=True, exist_ok=False)
    published = {k: v for k, v in metrics.items() if k != "episodes"}
    (target / "metrics.json").write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")
    (target / "REPORT.md").write_text("# GBA perturb-and-recover imitation stress test\n\n" +
        "\n".join(f"- {line}" for line in report[1:] if line) +
        "\n\nDetailed per-decision traces, observations, emulator states, ROM data, and model weights remain local and are not included.\n",
        encoding="utf-8")
    proof = sorted([*out.glob("goal_*.png"), *out.glob("field_return_*.png")])
    if proof:
        from PIL import ImageDraw
        tile = (240, 180)
        sheet = Image.new("RGB", (tile[0] * 2, tile[1] * ((len(proof[:4]) + 1) // 2)), "#10182b")
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(proof[:4]):
            with Image.open(path) as image:
                image.thumbnail((tile[0] - 8, tile[1] - 24))
                x, y = i % 2 * tile[0], i // 2 * tile[1]
                sheet.paste(image.convert("RGB"), (x + 4, y + 20))
                draw.text((x + 4, y + 4), path.stem, fill="white")
        sheet.save(out / "contact_sheet.png")
        sheet.save(target / "contact_sheet.png")
    print(out)
    print(target)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 16 <= args.horizon <= 96 or not 1 <= args.repeats <= 5:
        parser.error("--horizon must be 16..96 and --repeats must be 1..5")
    run(args.source_run_id, horizon=args.horizon, repeats=args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
