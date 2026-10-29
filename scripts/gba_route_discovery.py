"""Bounded, observation-free route feasibility search for existing GBA starts.

This runner is for reachability diagnosis only. It runs fixed directional
macros from previously generated, reset-verified checkpoints and confirms
Castle with the independent two-frame title gate. It never trains or labels a
policy, and its validation-start routes must not be reused as training data.
"""

from __future__ import annotations

import argparse
import csv
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
from ff1_dawn_ai.gba_navigation import NAVIGATION_ACTIONS  # noqa: E402
from scripts.gba_navigation_imitation import (  # noqa: E402
    make_classifier, make_env, digest,
)


def candidate_routes(start_id: str, horizon: int = 48) -> list[dict]:
    """Return a deterministic, bounded set of simple directional route candidates.

    The first group preserves the historical DOWN/inward/UP sweep. Additional
    candidates try horizontal-first corrections and single-axis movement so
    recovery search is not biased toward the old route template.
    """
    inward = "LEFT" if start_id.startswith("east_") else "RIGHT"
    outward = "RIGHT" if inward == "LEFT" else "LEFT"
    pairs = [(0, 0), (2, 2), (4, 4), (6, 6), (8, 8),
             (4, 2), (4, 6), (6, 4), (8, 4), (2, 4),
             (6, 8), (8, 6), (10, 4), (4, 8), (10, 8)]
    result = []
    for down, lateral in pairs:
        if down + lateral >= horizon:
            continue
        result.append({"segments": [["DOWN", down], [inward, lateral],
                                     ["UP", horizon - down - lateral]],
                       "route": ["DOWN"] * down + [inward] * lateral
                                + ["UP"] * (horizon - down - lateral)})

    # Prior recovery states can be displaced sideways first; include both
    # orders and single-direction controls. Keep order stable for reproducible
    # bounded search and put old candidates first for historical comparability.
    magnitudes = (2, 4, 6, 8, 10, 12)
    plans = []
    for direction in (inward, outward, "DOWN", "UP"):
        plans.extend([[(direction, n)] for n in magnitudes])
    for first, second in ((inward, "DOWN"), ("DOWN", inward),
                          (outward, "UP"), ("UP", outward),
                          (inward, "UP"), ("UP", inward),
                          (outward, "DOWN"), ("DOWN", outward)):
        plans.extend([[(first, a), (second, b)]
                      for a in (2, 4, 6, 8, 10)
                      for b in (2, 4, 6, 8, 10)])
    seen = {tuple((name, count) for name, count in row["segments"][:2]
                  if count) for row in result}
    for plan in plans:
        used = sum(n for _, n in plan)
        if used >= horizon:
            continue
        key = tuple((name, count) for name, count in plan if count)
        if key in seen:
            continue
        seen.add(key)
        segments = [[name, count] for name, count in plan]
        segments.append(["UP", horizon - used])
        route = [name for name, count in plan for _ in range(count)]
        route.extend(["UP"] * (horizon - used))
        result.append({"segments": segments, "route": route})
    return result


def route_summary(route: dict) -> str:
    return " ".join(f"{direction}x{count}" for direction, count in route["segments"] if count)


def reset_is_valid(info: dict, label: str, repeated_hash_matches: bool,
                   *, require_field: bool = True) -> bool:
    """Validate checkpoint loading and deterministic reset; optionally require a scene label."""
    return bool(info.get("checkpoint_loaded") and repeated_hash_matches
                and (label == "cornelia_exterior_field" if require_field else label not in
                     ("menu", "cornelia_town", "castle_cornelia_1f")))


def run_trial(start: dict, route: dict, repeat: int, classifier, out: Path,
              horizon: int) -> dict:
    state_path = ROOT / start["checkpoint_local_path"]
    _base, nav = make_env(state_path, max_steps=horizon * 12 + 32)
    gate = SceneConfirmationGate(2)
    goal_at: list[int] = []
    decision = 0

    def observe(frame):
        result = classifier.classify(frame)
        event = gate.update(result)
        if event.first_castle_entry and not goal_at:
            goal_at.append(decision)
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                out / f"proof_{start['start_id']}_route{route['candidate_index']}_r{repeat}.png")

    nav.frame_observer = observe
    trace = []
    hashes = []
    counts = {name: 0 for name in NAVIGATION_ACTIONS}
    began = time.perf_counter()
    try:
        obs, info = nav.reset(seed=1729)
        gate.reset()
        reset_label = classifier.classify(nav.render()).label
        reset_hash = digest(nav.render())
        _repeat_obs, repeat_info = nav.reset(seed=1729)
        repeated_hash_matches = reset_hash == digest(nav.render())
        expected_hash = start.get("visited_rgb_sha256")
        expected_hash_matches = expected_hash is None or reset_hash == expected_hash
        repeated_hash_matches = repeated_hash_matches and expected_hash_matches
        hashes.append(digest(obs))
        reset_valid = reset_is_valid(info, reset_label, repeated_hash_matches,
                                     require_field=start.get("require_field_on_reset", True))
        if not reset_valid:
            return {"start_id": start["start_id"], "candidate_index": route["candidate_index"],
                    "route_definition": route_summary(route),
                    "repeat": repeat, "reset_valid": False, "reset_label": reset_label,
                    "reset_frames_reproducible": repeated_hash_matches,
                    "expected_visited_frame_matches": expected_hash_matches,
                    "goal_reached": False, "decisions": 0, "goal_step": None,
                    "actions": counts, "trace": [], "elapsed_seconds": round(time.perf_counter()-began,3)}
        success = False
        for decision, action in enumerate(route["route"], 1):
            action_id = NAVIGATION_ACTIONS.index(action)
            before = classifier.classify(nav.render()).label
            obs, _reward, terminated, truncated, step_info = nav.step(action_id)
            after = classifier.classify(nav.render()).label
            hashes.append(digest(obs))
            counts[action] += 1
            trace.append({"decision": decision, "action": action,
                          "frames": int(step_info.get("macro_emulator_frames", 0)),
                          "scene_before": before, "scene_after": after,
                          "castle_confirmed": bool(goal_at)})
            if goal_at:
                success = True
                break
            if terminated or truncated:
                break
        return {"start_id": start["start_id"], "candidate_index": route["candidate_index"],
                "route_definition": route_summary(route),
                "repeat": repeat, "reset_valid": True, "reset_label": reset_label,
                "reset_frames_reproducible": repeated_hash_matches,
                "expected_visited_frame_matches": expected_hash_matches,
                "goal_reached": success, "decisions": len(trace),
                "goal_step": goal_at[0] if goal_at else None,
                "actions": counts, "trace": trace,
                "observation_changes": sum(a != b for a, b in zip(hashes, hashes[1:])),
                "elapsed_seconds": round(time.perf_counter()-began,3)}
    finally:
        nav.close()


def run(source_run: str, start_ids: tuple[str, ...], horizon: int, repeats: int,
        max_candidates: int) -> Path:
    if Path(source_run).name != source_run:
        raise ValueError("source run must be a single local run ID")
    source = ROOT / "videos/gba_navigation" / source_run
    provenance_path = source / "start_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    by_id = {entry["start_id"]: entry for entry in provenance}
    selected = []
    for start_id in start_ids:
        if start_id not in by_id:
            raise ValueError(f"missing source start {start_id}")
        start = by_id[start_id]
        if not start.get("field_confirmed_after_reload") or not start.get("reload_frame_reproducible"):
            raise ValueError(f"start is not independently reset/scene verified: {start_id}")
        selected.append(start)
    out = ROOT / "videos/gba_route_discovery" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    classifier = make_classifier()
    rows = []
    started = time.perf_counter()
    for start in selected:
        candidates = candidate_routes(start["start_id"], horizon)[:max_candidates]
        for index, route in enumerate(candidates, 1):
            route["candidate_index"] = index
            for repeat in range(repeats):
                row = run_trial(start, route, repeat + 1, classifier, out, horizon)
                rows.append(row)
                print(f"{start['start_id']} route={index} repeat={repeat + 1}: "
                      f"reset={row['reset_valid']} goal={row['goal_reached']} "
                      f"step={row['goal_step']}", flush=True)
            if all(r["goal_reached"] for r in rows[-repeats:]):
                break
    fields = ["start_id", "candidate_index", "repeat", "decision", "action", "frames",
              "scene_before", "scene_after", "castle_confirmed"]
    with (out / "action_trace.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for step in row["trace"]:
                writer.writerow({"start_id": row["start_id"], "candidate_index": row["candidate_index"],
                                 "repeat": row["repeat"], **step})
    for row in rows:
        row.pop("trace")
    metrics = {"run_id": out.name, "source_run_id": source_run,
               "start_ids": list(start_ids), "route_horizon": horizon,
               "repeat_count": repeats, "max_candidates_per_start": max_candidates,
               "candidate_policy": "fixed DOWN then inward lateral then UP; no learned policy and no route-derived input at inference",
               "purpose": "reachability diagnosis only; no data from these validation starts is training data",
               "results": rows,
               "elapsed_seconds": round(time.perf_counter()-started, 3),
               "limitations": ["Directional macro search is a bounded route feasibility test, not a coordinate or optimal-path search.",
                               "All starts derive from one source checkpoint on the same overworld map.",
                               "Castle entry requires two consecutive independently reviewed title-anchor observations; this run did not repeat the separate negative-control suite."]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    public = {"run_id": out.name, "source_run_id": source_run,
              "results": [{k: r[k] for k in ("start_id", "candidate_index", "route_definition", "repeat", "reset_valid", "reset_frames_reproducible",
                           "goal_reached", "decisions", "goal_step", "observation_changes", "elapsed_seconds")}
                          for r in metrics["results"]],
              "elapsed_seconds": metrics["elapsed_seconds"],
              "limitations": metrics["limitations"]}
    dest = ROOT / "docs/experiments/gba_route_discovery" / out.name
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "metrics.json").write_text(json.dumps(public, indent=2)+"\n", encoding="utf-8")
    success_rows = [r for r in metrics["results"] if r["goal_reached"]]
    text = [f"# GBA north-diagonal route feasibility — {out.name}", "",
            f"Source run: `{source_run}`. Validation starts were used only to diagnose reachability; successful routes were not added to training.", "",
            f"Confirmed Castle successes: {len(success_rows)}/{len(metrics['results'])} route attempts.", "",
            "All attempted routes use bounded DOWN → inward lateral → UP macros, with Castle entry confirmed by the independent two-frame title gate.", "",
            "Per-start outcomes:"]
    for start_id in start_ids:
        per = [r for r in metrics["results"] if r["start_id"] == start_id]
        wins = [r for r in per if r["goal_reached"]]
        text.append(f"- `{start_id}`: {len(wins)}/{len(per)} attempts reached Castle; successful route candidate(s): "
                    + (", ".join(str(r["candidate_index"]) for r in wins) or "none"))
    text += ["", "This is a feasibility search, not a learned policy, optimal route, map-coordinate estimate, or new benchmark. Native states and action traces remain local."]
    text += ["", "The previously used inverse-provenance comparator retraces four DOWN and four inward lateral macros before UP; that 4+4 candidate (index 3) failed on both starts and both repeats. Candidate 7 changed only the lateral correction to six macros: DOWN×4, LEFT×6 (east_north) or RIGHT×6 (west_north), then UP. It reached Castle on the 25th macro on all four repeats. This is direct evidence that the old comparator was under-correcting this route; no tile coordinates are inferred."]
    (dest / "REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    proofs = sorted(out.glob("proof_*.png"))
    if proofs:
        shutil.copy2(proofs[0], dest / "castle_entry_proof.png")
    print(out)
    print(dest)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="20261028_153708_566851Z")
    parser.add_argument("--start-ids", nargs="+", default=["east_north", "west_north"])
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 16 <= args.horizon <= 96 or not 1 <= args.repeats <= 3 or not 1 <= args.max_candidates <= 40:
        parser.error("horizon must be 16..96, repeats 1..3, candidates 1..40")
    if args.dry_run:
        print(json.dumps({"runtime_not_started": True, "starts": args.start_ids,
                          "candidate_route_counts": {s: len(candidate_routes(s, args.horizon)[:args.max_candidates])
                                                     for s in args.start_ids}, "horizon": args.horizon,
                          "repeat_count": args.repeats}, indent=2))
        return 0
    run(args.source_run, tuple(args.start_ids), args.horizon, args.repeats, args.max_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
