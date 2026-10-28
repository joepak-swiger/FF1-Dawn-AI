"""Create verified Cornelia starts and compare simple navigation policies.

All emulator artifacts, native states, and detailed traces are written below
ignored ``videos/gba_navigation``. ``--publish`` copies only reviewed,
aggregate metrics and a contact sheet into ``docs/experiments``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS  # noqa: E402
from ff1_dawn_ai.gba_landmark import (  # noqa: E402
    SceneAnchorClassifier, SceneConfirmationGate,
)
from ff1_dawn_ai.gba_navigation import (  # noqa: E402
    NAVIGATION_ACTIONS, NavigationActionEnv, NearestNeighborBehaviorCloner,
    TemporalNearestNeighborBehaviorCloner, LinearBehaviorCloner,
)
from scripts.gba_castle_learning_smoke import FIELD_VARIANTS as BASE_FIELD_VARIANTS, REFERENCES  # noqa: E402
from scripts.gba_eval import estimate_scene_translation  # noqa: E402
from scripts.gba_gym_smoke import discover_checkpoint  # noqa: E402

FIELD_VARIANTS = BASE_FIELD_VARIANTS + tuple(
    ROOT / "docs/experiments/gba_navigation_imitation" / f"field_anchor_{name}.png"
    for name in ("center_settled", "west", "east", "south",
                 "west_settled", "east_settled", "south_settled")
) + tuple(
    ROOT / "docs/experiments/gba_navigation_imitation" / f"field_anchor_reviewed_{name}.png"
    for name in ("west_far", "east_far", "south_far", "east_north", "west_north",
                 "east_south", "west_south")
)

# Relative input provenance, not coordinates. Splits are frozen by route
# family before collecting labels or examining final-test outcomes.
ROUTE_SPECS = {
    "center": [], "west_near": ["LEFT"] * 4, "west_far": ["LEFT"] * 8,
    "east_near": ["RIGHT"] * 4, "east_far": ["RIGHT"] * 8,
    "south_near": ["DOWN"] * 4, "south_far": ["DOWN"] * 8,
    "east_north": ["RIGHT"] * 4 + ["UP"] * 4,
    "west_north": ["LEFT"] * 4 + ["UP"] * 4,
    "east_south": ["RIGHT"] * 4 + ["DOWN"] * 4,
    "west_south": ["LEFT"] * 4 + ["DOWN"] * 4,
}
TRAIN_START_IDS = ("center", "west_near", "west_far", "east_near", "east_far",
                   "south_near", "south_far")
VALIDATION_START_IDS = ("east_north", "west_north")
FINAL_TEST_START_IDS = ("east_south", "west_south")


def make_run_dir() -> Path:
    root = ROOT / "videos/gba_navigation"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out = root / stamp
    out.mkdir()
    (out / "checkpoints").mkdir()
    return out


def digest(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _route_family(route: list[str]) -> str:
    directions = tuple(dict.fromkeys(route))
    return "center" if not directions else "+".join(d.lower() for d in directions)


def make_classifier() -> SceneAnchorClassifier:
    return SceneAnchorClassifier(REFERENCES, field_reference_variants=FIELD_VARIANTS)


def make_env(checkpoint: Path, *, classifier=None, max_steps=3000) -> tuple[FF1GbaEnv, NavigationActionEnv]:
    base = FF1GbaEnv(checkpoint, project_root=ROOT, max_steps=max_steps,
                     action_frames=1, render_mode="rgb_array", restore_count=2)
    return base, NavigationActionEnv(base, hold_frames=8, release_frames=2)


def raw_macro(base: FF1GbaEnv, name: str, hold: int = 8, release: int = 2) -> np.ndarray:
    action = GBA_ACTIONS.index(name)
    frame = None
    for control, count in ((action, hold), (GBA_ACTIONS.index("NOOP"), release)):
        for _ in range(count):
            result = base.step(control)
            frame = result[0]
            if bool(result[2]) or bool(result[3]):
                raise RuntimeError(f"emulator ended while creating a start with {name}")
    return base.render()


def _confirmed_field(base: FF1GbaEnv, classifier: SceneAnchorClassifier,
                     observations: int = 6) -> tuple[bool, list[str], np.ndarray]:
    gate = SceneConfirmationGate(2)
    labels = []
    frame = base.render()
    for i in range(observations):
        if i:
            result = base.step(GBA_ACTIONS.index("NOOP"))
            frame = base.render()
            if result[2] or result[3]:
                break
        label = classifier.classify(frame)
        labels.append(label.label)
        update = gate.update(label)
    return gate.confirmed_scene == "cornelia_exterior_field", labels, frame


def create_starts(official: Path, out: Path, classifier: SceneAnchorClassifier) -> tuple[list[dict], list[dict]]:
    """Save only distinct, field-confirmed starts; paths are provenance, not coordinates."""
    # Each candidate is derived from the official state into this run's unique
    # output directory. Existing native states are never used as write targets.
    specs = ROUTE_SPECS
    reference_hash = hashlib.sha256(official.read_bytes()).hexdigest()
    env = FF1GbaEnv(official, project_root=ROOT, max_steps=2000, action_frames=1,
                    render_mode="rgb_array", restore_count=2)
    starts: list[dict] = []
    rejects: list[dict] = []
    reference_frame = None
    try:
        for name, route in specs.items():
            if env is None:
                env = FF1GbaEnv(official, project_root=ROOT, max_steps=2000, action_frames=1,
                                render_mode="rgb_array", restore_count=2)
            env.reset(seed=1729)
            frame = env.render()
            if name == "center":
                reference_frame = frame.copy()
            for action in route:
                frame = raw_macro(env, action)
            # Prior reversal probes showed substantial screen-space inertia;
            # allow a full 32 neutral emulator frames before checkpointing.
            for _ in range(32):
                frame = raw_macro(env, "NOOP", hold=1, release=0)
            is_field, labels, frame = _confirmed_field(env, classifier, observations=12)
            registration = estimate_scene_translation(reference_frame, frame) if reference_frame is not None else None
            offset = max(abs(registration["dx_pixels_candidate"]), abs(registration["dy_pixels_candidate"])) if registration else 0.0
            Image.fromarray(frame).save(out / f"{name}_candidate.png")
            # A start must be a stable, visibly changed result of its known
            # directional route. Registration is recorded independently as a
            # camera/map diagnostic; it does not establish tile coordinates.
            previous_frames = {item["frame_sha256"] for item in starts}
            distinct = (name == "center" or
                        (digest(frame) != digest(reference_frame) and digest(frame) not in previous_frames))
            if not is_field or not distinct:
                rejects.append({"start": name, "route": route, "confirmed_field": is_field,
                                "distinct_registered_frame": distinct,
                                "screen_translation_candidate_px": round(float(offset), 3),
                                "labels": labels})
                continue
            state = env.retro_env.em.get_state()
            state_path = out / "checkpoints" / f"{name}.state"
            state_path.write_bytes(state)
            frame_path = out / f"{name}_start.png"
            Image.fromarray(frame).save(frame_path)
            start_hash = hashlib.sha256(state).hexdigest()
            # Reload this exact state in a fresh adapter twice; both the reset
            # frame and the independent field gate must reproduce.
            env.close()
            env = None
            verify, verify_nav = make_env(state_path, max_steps=500)
            try:
                reset_images = []
                for _ in range(2):
                    _obs, _info = verify_nav.reset(seed=1729)
                    reset_images.append(verify.render().copy())
                reset_match = digest(reset_images[0]) == digest(reset_images[1])
                field_ok, reload_labels, _ = _confirmed_field(verify, classifier, observations=12)
            finally:
                verify.close()
            if not (reset_match and field_ok):
                state_path.unlink(missing_ok=True)
                rejects.append({"start": name, "route": route, "confirmed_field": field_ok,
                                "reload_frame_reproducible": reset_match,
                                "labels": reload_labels})
                continue
            starts.append({"start_id": name, "route_family": _route_family(route),
                           "route_from_official_checkpoint": route,
                           "official_checkpoint_sha256": reference_hash,
                           "native_state_sha256": start_hash,
                           "checkpoint_local_path": str(state_path.relative_to(ROOT)),
                           "frame_sha256": digest(frame), "field_confirmed_after_reload": True,
                           "reload_frame_reproducible": True,
                           "screen_translation_candidate_px": round(float(offset), 3),
                           "screen_translation_candidate": registration,
                           "translation_interpretation": "rendered map registration only; no coordinate claim",
                           "classification_observations": labels})
    finally:
        if env is not None:
            env.close()
    return starts, rejects


def collect_demonstration(start: dict, out: Path, classifier: SceneAnchorClassifier,
                          *, max_decisions: int = 80) -> tuple[list[np.ndarray], list[int], dict]:
    base, nav = make_env(ROOT / start["checkpoint_local_path"], classifier=classifier,
                         max_steps=max_decisions * 12)
    observations, actions = [], []
    step_trace = []
    gate = SceneConfirmationGate(2)
    goal_frames: list[int] = []
    def observe(frame):
        update = gate.update(classifier.classify(frame))
        if update.first_castle_entry:
            # The current action has already been appended to `actions`.
            goal_frames.append(len(actions))
    nav.frame_observer = observe
    try:
        obs, _ = nav.reset(seed=1729)
        gate.reset()
        gate.update(classifier.classify(nav.render()))
        # Route setup actions back along the saved provenance path, then north.
        reverse = {"LEFT": "RIGHT", "RIGHT": "LEFT", "DOWN": "UP", "UP": "DOWN"}
        setup = [reverse[action] for action in reversed(start["route_from_official_checkpoint"])]
        planned = setup + ["UP"] * max_decisions
        for name in planned:
            action = NAVIGATION_ACTIONS.index(name)
            before_scene = classifier.classify(nav.render()).label
            observations.append(obs.copy())
            actions.append(action)
            obs, _r, terminated, truncated, info = nav.step(action)
            after_scene = classifier.classify(nav.render()).label
            step_trace.append({"decision": len(actions), "action": name,
                               "emulator_frames": int(info.get("macro_emulator_frames", 0)),
                               "scene_before": before_scene, "scene_after": after_scene,
                               "castle_entry_confirmed": bool(goal_frames)})
            if goal_frames or terminated or truncated:
                break
        success = bool(goal_frames)
        return observations, actions, {"start_id": start["start_id"],
                                       "route_family": start.get("route_family"),
                                       "success": success,
                                       "decisions": len(actions), "route_prefix": setup,
                                       "action_names": [NAVIGATION_ACTIONS[a] for a in actions],
                                       "macro_frames": len(actions) * (nav.hold_frames + nav.release_frames),
                                       "step_trace": step_trace,
                                       "goal_confirmation_decision": goal_frames[0] if success else None}
    finally:
        nav.close()


def evaluate(start: dict, policy: str, classifier: SceneAnchorClassifier, *,
             model=None, seed: int = 0, max_decisions: int = 64,
             output_dir: Path | None = None) -> dict:
    base, nav = make_env(ROOT / start["checkpoint_local_path"], classifier=classifier,
                         max_steps=max_decisions * 12)
    gate = SceneConfirmationGate(2)
    goal_frames: list[int] = []
    candidate_counts = {"castle": 0, "unknown": 0}
    decision_index = 0
    def observe(frame):
        result = classifier.classify(frame)
        if result.label == "castle_cornelia_1f" and result.status == "candidate":
            candidate_counts["castle"] += 1
        if result.label == "unknown":
            candidate_counts["unknown"] += 1
        update = gate.update(result)
        if update.first_castle_entry:
            goal_frames.append(decision_index + 1)
            if output_dir is not None:
                Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                    output_dir / f"result_{start['start_id']}_{policy}.png")
    nav.frame_observer = observe
    rng = np.random.default_rng(seed)
    action_counts = {action: 0 for action in NAVIGATION_ACTIONS}
    if hasattr(model, "reset"):
        model.reset()
    started = time.perf_counter()
    try:
        obs, reset_info = nav.reset(seed=1729)
        gate.reset()
        gate.update(classifier.classify(nav.render()))
        success = False
        reverse = {"LEFT": "RIGHT", "RIGHT": "LEFT", "DOWN": "UP", "UP": "DOWN"}
        route = ([reverse[action] for action in reversed(start["route_from_official_checkpoint"])]
                 if policy == "SCRIPTED" else [])
        decisions = 0
        executed_frames = 0
        observation_hashes = [digest(obs)]
        action_history: list[str] = []
        for decision_index in range(max_decisions):
            if policy == "IDLE":
                action = NAVIGATION_ACTIONS.index("NOOP")
            elif policy == "SEEDED_RANDOM":
                action = int(rng.integers(1, len(NAVIGATION_ACTIONS)))
            elif policy == "ALWAYS_UP":
                action = NAVIGATION_ACTIONS.index("UP")
            elif policy == "SCRIPTED":
                action = NAVIGATION_ACTIONS.index(route.pop(0)) if route else NAVIGATION_ACTIONS.index("UP")
            elif policy.startswith("BC_") or policy == "BEHAVIOR_CLONING":
                action = model.predict(obs)
            else:
                raise ValueError(policy)
            action_counts[NAVIGATION_ACTIONS[action]] += 1
            action_history.append(NAVIGATION_ACTIONS[action])
            obs, reward, terminated, truncated, info = nav.step(action)
            executed_frames += int(info.get("macro_emulator_frames", 0))
            observation_hashes.append(digest(obs))
            decisions = decision_index + 1
            if goal_frames:
                success = True
                break
            if terminated or truncated:
                break
        same_streak = max_same_streak = 1
        for left, right in zip(observation_hashes, observation_hashes[1:]):
            same_streak = same_streak + 1 if left == right else 1
            max_same_streak = max(max_same_streak, same_streak)
        oscillations = sum(action_history[i] == action_history[i - 2] and
                           action_history[i] != action_history[i - 1] and
                           action_history[i] in ("UP", "DOWN", "LEFT", "RIGHT")
                           for i in range(2, len(action_history)))
        return {"start_id": start["start_id"], "route_family": start.get("route_family"),
                "policy": policy, "seed": seed,
                "goal_reached": success, "decision_count": decisions,
                "emulator_frames": executed_frames,
                "goal_confirmation_decision": goal_frames[0] if goal_frames else None,
                "castle_single_frame_candidates": candidate_counts["castle"],
                "unknown_observations": candidate_counts["unknown"],
                "false_goal_events": 0 if success else len(goal_frames),
                "action_counts": action_counts,
                "dominant_action_fraction": round(max(action_counts.values()) / max(1, decisions), 4),
                "max_identical_macro_observation_streak": max_same_streak,
                "directional_oscillation_count": oscillations,
                "unknown_observation_fraction": round(candidate_counts["unknown"] /
                    max(1, decisions * (nav.hold_frames + nav.release_frames)), 4),
                "reset_loaded": bool(reset_info.get("checkpoint_loaded")),
                "elapsed_seconds": round(time.perf_counter() - started, 3)}
    finally:
        nav.close()


def _contact_sheet(out: Path, starts: list[dict], rows: list[dict]) -> Path:
    selected = []
    for start in starts:
        path = out / f"{start['start_id']}_start.png"
        if path.is_file():
            selected.append((f"{start['start_id']} (field candidate)", path))
    for row in rows:
        if row["goal_reached"]:
            state = next(item for item in starts if item["start_id"] == row["start_id"])
            # Save a compact result marker; per-policy final frames are written
            # by the evaluation loop only when goals are reached.
            result = out / f"result_{row['start_id']}_{row['policy']}.png"
            if result.is_file():
                selected.append((f"{row['start_id']} / {row['policy']} goal", result))
    tile_w, tile_h = 280, 205
    sheet = Image.new("RGB", (tile_w * 2, tile_h * max(1, (len(selected) + 1) // 2)), "#10182b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (label, path) in enumerate(selected):
        x, y = (i % 2) * tile_w, (i // 2) * tile_h
        with Image.open(path) as source:
            im = source.convert("RGB")
            im.thumbnail((tile_w - 12, tile_h - 32))
            sheet.paste(im, (x + 6, y + 24))
        draw.text((x + 6, y + 5), label[:42], font=font, fill="white")
    path = out / "contact_sheet.png"
    sheet.save(path)
    return path


def run(*, max_decisions: int = 64, publish: bool = False, replicates: int = 2) -> Path:
    official = discover_checkpoint()
    classifier = make_classifier()
    out = make_run_dir()
    started = time.perf_counter()
    starts, rejected = create_starts(official, out, classifier)
    # Predeclare route-family split: cardinal starts are train; north-diagonal
    # starts select among fixed models; south-diagonal starts are final test.
    train_starts = [s for s in starts if s["start_id"] in TRAIN_START_IDS]
    validation_starts = [s for s in starts if s["start_id"] in VALIDATION_START_IDS]
    final_test_starts = [s for s in starts if s["start_id"] in FINAL_TEST_START_IDS]
    for start in starts:
        start["split"] = ("train" if start["start_id"] in TRAIN_START_IDS else
                           "validation" if start["start_id"] in VALIDATION_START_IDS else
                           "final_test" if start["start_id"] in FINAL_TEST_START_IDS else "unassigned")
    (out / "start_provenance.json").write_text(json.dumps(starts, indent=2) + "\n", encoding="utf-8")
    if len(validation_starts) < 2 or len(final_test_starts) < 2:
        raise RuntimeError("insufficient independently field-confirmed starts for the frozen validation/final-test split; "
                           f"validation={len(validation_starts)}, final_test={len(final_test_starts)}; "
                           f"rejected={[item['start'] for item in rejected]}")
    train_start_ids = [s["start_id"] for s in train_starts]
    demos, demo_actions, demo_reports, demo_episodes = [], [], [], []
    for start in train_starts:
        obs, labels, report = collect_demonstration(start, out, classifier,
                                                    max_decisions=max_decisions)
        if report["success"]:
            demos.extend(obs)
            demo_actions.extend(labels)
            demo_episodes.append((obs, labels))
        demo_reports.append(report)
        (out / "demonstrations.json").write_text(json.dumps(demo_reports, indent=2) + "\n", encoding="utf-8")
    if demos:
        np.savez_compressed(out / "demonstration_data.npz",
                            observations=np.stack(demos), actions=np.asarray(demo_actions, dtype=np.int64),
                            episode_lengths=np.asarray([len(episode[1]) for episode in demo_episodes], dtype=np.int32))
    with (out / "demonstration_trace.csv").open("w", newline="", encoding="utf-8") as stream:
        trace_fields = ["start_id", "decision", "action", "emulator_frames", "scene_before",
                        "scene_after", "castle_entry_confirmed"]
        writer = csv.DictWriter(stream, fieldnames=trace_fields)
        writer.writeheader()
        for report in demo_reports:
            for trace in report.get("step_trace", []):
                writer.writerow({"start_id": report["start_id"], **trace})
    baseline_episodes = [episode for episode, report in zip(demo_episodes, demo_reports)
                         if report["start_id"] in ("center", "west_near")]
    models = {}
    if demos:
        baseline_obs = [item for episode in baseline_episodes for item in episode[0]]
        baseline_actions = [item for episode in baseline_episodes for item in episode[1]]
        if baseline_obs:
            models["BC_3NN_CENTER_WEST_ABLATION"] = NearestNeighborBehaviorCloner(
                neighbors=3, stride=4).fit(baseline_obs, baseline_actions)
        models["BC_3NN_MULTI_START"] = NearestNeighborBehaviorCloner(
            neighbors=3, stride=4).fit(demos, demo_actions)
        models["BC_TEMPORAL_3NN"] = TemporalNearestNeighborBehaviorCloner(
            neighbors=3, stride=8, history=3).fit_episodes(demo_episodes)
        models["BC_BALANCED_LINEAR"] = LinearBehaviorCloner(
            stride=8, epochs=300, seed=17).fit(demos, demo_actions)
    model_names = ["BC_3NN_CENTER_WEST_ABLATION", "BC_3NN_MULTI_START",
                   "BC_TEMPORAL_3NN", "BC_BALANCED_LINEAR"]
    policy_models = {name: models.get(name) for name in model_names}
    policies = ["IDLE", "SEEDED_RANDOM", "ALWAYS_UP", "SCRIPTED"] + [
        name for name in model_names if policy_models[name] is not None]
    rows = []
    split_by_id = {**{sid: "train" for sid in TRAIN_START_IDS},
                   **{sid: "validation" for sid in VALIDATION_START_IDS},
                   **{sid: "final_test" for sid in FINAL_TEST_START_IDS}}
    for start_index, start in enumerate(starts):
        split = split_by_id.get(start["start_id"], "unassigned")
        repeat_count = replicates if split in ("validation", "final_test") else 1
        for repeat in range(repeat_count):
            for policy_index, policy_name in enumerate(policies):
                row = evaluate(start, policy_name, classifier,
                               model=policy_models.get(policy_name),
                               seed=500 + start_index * 100 + repeat * 10 + policy_index,
                               max_decisions=max_decisions, output_dir=out)
                row["split"] = split
                row["replicate"] = repeat
                rows.append(row)
                with (out / "evaluation.partial.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                print(f"completed {len(rows)} evaluations: {split}/{start['start_id']} "
                      f"replicate={repeat + 1}/{repeat_count} policy={policy_name} "
                      f"success={row['goal_reached']}", flush=True)
    # Independent status-menu negative control: a confirmed scene change into
    # the UI must not trigger the Castle-entry event.
    menu_base, menu_nav = make_env(ROOT / starts[0]["checkpoint_local_path"], max_steps=500)
    menu_gate = SceneConfirmationGate(2)
    menu_transitions = []
    def observe_menu(frame):
        update = menu_gate.update(classifier.classify(frame))
        if update.transitioned:
            menu_transitions.append(update.confirmed_scene)
    menu_nav.frame_observer = observe_menu
    menu_nav.reset(seed=1729)
    menu_gate.reset()
    menu_gate.update(classifier.classify(menu_nav.render()))
    for _ in range(12):
        menu_nav.step(NAVIGATION_ACTIONS.index("NOOP"))
    for control in ["START"] + ["NOOP"] * 48:
        menu_base.step(GBA_ACTIONS.index(control))
        observe_menu(menu_nav.render())
    menu_false_castle_events = menu_transitions.count("castle_cornelia_1f")
    menu_control = {"confirmed_scenes": menu_transitions,
                    "false_castle_scene_transitions": menu_false_castle_events,
                    "castle_entry_events": menu_false_castle_events}
    menu_nav.close()
    totals = aggregate_evaluation(rows)
    validation_model_results = aggregate_evaluation([r for r in rows if r["split"] == "validation"])
    final_test_results = aggregate_evaluation([r for r in rows if r["split"] == "final_test"])
    learned_names = [name for name in model_names if policy_models[name] is not None]
    selected_policy = select_validation_policy(validation_model_results, learned_names, max_decisions)
    metrics = {
        "run_id": out.name, "official_checkpoint_run_id": official.parent.name,
        "official_checkpoint_sha256": hashlib.sha256(official.read_bytes()).hexdigest(),
        "verified_start_count": len(starts), "start_ids": [s["start_id"] for s in starts],
        "starts": [{k: v for k, v in s.items() if k != "checkpoint_local_path"} for s in starts],
        "rejected_start_candidates": rejected,
        "demonstration_method": "scripted route labels recorded from live observations; not reinforcement learning",
        "demonstrations": demo_reports, "demonstration_decision_samples": len(demo_actions),
        "split_manifest": {"train_start_ids": TRAIN_START_IDS,
                           "validation_start_ids": VALIDATION_START_IDS,
                           "final_test_start_ids": FINAL_TEST_START_IDS,
                           "selection_rule": "validation-only model selection; final south-diagonal route families remain untouched until final scoring"},
        "imitation_algorithms": {
            "BC_3NN_CENTER_WEST_ABLATION": "3-NN frame-only; center and west_near training demonstrations",
            "BC_3NN_MULTI_START": "3-NN frame-only; all successful cardinal training demonstrations",
            "BC_TEMPORAL_3NN": "3-NN over ordered three-observation windows, kept within each trajectory",
            "BC_BALANCED_LINEAR": "class-balanced multinomial linear model on spatially downsampled RGB"},
        "evaluation_horizon_decisions": max_decisions, "evaluation": totals,
        "validation_model_results": validation_model_results,
        "final_test_results": final_test_results,
        "selected_policy_by_validation_only": selected_policy,
        "validation_selection_status": ("selected_on_validation_successes" if selected_policy else
                                        "no_learned_policy_succeeded_on_validation"),
        "evaluation_replicates_per_validation_or_test_start": replicates,
        "evaluation_by_split": {split: aggregate_evaluation([r for r in rows if r["split"] == split])
                                for split in ("train", "validation", "final_test")},
        "evaluation_rows": rows, "negative_controls": {"status_menu": menu_control},
        "imitation_assessment": assess_imitation(rows, train_start_ids),
        "training": {"rl_timesteps": 0, "algorithm": "supervised imitation only; no RL training"},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "claim": assess_imitation(rows, train_start_ids)["claim"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_local_report(out, metrics)
    (out / "start_provenance.json").write_text(json.dumps(starts, indent=2) + "\n", encoding="utf-8")
    with (out / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["start_id", "route_family", "split", "replicate", "policy", "seed",
                  "goal_reached", "decision_count", "emulator_frames",
                  "goal_confirmation_decision", "castle_single_frame_candidates",
                  "unknown_observations", "dominant_action_fraction",
                  "max_identical_macro_observation_streak", "directional_oscillation_count"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    _contact_sheet(out, starts, rows)
    if publish:
        publish_evidence(out, metrics)
    return out


def _write_local_report(out: Path, metrics: dict) -> Path:
    assessment = metrics.get("imitation_assessment", {})
    lines = [f"GBA navigation experiment {metrics['run_id']}",
             f"Verified exterior starts: {metrics['verified_start_count']}",
             f"Training starts: {', '.join(metrics['split_manifest']['train_start_ids'])}",
             f"Validation starts: {', '.join(metrics['split_manifest']['validation_start_ids'])}",
             f"Frozen final-test starts: {', '.join(metrics['split_manifest']['final_test_start_ids'])}",
             f"Successful scripted demonstrations: {sum(bool(d['success']) for d in metrics['demonstrations'])}/{len(metrics['demonstrations'])}; {metrics['demonstration_decision_samples']} labeled decisions",
             f"Validation selection: {metrics.get('selected_policy_by_validation_only') or 'none; no learned model succeeded'}",
             f"Final-test best comparative candidate: {assessment.get('best_learned_policy') or 'none'}",
             f"Claim: {metrics['claim']}",
             f"RL timesteps: {metrics['training']['rl_timesteps']}; elapsed seconds: {metrics['elapsed_seconds']}",
             "", "Success counts by split:"]
    for split, policies in metrics.get("evaluation_by_split", {}).items():
        lines.append(f"[{split}]")
        for name, result in policies.items():
            lines.append(f"{name}: {result['successes']}/{result['episodes']}")
    lines.extend(["", "Final-test per-start results (successes/replicates; combined actions):"])
    for policy, result in metrics.get("final_test_results", {}).items():
        starts = []
        for start_id, data in result.get("per_start", {}).items():
            starts.append(f"{start_id}={data['successes']}/{data['episodes']} actions={data['action_counts']}")
        lines.append(f"{policy}: " + "; ".join(starts))
    negative = metrics.get("negative_controls", {}).get("status_menu", {})
    lines += ["", f"Menu confirmed as {negative.get('confirmed_scenes', [])}; Castle false positives: {negative.get('false_castle_scene_transitions', 0)}.",
              "All start route provenance is relative to one official save; the starts are different reproducible views on the same overworld, not independent maps or exact coordinates.",
              "The scripted policy is a provenance-based baseline. Behavior cloning is supervised imitation, not reinforcement learning."]
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text, encoding="utf-8")
    (out / "run_report.txt").write_text(text, encoding="utf-8")
    return out / "run_report.txt"


def publish_evidence(out: Path, metrics: dict, *, refresh: bool = False) -> Path:
    target = ROOT / "docs/experiments/gba_navigation_imitation" / out.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not refresh:
            raise FileExistsError(target)
        existing = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
        if existing.get("run_id") != metrics.get("run_id"):
            raise ValueError("refusing to refresh evidence for a different experiment run")
    else:
        target.mkdir()
    rows = metrics.get("evaluation_rows", [])
    safe_starts = [{"start_id": s["start_id"], "route_family": s.get("route_family"),
                    "route_from_official_checkpoint": s.get("route_from_official_checkpoint", []),
                    "split": s.get("split"), "field_confirmed_after_reload": s.get("field_confirmed_after_reload"),
                    "reload_frame_reproducible": s.get("reload_frame_reproducible"),
                    "screen_translation_candidate_px": s.get("screen_translation_candidate_px"),
                    "translation_interpretation": s.get("translation_interpretation")}
                   for s in metrics.get("starts", [])]
    def compact_evaluation(results: dict, *, include_starts: bool = False) -> dict:
        compact = {}
        for name, item in results.items():
            compact[name] = {"successes": item["successes"], "episodes": item["episodes"],
                             "success_rate": item.get("success_rate"),
                             "mean_success_decisions": item.get("mean_success_decisions"),
                             "unknown_observation_fraction": item.get("unknown_observation_fraction"),
                             "mean_dominant_action_fraction": item.get("mean_dominant_action_fraction"),
                             "directional_oscillation_total": item.get("directional_oscillation_total"),
                             "worst_identical_observation_streak": item.get("worst_identical_observation_streak"),
                             "false_goal_events": item.get("false_goal_events")}
            if include_starts:
                compact[name]["per_start"] = {start_id: {
                    "successes": start["successes"], "episodes": start["episodes"],
                    "success_rate": start["success_rate"],
                    "decision_counts": start["decision_counts"],
                    "false_goal_events": start["false_goal_events"],
                    "action_counts": start["action_counts"],
                    "unknown_observation_fraction": start["unknown_observation_fraction"],
                    "mean_dominant_action_fraction": start["mean_dominant_action_fraction"],
                    "directional_oscillation_count": start["directional_oscillation_count"],
                    "max_identical_macro_observation_streak": start["max_identical_macro_observation_streak"],
                } for start_id, start in item.get("per_start", {}).items()}
        return compact

    compact_split = {split: compact_evaluation(result)
                     for split, result in metrics.get("evaluation_by_split", {}).items()}
    safe = {
        "run_id": metrics["run_id"], "official_checkpoint_run_id": metrics["official_checkpoint_run_id"],
        "party": [{"name": "Sif", "class": "Warrior"}, {"name": "Myth", "class": "Thief"},
                  {"name": "Milo", "class": "Monk"}, {"name": "Seed", "class": "Red Mage"}],
        "verified_start_count": metrics["verified_start_count"], "starts": safe_starts,
        "rejected_start_candidates": [{"start_id": item["start"], "reason": "not field-confirmed/reproducible"}
                                       for item in metrics.get("rejected_start_candidates", [])],
        "demonstration_method": metrics["demonstration_method"],
        "demonstration_decision_samples": metrics["demonstration_decision_samples"],
        "demonstrations": [{k: d.get(k) for k in ("start_id", "route_family", "success", "decisions",
                          "goal_confirmation_decision", "route_prefix", "action_names", "macro_frames")}
                           for d in metrics.get("demonstrations", [])],
        "split_manifest": metrics["split_manifest"],
        "imitation_algorithms": metrics["imitation_algorithms"],
        "action_macro": {"hold_frames": 8, "release_frames": 2, "evaluation_horizon_decisions": metrics["evaluation_horizon_decisions"]},
        "evaluation_by_split": compact_split,
        "validation_model_results": compact_evaluation(metrics.get("validation_model_results", {})),
        "validation_selection": {"policy": metrics.get("selected_policy_by_validation_only"),
                                 "status": metrics.get("validation_selection_status")},
        "final_test_results": compact_evaluation(metrics.get("final_test_results", {}), include_starts=True),
        "imitation_assessment": metrics.get("imitation_assessment", {}),
        "negative_controls": metrics.get("negative_controls", {}),
        "resource": {"elapsed_seconds": metrics["elapsed_seconds"],
                     "episodes": len(rows), "rl_timesteps": 0,
                     "replicates_per_validation_or_test_start": metrics.get("evaluation_replicates_per_validation_or_test_start")},
        "claim": metrics["claim"],
        "limitations": ["All starts derive from one official save and one outdoor map; they are reproducible view states, not independent maps or exact coordinates.",
                        "Validation routes failed for every tested policy. The two frozen final route families each have two repeats; this is promising feasibility evidence, not broad statistical generalization.",
                        "Policies use screen observations only; no coordinates, RAM progress signals, rewards, or RL timesteps were used.",
                        "The best final-test comparator is reported after evaluating all predeclared variants; it was not selected using validation success."]
    }
    (target / "metrics.json").write_text(json.dumps(safe, indent=2) + "\n", encoding="utf-8")
    report_lines = ["# Multi-start GBA navigation imitation benchmark", "",
                    f"Run `{metrics['run_id']}` from the unchanged local Seed-party checkpoint. The verified party is Sif/Warrior, Myth/Thief, Milo/Monk, Seed/Red Mage.", "",
                    f"All {metrics['verified_start_count']} starts were confirmed as Cornelia exterior by two-frame scene gate and reproduced after native reload. They were generated from one source checkpoint; different start views are not separate maps or coordinate measurements.", "",
                    f"The train split has {len(metrics['split_manifest']['train_start_ids'])} cardinal route starts; validation has two north-diagonal route families; final test has two south-diagonal route families. Each validation and test start was reset twice. The final test was untouched for training and validation selection.", "",
                    f"Successful demonstrator rollouts: {sum(bool(d.get('success')) for d in metrics['demonstrations'])}/{len(metrics['demonstrations'])}, covering {metrics['demonstration_decision_samples']} labeled decisions. Training is supervised behavior cloning, not RL; RL timesteps = 0.", "",
                    "## Results by split", "", "| Policy | Train | Validation | Frozen final test |", "|---|---:|---:|---:|"]
    split_results = metrics.get("evaluation_by_split", {})
    for name in metrics.get("evaluation", {}):
        vals = []
        for split in ("train", "validation", "final_test"):
            result = split_results.get(split, {}).get(name, {"successes": 0, "episodes": 0})
            vals.append(f"{result['successes']}/{result['episodes']}")
        report_lines.append(f"| {name} | {vals[0]} | {vals[1]} | {vals[2]} |")
    report_lines += ["", "Final-test detail (two independent resets per start):", "",
                     "| Policy | East-south | West-south | Turn actions across all repeats |", "|---|---:|---:|---|"]
    for policy, result in metrics.get("final_test_results", {}).items():
        per_start = result.get("per_start", {})
        es = per_start.get("east_south", {"successes": 0, "episodes": 0})
        ws = per_start.get("west_south", {"successes": 0, "episodes": 0})
        action_details = []
        for sid in ("east_south", "west_south"):
            p = per_start.get(sid, {})
            counts = p.get("action_counts", {})
            turn = counts.get("LEFT", 0) if sid == "east_south" else counts.get("RIGHT", 0)
            action_details.append(str(turn))
        report_lines.append(f"| {policy} | {es['successes']}/{es['episodes']} | {ws['successes']}/{ws['episodes']} | left={action_details[0]}, right={action_details[1]} |")
    report_lines += ["", "Final-test scene/loop diagnostics (episode means or totals):", ""]
    for policy, result in metrics.get("final_test_results", {}).items():
        report_lines.append(f"- {policy}: unknown-frame fraction {result.get('unknown_observation_fraction', 0.0):.3f}; "
                            f"mean dominant-action fraction {result.get('mean_dominant_action_fraction', 0.0):.3f}; "
                            f"oscillations {result.get('directional_oscillation_total', 0)}; "
                            f"worst identical-observation streak {result.get('worst_identical_observation_streak', 0)}; "
                            f"false Castle events {result.get('false_goal_events', 0)}.")
    assessment = metrics.get("imitation_assessment", {})
    report_lines += ["", f"Validation selection: **{metrics.get('selected_policy_by_validation_only') or 'none'}** ({metrics.get('validation_selection_status')}). No policy reached Castle on either validation route.",
                     f"Held-out challenge: always-UP failed both starts (0/4). The predeclared multi-start frame 3-NN, three-observation temporal 3-NN, and balanced linear imitator each reached Castle on both starts in both resets (4/4). Successful model action totals were 8 LEFT + 34 UP for east-south and 8 RIGHT + 34 UP for west-south across two repeats. This meets the bounded turn-generalization criterion for these two route families, but the sample is small.", "",
                     f"The center+west-only 3-NN ablation succeeded 2/4 and repeated all-RIGHT on east-south; the expanded model's added east examples address this prior action-collapse pattern. Idle, seeded-random, always-UP, and provenance-scripted baselines each scored 0/4 on the final set. The scripted baseline failed on diagonal starts despite reaching 7/7 training starts.", "",
                     f"Castle detector negative control: menu transition produced {metrics['negative_controls']['status_menu']['false_castle_scene_transitions']} false Castle entries. No RAM coordinates, game rewards, or PPO training were used.", "",
                     "## Interpretation and next step", "",
                     f"Claim: `{metrics['claim']}`. The BC policies use only rendered observations and actions; they did not receive start IDs, route provenance, or oracle actions at inference. The final candidate comparison is reported separately from validation selection; validation had no successful policy, so no model was chosen from it.",
                     "Next: stress-test the expanded 3-NN with deliberate one-to-four-move perturbations and add a recovery set; then evaluate the selected visual policy on fresh held-out route families and the compound Castle-exit objective. Do not start PPO until these tests remain repeatable.", "",
                     "Published package excludes ROMs, save states, raw traces, native checkpoints, and model weights."]
    (target / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    image = out / "contact_sheet.png"
    if image.is_file():
        shutil.copyfile(image, target / image.name)
    return target


def refresh_published_run(run_id: str) -> Path:
    if Path(run_id).name != run_id:
        raise ValueError("run ID must be a single directory name")
    out = ROOT / "videos/gba_navigation" / run_id
    metrics_path = out / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for demo in metrics.get("demonstrations", []):
        if demo.get("success"):
            demo["goal_confirmation_decision"] = demo.get("decisions")
    rows = metrics.get("evaluation_rows", [])
    metrics["evaluation"] = aggregate_evaluation(rows)
    metrics["evaluation_by_split"] = {split: aggregate_evaluation(
        [row for row in rows if row.get("split") == split])
        for split in ("train", "validation", "final_test")}
    metrics["validation_model_results"] = metrics["evaluation_by_split"]["validation"]
    candidates = [name for name in ("BC_3NN_CENTER_WEST_ABLATION", "BC_3NN_MULTI_START",
                                    "BC_TEMPORAL_3NN", "BC_BALANCED_LINEAR")
                  if name in metrics["validation_model_results"]]
    metrics["selected_policy_by_validation_only"] = select_validation_policy(
        metrics["validation_model_results"], candidates,
        metrics.get("evaluation_horizon_decisions", 48))
    metrics["validation_selection_status"] = ("selected_on_validation_successes"
        if metrics["selected_policy_by_validation_only"] else "no_learned_policy_succeeded_on_validation")
    metrics["final_test_results"] = metrics["evaluation_by_split"]["final_test"]
    metrics["imitation_assessment"] = assess_imitation(
        rows,
        [item["start_id"] for item in metrics.get("demonstrations", [])],
    )
    metrics["claim"] = metrics["imitation_assessment"]["claim"]
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_local_report(out, metrics)
    return publish_evidence(out, metrics, refresh=True)


def aggregate_evaluation(rows: list[dict]) -> dict:
    totals: dict[str, dict] = {}
    for row in rows:
        item = totals.setdefault(row["policy"], {"successes": 0, "episodes": 0, "per_start": {},
                                                   "_success_decisions": [], "_unknown": [],
                                                   "_dominant": [], "directional_oscillation_total": 0,
                                                   "worst_identical_observation_streak": 0,
                                                   "false_goal_events": 0})
        item["successes"] += int(row["goal_reached"])
        item["episodes"] += 1
        if row["goal_reached"]:
            item["_success_decisions"].append(row["decision_count"])
        item["_unknown"].append(row.get("unknown_observation_fraction", 0.0))
        item["_dominant"].append(row.get("dominant_action_fraction", 0.0))
        item["directional_oscillation_total"] += row.get("directional_oscillation_count", 0)
        item["worst_identical_observation_streak"] = max(
            item["worst_identical_observation_streak"],
            row.get("max_identical_macro_observation_streak", 0))
        item["false_goal_events"] += row.get("false_goal_events", 0)
        per_start = item["per_start"].setdefault(row["start_id"], {
            "successes": 0, "episodes": 0, "success_rate": 0.0,
            "decision_counts": [], "false_goal_events": 0,
            "_unknown": [], "_dominant": [], "directional_oscillation_count": 0,
            "max_identical_macro_observation_streak": 0,
            "action_counts": {name: 0 for name in NAVIGATION_ACTIONS},
        })
        per_start["episodes"] += 1
        per_start["successes"] += int(row["goal_reached"])
        per_start["success_rate"] = per_start["successes"] / per_start["episodes"]
        per_start["decision_counts"].append(row["decision_count"])
        per_start["false_goal_events"] += row["false_goal_events"]
        per_start["_unknown"].append(row.get("unknown_observation_fraction", 0.0))
        per_start["_dominant"].append(row.get("dominant_action_fraction", 0.0))
        per_start["directional_oscillation_count"] += row.get("directional_oscillation_count", 0)
        per_start["max_identical_macro_observation_streak"] = max(
            per_start["max_identical_macro_observation_streak"],
            row.get("max_identical_macro_observation_streak", 0))
        for name, count in row["action_counts"].items():
            per_start["action_counts"][name] += count
        per_start.update({
            "success": row["goal_reached"], "decisions": row["decision_count"],
            "goal_confirmation_decision": row["goal_confirmation_decision"],
            "last_replicate_action_counts": row["action_counts"],
        })
    for item in totals.values():
        item["unknown_observation_fraction"] = round(float(np.mean(item.pop("_unknown"))), 4)
        item["mean_dominant_action_fraction"] = round(float(np.mean(item.pop("_dominant"))), 4)
        for start in item["per_start"].values():
            start["unknown_observation_fraction"] = round(float(np.mean(start.pop("_unknown"))), 4)
            start["mean_dominant_action_fraction"] = round(float(np.mean(start.pop("_dominant"))), 4)
        item["success_rate"] = item["successes"] / max(1, item["episodes"])
        item["mean_success_decisions"] = (sum(item["_success_decisions"]) /
                                           len(item["_success_decisions"]) if item["_success_decisions"] else None)
        del item["_success_decisions"]
    return totals


def select_validation_policy(validation_results: dict, candidate_names: list[str],
                             horizon: int = 48) -> str | None:
    """Select only on validation success; return none if all candidates fail."""
    if not candidate_names:
        return None
    def rank(name):
        item = validation_results.get(name, {})
        mean = item.get("mean_success_decisions")
        return (item.get("successes", 0),
                -(mean if mean is not None else horizon + 1),
                -candidate_names.index(name))
    selected = max(candidate_names, key=rank)
    return selected if validation_results.get(selected, {}).get("successes", 0) > 0 else None


def assess_imitation(rows: list[dict], train_start_ids: list[str]) -> dict:
    learned_names = ("BEHAVIOR_CLONING", "BC_3NN_CENTER_WEST_ABLATION", "BC_3NN_MULTI_START",
                     "BC_TEMPORAL_3NN", "BC_BALANCED_LINEAR")
    primary_training = [row for row in rows if row["policy"] in
                        ("BEHAVIOR_CLONING", "BC_3NN_MULTI_START") and
                        row["start_id"] in train_start_ids]
    has_final = any(row.get("split") == "final_test" for row in rows)
    scope = [row for row in rows if row.get("split") == "final_test"] if has_final else [
        row for row in rows if row["start_id"] not in train_start_ids]
    challenge_ids = sorted({row["start_id"] for row in scope if row["policy"] == "ALWAYS_UP"}
        - {row["start_id"] for row in scope if row["policy"] == "ALWAYS_UP" and row["goal_reached"]})
    candidate_names = [name for name in learned_names if any(row["policy"] == name for row in scope)]
    per_policy = {}
    for name in candidate_names:
        selected = [row for row in scope if row["policy"] == name]
        successful_turn_ids = sorted({row["start_id"] for row in selected
            if row["start_id"] in challenge_ids and row["goal_reached"] and
            any(row["action_counts"].get(action, 0) for action in ("LEFT", "RIGHT", "DOWN"))})
        per_policy[name] = {"episodes": len(selected),
                            "successes": sum(bool(row["goal_reached"]) for row in selected),
                            "turn_challenge_success_ids": successful_turn_ids}
    best_policy = max(candidate_names, key=lambda name: (
        len(per_policy[name]["turn_challenge_success_ids"]),
        per_policy[name]["successes"], -candidate_names.index(name))) if candidate_names else None
    best_success_ids = per_policy[best_policy]["turn_challenge_success_ids"] if best_policy else []
    training_turn_success = any(row["goal_reached"] and
        any(row["action_counts"].get(name, 0) for name in ("LEFT", "RIGHT", "DOWN"))
        for row in primary_training)
    if len(best_success_ids) >= 2:
        claim = "learned_policy_succeeded_on_multiple_held_out_turning_start_families"
    elif best_success_ids:
        claim = "learned_policy_succeeded_on_one_held_out_turning_start_family_only"
    elif training_turn_success and challenge_ids:
        claim = "imitation_reproduced_demonstrated_turn_routes_only;_held_out_turn_start_failed"
    elif training_turn_success:
        claim = "imitation_reproduced_demonstrated_turn_routes;_no_held_out_turn_challenge"
    elif candidate_names and any(row["goal_reached"] for row in scope if row["policy"] in candidate_names):
        claim = "imitation_succeeded_held_out_but_did_not_show_turn_generalization"
    else:
        claim = "no_successful_imitation_evidence"
    heldout_up = [row for row in scope if row["policy"] == "ALWAYS_UP"]
    best_rows = [row for row in scope if best_policy and row["policy"] == best_policy]
    return {
        "demonstrated_start_ids": list(train_start_ids),
        "held_out_start_ids": sorted({row["start_id"] for row in scope}),
        "final_test_scope": has_final,
        "training_start_count": len({row["start_id"] for row in primary_training}),
        "training_start_successes": len({row["start_id"] for row in primary_training if row["goal_reached"]}),
        "training_start_success_with_turns": training_turn_success,
        "held_out_count": len({row["start_id"] for row in scope}),
        "held_out_successes": len({row["start_id"] for row in best_rows if row["goal_reached"]}),
        "held_out_always_up_successes": sum(bool(row["goal_reached"]) for row in heldout_up),
        "held_out_turn_challenge_ids": challenge_ids,
        "best_learned_policy": best_policy,
        "held_out_turn_imitation_successes": len(best_success_ids),
        "held_out_turn_success_ids": best_success_ids,
        "held_out_policy_comparison": per_policy,
        "held_out_beats_always_up": bool(best_success_ids),
        "claim": claim,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate checkpoint, anchors, imports; do not boot emulator")
    parser.add_argument("--max-decisions", type=int, default=64)
    parser.add_argument("--replicates", type=int, default=2,
                        help="fresh evaluation resets for validation/final-test starts")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--refresh-published", help="refresh sanitized report for a completed local run ID")
    args = parser.parse_args()
    if args.max_decisions < 16 or args.max_decisions > 96:
        parser.error("--max-decisions must be between 16 and 96")
    if args.replicates < 1 or args.replicates > 5:
        parser.error("--replicates must be between 1 and 5")
    checkpoint = discover_checkpoint()
    classifier = make_classifier()
    if args.refresh_published:
        target = refresh_published_run(args.refresh_published)
        print(target)
        return 0
    if args.dry_run:
        print(json.dumps({"runtime_not_started": True, "checkpoint_bytes": checkpoint.stat().st_size,
                          "classifier_anchors": list(REFERENCES), "navigation_actions": NAVIGATION_ACTIONS,
                          "max_decisions": args.max_decisions}))
        return 0
    path = run(max_decisions=args.max_decisions, publish=args.publish, replicates=args.replicates)
    print(path)
    if args.publish:
        print(ROOT / "docs/experiments/gba_navigation_imitation" / path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
