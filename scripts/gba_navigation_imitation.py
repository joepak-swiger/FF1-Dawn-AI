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
import re
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
)
from scripts.gba_castle_learning_smoke import FIELD_VARIANTS as BASE_FIELD_VARIANTS, REFERENCES  # noqa: E402
from scripts.gba_eval import estimate_scene_translation  # noqa: E402
from scripts.gba_gym_smoke import discover_checkpoint  # noqa: E402

FIELD_VARIANTS = BASE_FIELD_VARIANTS + tuple(
    ROOT / "docs/experiments/gba_navigation_imitation" / f"field_anchor_{name}.png"
    for name in ("center_settled", "west", "east", "south",
                 "west_settled", "east_settled", "south_settled")
)


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
    # Four bounded paths from the original official checkpoint. Each path is
    # used to generate a different restartable state; no existing state is
    # written or modified.
    specs = {
        "center": [],
        "west_offset": ["LEFT"] * 4,
        "east_offset": ["RIGHT"] * 4,
        "south_offset": ["DOWN"] * 4,
    }
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
            distinct = name == "center" or digest(frame) != digest(reference_frame)
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
            starts.append({"start_id": name, "route_from_official_checkpoint": route,
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
            observations.append(obs.copy())
            actions.append(action)
            obs, _r, terminated, truncated, _info = nav.step(action)
            if goal_frames or terminated or truncated:
                break
        success = bool(goal_frames)
        return observations, actions, {"start_id": start["start_id"], "success": success,
                                       "decisions": len(actions), "route_prefix": setup,
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
    started = time.perf_counter()
    try:
        obs, reset_info = nav.reset(seed=1729)
        gate.reset()
        gate.update(classifier.classify(nav.render()))
        success = False
        route = [
            {"center": [], "west_offset": ["RIGHT"] * 4,
             "east_offset": ["LEFT"] * 4, "south_offset": ["UP"] * 4}.get(start["start_id"], [])
            if policy == "SCRIPTED" else []
        ][0]
        decisions = 0
        for decision_index in range(max_decisions):
            if policy == "IDLE":
                action = NAVIGATION_ACTIONS.index("NOOP")
            elif policy == "SEEDED_RANDOM":
                action = int(rng.integers(1, len(NAVIGATION_ACTIONS)))
            elif policy == "ALWAYS_UP":
                action = NAVIGATION_ACTIONS.index("UP")
            elif policy == "SCRIPTED":
                action = NAVIGATION_ACTIONS.index(route.pop(0)) if route else NAVIGATION_ACTIONS.index("UP")
            elif policy == "BEHAVIOR_CLONING":
                action = model.predict(obs)
            else:
                raise ValueError(policy)
            action_counts[NAVIGATION_ACTIONS[action]] += 1
            obs, reward, terminated, truncated, info = nav.step(action)
            decisions = decision_index + 1
            if goal_frames:
                success = True
                break
            if terminated or truncated:
                break
        return {"start_id": start["start_id"], "policy": policy, "seed": seed,
                "goal_reached": success, "decision_count": decisions,
                "emulator_frames": sum(action_counts.values()) * (nav.hold_frames + nav.release_frames),
                "goal_confirmation_decision": goal_frames[0] if goal_frames else None,
                "castle_single_frame_candidates": candidate_counts["castle"],
                "unknown_observations": candidate_counts["unknown"],
                "false_goal_events": 0 if success else len(goal_frames),
                "action_counts": action_counts,
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


def run(*, max_decisions: int = 64, publish: bool = False) -> Path:
    official = discover_checkpoint()
    classifier = make_classifier()
    out = make_run_dir()
    started = time.perf_counter()
    starts, rejected = create_starts(official, out, classifier)
    # Hold out the east offset when present: this is a start where the measured
    # always-UP policy fails. Train only on center + west and evaluate transfer
    # to the opposite side, plus any additional nondemonstrated starts.
    preferred_demo_ids = ("center", "west_offset")
    train_starts = [s for s in starts if s["start_id"] in preferred_demo_ids]
    if len(train_starts) < 2 and len(starts) > 1:
        train_starts = [s for s in starts if s["start_id"] != "east_offset"][:2]
    if len(starts) == 1:
        train_starts = []
    train_start_ids = [s["start_id"] for s in train_starts]
    heldout_start_ids = [s["start_id"] for s in starts if s["start_id"] not in train_start_ids]
    demos, demo_actions, demo_reports = [], [], []
    for start in train_starts:
        obs, labels, report = collect_demonstration(start, out, classifier,
                                                    max_decisions=max_decisions)
        demos.extend(obs)
        demo_actions.extend(labels)
        demo_reports.append(report)
    policy = NearestNeighborBehaviorCloner(neighbors=3, stride=4).fit(demos, demo_actions) if demos else None
    rows = []
    for start_index, start in enumerate(starts):
        policies = ["IDLE", "SEEDED_RANDOM", "ALWAYS_UP", "SCRIPTED"]
        if policy is not None:
            policies.append("BEHAVIOR_CLONING")
        for policy_index, policy_name in enumerate(policies):
            row = evaluate(start, policy_name, classifier, model=policy,
                           seed=500 + start_index * 10 + policy_index,
                           max_decisions=max_decisions, output_dir=out)
            rows.append(row)
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
    metrics = {
        "run_id": out.name, "official_checkpoint_run_id": official.parent.name,
        "official_checkpoint_sha256": hashlib.sha256(official.read_bytes()).hexdigest(),
        "verified_start_count": len(starts), "start_ids": [s["start_id"] for s in starts],
        "starts": [{k: v for k, v in s.items() if k != "checkpoint_local_path"} for s in starts],
        "rejected_start_candidates": rejected,
        "demonstration_method": "scripted route labels recorded from live observations; not reinforcement learning",
        "demonstrations": demo_reports, "demonstration_decision_samples": len(demo_actions),
        "imitation_algorithm": "3-nearest-neighbor behavior cloning over downsampled rendered observations",
        "evaluation_horizon_decisions": max_decisions, "evaluation": totals,
        "evaluation_rows": rows, "negative_controls": {"status_menu": menu_control},
        "imitation_assessment": assess_imitation(rows, train_start_ids),
        "training": {"rl_timesteps": 0, "algorithm": "supervised imitation only; no RL training"},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "claim": assess_imitation(rows, train_start_ids)["claim"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (out / "report.txt").write_text(
        f"Run {out.name}\nVerified starts: {len(starts)} ({', '.join(s['start_id'] for s in starts)})\n"
        f"Demonstration decisions: {len(demo_actions)} from {len(demo_reports)} scripted successful episodes\n"
        + "\n".join(f"{name}: {data['successes']}/{data['episodes']} by start {data['per_start']}"
                     for name, data in totals.items())
        + f"\nClaim: {metrics['claim']}\nRL timesteps: 0\nElapsed: {metrics['elapsed_seconds']} sec\n",
        encoding="utf-8")
    (out / "start_provenance.json").write_text(json.dumps(starts, indent=2) + "\n", encoding="utf-8")
    with (out / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["start_id", "policy", "seed", "goal_reached", "decision_count",
                  "goal_confirmation_decision", "castle_single_frame_candidates", "unknown_observations"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    _contact_sheet(out, starts, rows)
    if publish:
        publish_evidence(out, metrics)
    return out


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
    safe = {key: value for key, value in metrics.items()
            if key not in {"evaluation_rows", "official_checkpoint_sha256"}}
    safe["starts"] = [{k: v for k, v in s.items() if k not in {"native_state_sha256"}}
                       for s in safe.get("starts", [])]
    (target / "metrics.json").write_text(json.dumps(safe, indent=2) + "\n", encoding="utf-8")
    report = ["# GBA Cornelia navigation imitation experiment", "",
              f"Run `{metrics['run_id']}` used the local official Seed-party checkpoint. The published package includes no native states, ROM data, raw traces, or model weights.", "",
              f"Verified distinct starts: {metrics['verified_start_count']} ({', '.join(metrics['start_ids'])}). Starting screens passed the two-observation exterior scene gate, reproduced after native reload, and were produced by recorded directional routes. Screen-space registration is a visual difference check, not a coordinate claim.", "",
              f"Demonstration: {metrics['demonstration_decision_samples']} labeled decisions from scripted routes. The nearest-neighbor policy is supervised behavior cloning; RL timesteps: 0.", "",
              f"Total experiment wall time: {metrics['elapsed_seconds']:.1f} seconds.",
              f"Training start IDs: {', '.join(metrics.get('imitation_assessment', {}).get('demonstrated_start_ids', [])) or 'none'}. Held-out start IDs: {', '.join(metrics.get('imitation_assessment', {}).get('held_out_start_ids', [])) or 'none'}.",
              f"Behavior cloning on demonstrated starts: {metrics.get('imitation_assessment', {}).get('training_start_successes', 0)}/{metrics.get('imitation_assessment', {}).get('training_start_count', 0)}. Held-out BC: {metrics.get('imitation_assessment', {}).get('held_out_successes', 0)}/{metrics.get('imitation_assessment', {}).get('held_out_count', 0)}; always-UP on held-out starts: {metrics.get('imitation_assessment', {}).get('held_out_always_up_successes', 0)}/{metrics.get('imitation_assessment', {}).get('held_out_count', 0)}.", "",
              f"Held-out turn-challenge starts (always-UP failures): {metrics.get('imitation_assessment', {}).get('held_out_turn_challenge_ids', [])}; BC successes on those starts: {metrics.get('imitation_assessment', {}).get('held_out_turn_imitation_successes', 0)}.",
              "## Outcomes", ""]
    for name, data in metrics["evaluation"].items():
        report.append(f"- {name}: {data['successes']}/{data['episodes']}; per-start results: `{json.dumps(data['per_start'], sort_keys=True)}`.")
    menu = metrics.get("negative_controls", {}).get("status_menu", {})
    report += ["", f"Claim: `{metrics['claim']}`.",
               f"Menu negative control: confirmed scenes `{menu.get('confirmed_scenes', [])}`; Castle-entry false positives `{menu.get('false_castle_scene_transitions', 'not recorded')}`.",
               "The scripted baseline is a known route and is not learning. Behavior-cloning results are only generalization evidence when a start was excluded from demonstration collection. Held-out south-start success also matched always-UP; it does not establish turn generalization. No RL reward or RAM progress signal was used.", ""]
    (target / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
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
    metrics["evaluation"] = aggregate_evaluation(metrics.get("evaluation_rows", []))
    metrics["imitation_assessment"] = assess_imitation(
        metrics.get("evaluation_rows", []),
        [item["start_id"] for item in metrics.get("demonstrations", [])],
    )
    metrics["claim"] = metrics["imitation_assessment"]["claim"]
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report_path = out / "report.txt"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        report = re.sub(r"^Claim: .*?$", f"Claim: {metrics['claim']}", report, flags=re.MULTILINE)
        report += ("\nImitation assessment: demonstrated-start success "
                   f"{metrics['imitation_assessment']['training_start_successes']}/"
                   f"{metrics['imitation_assessment']['training_start_count']}; held-out success "
                   f"{metrics['imitation_assessment']['held_out_successes']}/"
                   f"{metrics['imitation_assessment']['held_out_count']}; held-out always-UP success "
                   f"{metrics['imitation_assessment']['held_out_always_up_successes']}/"
                   f"{metrics['imitation_assessment']['held_out_count']}.\n")
        report_path.write_text(report, encoding="utf-8")
    return publish_evidence(out, metrics, refresh=True)


def aggregate_evaluation(rows: list[dict]) -> dict:
    totals: dict[str, dict] = {}
    for row in rows:
        item = totals.setdefault(row["policy"], {"successes": 0, "episodes": 0, "per_start": {}})
        item["successes"] += int(row["goal_reached"])
        item["episodes"] += 1
        item["per_start"][row["start_id"]] = {
            "success": row["goal_reached"], "decisions": row["decision_count"],
            "goal_confirmation_decision": row["goal_confirmation_decision"],
            "false_goal_events": row["false_goal_events"],
            "action_counts": row["action_counts"],
        }
    return totals


def assess_imitation(rows: list[dict], train_start_ids: list[str]) -> dict:
    training = [row for row in rows if row["policy"] == "BEHAVIOR_CLONING"
                and row["start_id"] in train_start_ids]
    heldout = [row for row in rows if row["policy"] == "BEHAVIOR_CLONING"
               and row["start_id"] not in train_start_ids]
    heldout_up = [row for row in rows if row["policy"] == "ALWAYS_UP"
                  and row["start_id"] not in train_start_ids]
    turned_and_succeeded = any(row["goal_reached"] and
        any(count for name, count in row["action_counts"].items() if name != "UP")
        for row in training)
    up_by_start = {row["start_id"]: row["goal_reached"] for row in rows
                   if row["policy"] == "ALWAYS_UP"}
    heldout_turn_ids = sorted({row["start_id"] for row in heldout
                               if up_by_start.get(row["start_id"]) is False})
    heldout_turn_bc = [row for row in heldout if row["start_id"] in heldout_turn_ids
                       and row["goal_reached"] and
                       any(n for action, n in row["action_counts"].items() if action != "UP")]
    if heldout_turn_bc:
        claim = "imitation_generalized_turn_navigation_to_held_out_start"
    elif turned_and_succeeded and heldout_turn_ids:
        claim = "imitation_reproduced_demonstrated_turn_routes_only;_held_out_turn_start_failed"
    elif turned_and_succeeded:
        claim = "imitation_reproduced_demonstrated_turn_routes;_no_held_out_turn_challenge"
    elif heldout and any(row["goal_reached"] for row in heldout):
        claim = "imitation_succeeded_held_out_but_did_not_show_turn_generalization"
    else:
        claim = "no_successful_imitation_evidence"
    return {
        "demonstrated_start_ids": list(train_start_ids),
        "held_out_start_ids": sorted({row["start_id"] for row in heldout}),
        "training_start_count": len(training),
        "training_start_successes": sum(bool(row["goal_reached"]) for row in training),
        "training_start_success_with_turns": turned_and_succeeded,
        "held_out_count": len(heldout),
        "held_out_successes": sum(bool(row["goal_reached"]) for row in heldout),
        "held_out_always_up_successes": sum(bool(row["goal_reached"]) for row in heldout_up),
        "held_out_turn_challenge_ids": heldout_turn_ids,
        "held_out_turn_imitation_successes": len(heldout_turn_bc),
        "held_out_beats_always_up": bool(heldout_turn_bc),
        "claim": claim,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate checkpoint, anchors, imports; do not boot emulator")
    parser.add_argument("--max-decisions", type=int, default=64)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--refresh-published", help="refresh sanitized report for a completed local run ID")
    args = parser.parse_args()
    if args.max_decisions < 16 or args.max_decisions > 96:
        parser.error("--max-decisions must be between 16 and 96")
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
    path = run(max_decisions=args.max_decisions, publish=args.publish)
    print(path)
    if args.publish:
        print(ROOT / "docs/experiments/gba_navigation_imitation" / path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
