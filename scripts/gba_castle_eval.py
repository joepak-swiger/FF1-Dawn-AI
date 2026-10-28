"""Evaluate entry into Castle Cornelia 1F from the verified GBA checkpoint.

Evaluation only: resettable scripted, seeded-random, idle, side-route, and
START-menu trials. Rewards remain neutral and no policy is trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS  # noqa: E402
from ff1_dawn_ai.gba_landmark import CastleCorneliaDetector  # noqa: E402
from scripts.gba_eval import estimate_scene_translation  # noqa: E402
from scripts.gba_gym_smoke import discover_checkpoint  # noqa: E402

DEFAULT_REFERENCE = ROOT / "docs" / "experiments" / "gba_castle_goal" / "reference_cornelia_1f.png"


def make_run_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = parent / stamp
    output.mkdir()
    return output


def digest(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def run(checkpoint: Path, reference: Path, *, repeats: int = 3, horizon: int = 192,
        random_seeds: tuple[int, ...] = tuple(range(100, 112))) -> Path:
    if repeats < 3 or horizon < 1 or not random_seeds:
        raise ValueError("use at least 3 repeats, positive horizon, and seeded random trials")
    checkpoint, reference = checkpoint.resolve(), reference.resolve()
    if not checkpoint.is_file() or not reference.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else reference)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    detector = CastleCorneliaDetector(reference)
    out = make_run_dir(ROOT / "videos" / "gba_castle_goal")
    # The same evaluation episode may include entry plus a bounded exit route.
    env = FF1GbaEnv(checkpoint, project_root=ROOT, max_steps=(2 * horizon) + 300,
                    action_frames=1, render_mode="rgb_array", restore_count=2)
    rows: list[dict] = []
    trials: list[dict] = []
    proof_frames: list[tuple[str, np.ndarray]] = []
    reset_hashes: list[str] = []
    start_time = time.perf_counter()

    def reset(trial_id: str) -> np.ndarray:
        _obs, info = env.reset(seed=1729)
        frame = env.render()
        reset_hashes.append(digest(frame))
        result = detector.classify(frame)
        rows.append({"trial": trial_id, "step": 0, "action": "RESET", "label": result.label,
                     "status": result.status, "confidence": round(result.confidence, 6),
                     "ram_0x33FC": int(env.retro_env.get_ram()[0x33FC]),
                     "ram_0x3404": int(env.retro_env.get_ram()[0x3404]),
                     "frame_sha256": digest(frame)})
        return frame

    def step(trial_id: str, step_no: int, action: str) -> tuple[np.ndarray, dict]:
        _obs, reward, terminated, truncated, info = env.step(GBA_ACTIONS.index(action))
        if reward != 0:
            raise RuntimeError("expected reward-neutral evaluation")
        frame = env.render()
        result = detector.classify(frame)
        ram = np.asarray(env.retro_env.get_ram(), dtype=np.uint8)
        row = {"trial": trial_id, "step": step_no, "action": action, "label": result.label,
               "status": result.status, "confidence": round(result.confidence, 6),
               "ram_0x33FC": int(ram[0x33FC]), "ram_0x3404": int(ram[0x3404]),
               "frame_sha256": digest(frame), "terminated": bool(terminated), "truncated": bool(truncated)}
        rows.append(row)
        if terminated or truncated:
            raise RuntimeError(f"unexpected episode end in {trial_id} at step {step_no}")
        return frame, row

    def trial(policy: str, repeat: int, actions: list[str], *, save_proof: bool = False) -> None:
        trial_id = f"{policy.lower()}_{repeat:02d}"
        frame = reset(trial_id)
        if save_proof:
            proof_frames.append((f"{policy} reset", frame.copy()))
        first_candidate: int | None = None
        consecutive = 0
        confirmed_at: int | None = None
        prev_frame = frame
        for step_no, action in enumerate(actions, 1):
            frame, row = step(trial_id, step_no, action)
            if row["status"] == "confirmed":
                first_candidate = step_no if first_candidate is None else first_candidate
                consecutive += 1
                if consecutive >= 2 and confirmed_at is None:
                    confirmed_at = step_no
            else:
                consecutive = 0
            if save_proof and (step_no in {1, 8, 32, 64, 96, 112, 120, 124, 126, 127, 128, 129, 130, 132, 144, 160, horizon}
                               or 108 <= step_no <= 132
                               or (row["status"] == "confirmed" and consecutive <= 2)):
                name = f"{trial_id}_step_{step_no:03d}.png"
                Image.fromarray(frame).save(out / name)
                if step_no in {max(1, (first_candidate or step_no) - 1), first_candidate, confirmed_at}:
                    proof_frames.append((f"{policy} step {step_no}", frame.copy()))
            prev_frame = frame
            if confirmed_at is not None and step_no >= confirmed_at + 4:
                break
        # A target reached is independently corroborated by its static in-game
        # title on two consecutive rendered frames; hashes are only trace IDs.
        trials.append({"policy": policy, "repeat": repeat, "actions_executed": step_no,
                       "destination_first_match_step": first_candidate,
                       "destination_confirmed_step": confirmed_at,
                       "destination_reached": confirmed_at is not None,
                       "last_label": row["label"], "last_status": row["status"],
                       "ram_candidates_at_end": {"0x33FC": row["ram_0x33FC"], "0x3404": row["ram_0x3404"]}})

    start_time = time.perf_counter()
    try:
        for repeat in range(1, repeats + 1):
            trial("SCRIPTED_UP", repeat, ["UP"] * horizon, save_proof=(repeat == 1))
        for seed in random_seeds:
            rng = np.random.default_rng(seed)
            names = np.asarray(("UP", "DOWN", "LEFT", "RIGHT"))
            trial("SEEDED_RANDOM", seed, names[rng.integers(0, 4, size=horizon)].tolist(),
                  save_proof=(seed in {random_seeds[0], random_seeds[len(random_seeds) // 2], 110, random_seeds[-1]}))
        for repeat in range(1, repeats + 1):
            trial("IDLE", repeat, ["NOOP"] * horizon, save_proof=(repeat == 1))
        for repeat in range(1, repeats + 1):
            trial("SIDE_DOWN", repeat, ["DOWN"] * horizon, save_proof=(repeat == 1))

        # A separate out-and-back navigation check tests whether the named
        # castle scene can be exited with the same neutral action interface.
        exit_test: dict = {"castle_entry_confirmed": False, "castle_title_absence_sustained": False,
                           "field_return_confirmed": False}
        exit_frame = reset("castle_exit_test")
        entry_streak = 0
        for entry_step in range(1, horizon + 1):
            exit_frame, row = step("castle_exit_test_entry", entry_step, "UP")
            entry_streak = entry_streak + 1 if row["status"] == "confirmed" else 0
            if entry_streak >= 2:
                exit_test["castle_entry_confirmed"] = True
                exit_test["castle_entry_confirmation_step"] = entry_step
                Image.fromarray(exit_frame).save(out / "castle_exit_test_entry.png")
                break
        if exit_test["castle_entry_confirmed"]:
            for settle_step in range(1, 5):
                exit_frame, _ = step("castle_exit_test_settle", settle_step, "NOOP")
            exit_test["castle_start_frame_title_match"] = detector.classify(exit_frame).reached
            absent_streak = 0
            for exit_step in range(1, 129):
                exit_frame, row = step("castle_exit_test_down", exit_step, "DOWN")
                absent_streak = absent_streak + 1 if row["status"] != "confirmed" else 0
                if exit_step in (1, 8, 16, 24, 32, 48, 56, 58, 64, 72, 80, 96, 128) or (absent_streak and absent_streak <= 2):
                    Image.fromarray(exit_frame).save(out / f"castle_exit_test_down_{exit_step:03d}.png")
                if absent_streak >= 8 and "first_sustained_title_absence_step" not in exit_test:
                    exit_test["first_sustained_title_absence_step"] = exit_step - 7
                    exit_test["castle_title_absence_sustained"] = True
            exit_test["down_actions_to_title_absence"] = exit_test.get("first_sustained_title_absence_step")
            exit_test["final_detector_label"] = row["label"]
            exit_test["final_ram_candidates"] = {"0x33FC": row["ram_0x33FC"], "0x3404": row["ram_0x3404"]}
            exit_test["bounded_down_actions_after_entry"] = exit_step
            exit_test["visual_review_required"] = True

        # Distinct UI false-positive control from the same verified field state.
        frame = reset("start_menu_control")
        proof_frames.append(("Menu control reset", frame.copy()))
        # A single START edge may trigger a fade before the menu renders.
        # Observe through 64 frames so the diagnostic does not label a fade
        # as an actual menu.
        for i, action in enumerate(["START"] + ["NOOP"] * 63, 1):
            frame, row = step("start_menu_control", i, action)
            if i in (1, 2, 4, 8, 16, 24, 32, 48, 64):
                Image.fromarray(frame).save(out / f"start_menu_control_{i:02d}.png")
                if i in (8, 32, 64):
                    proof_frames.append((f"START control {i}f", frame.copy()))
        menu_rows = [r for r in rows if r["trial"] == "start_menu_control"]
        menu_positive_count = sum(r["status"] == "confirmed" for r in menu_rows)

        # Per-frame reversal timing diagnostic: 8 UP then direction reversal,
        # varying neutral-release intervals and comparing with a matched idle.
        reversal: dict[str, list[dict]] = {}
        try:
            for release in (0, 1, 2, 4, 8, 16, 32):
                reversal[str(release)] = []
                for repeat in range(1, repeats + 1):
                    name = f"reverse8_release{release}_r{repeat}"
                    start = reset(name)
                    frame_no = 0
                    frame = start
                    for _ in range(8):
                        frame_no += 1
                        frame, _ = step(name, frame_no, "UP")
                    outbound = frame.copy()
                    for _ in range(release):
                        frame_no += 1
                        frame, _ = step(name, frame_no, "NOOP")
                    for _ in range(8):
                        frame_no += 1
                        frame, _ = step(name, frame_no, "DOWN")
                    immediate = frame.copy()
                    for _ in range(16):
                        frame_no += 1
                        frame, _ = step(name, frame_no, "NOOP")
                    settled = frame.copy()
                    idle_name = f"reverse8_control_release{release}_r{repeat}"
                    idle = reset(idle_name)
                    for idle_no in range(1, 32 + release + 1):
                        idle, _ = step(idle_name, idle_no, "NOOP")
                    reversal[str(release)].append({
                        "repeat": repeat,
                        "release_noop_frames": release,
                        "outbound_scene_shift": estimate_scene_translation(start, outbound),
                        "immediate_return_scene_shift": estimate_scene_translation(start, immediate),
                        "settled_return_scene_shift": estimate_scene_translation(start, settled),
                        "matched_idle_scene_shift": estimate_scene_translation(start, idle),
                        "interpretation": "screen-space registration only; not tile position",
                    })
                    if repeat == 1 and release in (0, 4, 16, 32):
                        for label, proof in ((f"UP 8 / release {release}", outbound),
                                             (f"return / release {release}", settled),
                                             (f"idle control / release {release}", idle)):
                            proof_frames.append((label, proof.copy()))
        except Exception as exc:
            reversal["error"] = {"type": type(exc).__name__, "message": str(exc)}

        # Investigate the 64-frame DOWN anomaly with each-frame snapshots and a
        # matched no-op. This logs scene transition/text labels and RAM candidates.
        down_trace = []
        for action_name in ("NOOP", "DOWN"):
            name = f"down64_{action_name.lower()}"
            frame = reset(name)
            start_frame = frame.copy()
            for index in range(1, 65):
                frame, row = step(name, index, action_name)
                down_trace.append({"action": action_name, "frame": index,
                                   "label": row["label"], "ram_0x33FC": row["ram_0x33FC"],
                                   "ram_0x3404": row["ram_0x3404"],
                                   "scene_translation_candidate": estimate_scene_translation(start_frame, frame)})
                if index in (8, 16, 24, 32, 40, 48, 56, 64):
                    Image.fromarray(frame).save(out / f"{name}_{index:02d}.png")
                    if action_name == "DOWN":
                        proof_frames.append((f"DOWN {index}f", frame.copy()))

        # Compare candidate RAM byte pairs only against the independent visual
        # title match. This is a descriptive differential, not address meaning.
        castle_pairs = sorted({(r["ram_0x33FC"], r["ram_0x3404"]) for r in rows
                               if r["status"] == "confirmed"})
        noncastle_pair_counts: dict[str, int] = {}
        for r in rows:
            if r["status"] != "confirmed":
                key = f"{r['ram_0x33FC']}/{r['ram_0x3404']}"
                noncastle_pair_counts[key] = noncastle_pair_counts.get(key, 0) + 1
        candidate_pair_collisions = {
            f"{pair[0]}/{pair[1]}": noncastle_pair_counts.get(f"{pair[0]}/{pair[1]}", 0)
            for pair in castle_pairs
        }
        control_prefixes = ("seeded_random_", "idle_", "side_down_", "start_menu_control")
        negative_control_rows = [r for r in rows if r["trial"].startswith(control_prefixes)]
        control_false_positive_count = sum(r["status"] == "confirmed" for r in negative_control_rows)

        # package compact proof contact sheet
        cols, tile_w, tile_h = 3, 260, 190
        sheet = Image.new("RGB", (cols * tile_w, ((len(proof_frames) + cols - 1) // cols) * tile_h), "#10182b")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        for i, (label, image) in enumerate(proof_frames):
            x, y = i % cols * tile_w, i // cols * tile_h
            sheet.paste(Image.fromarray(image).resize((240, 160), Image.Resampling.NEAREST), (x + 8, y + 22))
            draw.text((x + 8, y + 4), label[:36], fill="white", font=font)
        sheet.save(out / "castle_goal_contact_sheet.png", optimize=True)

        grouped: dict[str, list[dict]] = {}
        for item in trials:
            grouped.setdefault(item["policy"], []).append(item)
        summary = {name: {"trials": len(items), "successes": sum(t["destination_reached"] for t in items),
                          "success_rate": round(sum(t["destination_reached"] for t in items) / len(items), 4),
                          "median_actions_to_confirmed_entry": float(np.median([
                              t["destination_confirmed_step"] for t in items if t["destination_reached"]
                          ])) if any(t["destination_reached"] for t in items) else None,
                          "action_counts": {direction: sum(
                              1 for r in rows if r["trial"] in {t["policy"].lower()+"_"+f"{t['repeat']:02d}" for t in items}
                              and r["action"] == direction) for direction in ("UP", "DOWN", "LEFT", "RIGHT", "NOOP")}}
                   for name, items in grouped.items()}
        checkpoint_after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metrics = {
            "run_id": out.name, "checkpoint_run_id": checkpoint.parent.name,
            "checkpoint_unchanged": checkpoint_hash == checkpoint_after,
            "checkpoint_sha256_before": checkpoint_hash, "checkpoint_sha256_after": checkpoint_after,
            "reset_observations_identical": len(set(reset_hashes)) == 1,
            "reset_count": len(reset_hashes), "action_frames_per_action": 1,
            "destination_detector": {"target": "Castle Cornelia 1F",
                                     "independent_evidence": "static in-game floor-title UI anchor from previously reviewed scene",
                                     "threshold": detector.threshold,
                                     "confirm_consecutive_frames": 2,
                                     "ram_used_for_classification": False,
                                     "nonmatch_label": "unknown"},
            "policy_summary": summary,
            "trials": trials,
            "castle_exit_test": exit_test,
            "start_menu_control": {"frames_checked": len(menu_rows),
                                    "false_positive_frames": menu_positive_count,
                                    "false_positive_rate": menu_positive_count / max(1, len(menu_rows)),
                                    "screenshots_retained_at_frames": [8, 16, 32, 64]},
            "all_control_screens": {"frames_checked": len(negative_control_rows),
                                    "castle_false_positive_frames": control_false_positive_count,
                                    "castle_false_positive_rate": control_false_positive_count / max(1, len(negative_control_rows))},
            "ram_candidate_label_differential": {
                "castle_confirmed_pairs_0x33fc_0x3404": [list(pair) for pair in castle_pairs],
                "same_pair_count_in_noncastle_or_uncertain_frames": candidate_pair_collisions,
                "interpretation": "empirical values grouped only by independent title-anchor result; not coordinate/map-ID validation"},
            "reversal_8_frame_release_diagnostic": reversal,
            "down_64_frame_diagnostic": {
                "per_frame_labels_and_ram": down_trace,
                "ram_offsets_are_provisional": True,
                "blocked_terrain_confirmed": False,
                "interpretation": "one controlled 64-frame down route versus elapsed-frame idle; scene and RAM remain non-coordinate proxies"},
            "ram_candidate_assessment": {
                "offsets": ["0x33FC", "0x3404"],
                "observed_across": "field reset, scripted castle route, random/idle/side controls and direction timing probes",
                "confirmed_scene_id_or_coordinates": False,
                "use_for_detector_or_rewards": False},
            "neutral_rewards_only": True, "training_started": False,
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            "local_trace": "action_trace.jsonl (local, ignored)"}
        (out / "action_trace.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        report = ["# Castle Cornelia navigation objective", "",
                  f"Run `{out.name}` started from the verified Seed-party checkpoint `{checkpoint.parent.name}`.",
                  "", "## Destination evidence", "",
                  "Entry is confirmed only when the game's static `Castle Cornelia 1F` title banner matches a previously reviewed UI reference on two consecutive frames. The detector does not consult RAM or pixel-change magnitude; all other frames are `unknown`.",
                  "", "## Policy results", ""]
        for name, result in summary.items():
            report.append(f"- {name}: {result['successes']}/{result['trials']} entries; median confirmed entry cost {result['median_actions_to_confirmed_entry']} one-frame actions.")
        report += ["", f"- START control: {menu_positive_count}/{len(menu_rows)} castle false positives across 65 post-input observations. Captures at 8/16 frames show a bright transition; the actual party/status menu is visible at 32/64 frames on visual review.",
                   f"- All random, idle, side-route, and START control frames: {control_false_positive_count}/{len(negative_control_rows)} castle false positives.",
                   f"- Castle exit probe: {json.dumps(exit_test, sort_keys=True)}",
                   f"- Identical native-reset observations: {metrics['reset_observations_identical']} across {len(reset_hashes)} resets.",
                   f"- Checkpoint unchanged: {metrics['checkpoint_unchanged']}.",
                   "- Candidate RAM offsets 0x33FC/0x3404 are logged across field/castle and controls but remain provisional; no coordinates or scene ID are inferred.",
                   "- The 8-frame reversal and 64-frame DOWN diagnostics save per-frame evidence; rendered-map movement is not equated with accepted tile steps or blocked terrain.",
                   "- Rewards remain zero; no policy training ran.", "",
                   "Raw screenshots, RAM/action traces, ROMs, and native states remain local. Review the contact sheet and sanitized package before publication."]
        (out / "run_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        return out
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=192)
    parser.add_argument("--random-seeds", type=int, nargs="*", default=list(range(100, 112)))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint or discover_checkpoint()
    if args.dry_run:
        if not args.reference.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(args.reference if not args.reference.is_file() else checkpoint)
        print(json.dumps({"runtime_not_started": True, "checkpoint_bytes": checkpoint.stat().st_size,
                          "reference": str(args.reference), "repeat_count": args.repeats,
                          "horizon": args.horizon, "random_seed_count": len(args.random_seeds)}, indent=2))
        return 0
    output = run(checkpoint, args.reference, repeats=args.repeats, horizon=args.horizon,
                 random_seeds=tuple(args.random_seeds))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
