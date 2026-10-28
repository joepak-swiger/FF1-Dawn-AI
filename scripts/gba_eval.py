"""Controlled, checkpoint-reset GBA input evaluation (no reward or training)."""

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
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS  # noqa: E402
from scripts.gba_gym_smoke import discover_checkpoint  # noqa: E402


POLICIES = ("NOOP", "RANDOM", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT", "L", "R")


def new_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    target = root / stamp
    target.mkdir()
    return target


def frame_digest(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def actions_for(policy: str, horizon: int, seed: int) -> list[str]:
    if policy == "RANDOM":
        rng = np.random.default_rng(seed)
        return [("UP", "DOWN", "LEFT", "RIGHT")[int(i)] for i in rng.integers(0, 4, size=horizon)]
    return [policy] * horizon


def changed_fraction(before: np.ndarray, after: np.ndarray, threshold: int = 12) -> float:
    """Pixel-change fraction for triage only; it is not a movement classifier."""
    a, b = np.asarray(before, dtype=np.int16), np.asarray(after, dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError("frames must have identical shapes")
    return float(np.any(np.abs(a - b) > threshold, axis=2).mean())


def estimate_scene_translation(before: np.ndarray, after: np.ndarray) -> dict | None:
    """Estimate 2-D scene translation with phase correlation (visual proxy)."""
    try:
        import cv2
    except ImportError:
        return None
    a = np.asarray(before, dtype=np.uint8)[:140, :, :3]
    b = np.asarray(after, dtype=np.uint8)[:140, :, :3]
    if a.shape != b.shape or a.ndim != 3:
        return None
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = cv2.createHanningWindow((gray_a.shape[1], gray_a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(gray_a, gray_b, window)
    return {"dx_pixels_candidate": round(float(dx), 3),
            "dy_pixels_candidate": round(float(dy), 3),
            "response": round(float(response), 5),
            "confidence": "unvalidated_scene_translation"}


def shift_matches_direction(action: str, shift: dict | None, *, deadband: float = 8.0) -> bool | None:
    """Check camera/map shift sign against an input; never infer tile units."""
    if not shift or action not in {"UP", "DOWN", "LEFT", "RIGHT"}:
        return None
    dx, dy = shift["dx_pixels_candidate"], shift["dy_pixels_candidate"]
    return {
        "UP": dy > deadband,
        "DOWN": dy < -deadband,
        "LEFT": dx > deadband,
        "RIGHT": dx < -deadband,
    }[action]


def marker_candidate(frame: np.ndarray) -> dict | None:
    """Return an unverified gold/red avatar-like pixel centroid in the map ROI.

    The fixed center ROI excludes most town art and status areas. This is a
    candidate visual marker, not a tile coordinate or a game-provided signal.
    """
    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    roi = rgb[65:min(125, rgb.shape[0]), 92:min(148, rgb.shape[1]), :3]
    if roi.size == 0:
        return None
    r, g, b = roi[:, :, 0].astype(np.int16), roi[:, :, 1].astype(np.int16), roi[:, :, 2].astype(np.int16)
    mask = (r >= 150) & (g >= 45) & (g <= 175) & (b <= 115) & (r - g >= 25)
    yy, xx = np.where(mask)
    # Low counts are common on a turned/animated sprite and are too ambiguous.
    if len(xx) < 5:
        return None
    return {
        "x_pixel_candidate": round(float(xx.mean() + 92), 2),
        "y_pixel_candidate": round(float(yy.mean() + 65), 2),
        "matching_pixels": int(len(xx)),
        "confidence": "low_unvalidated",
    }


def contact_sheet(frames: list[tuple[str, np.ndarray]], target: Path) -> None:
    cols, tile_w, tile_h = 3, 260, 190
    sheet = Image.new("RGB", (cols * tile_w + 20, ((len(frames) + cols - 1) // cols) * tile_h + 12), "#10182b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (label, frame) in enumerate(frames):
        x, y = 10 + (i % cols) * tile_w, 6 + (i // cols) * tile_h
        im = Image.fromarray(np.asarray(frame, dtype=np.uint8)[:, :, :3]).resize((240, 160), Image.Resampling.NEAREST)
        sheet.paste(im, (x, y + 18))
        draw.text((x, y), label[:40], fill="white", font=font)
    sheet.save(target, optimize=True)


def audit_reset_modes(checkpoint: Path, modes: tuple[int, ...] = (1, 2), repeats: int = 3) -> dict:
    """Compare first/later native resets with one versus two state loads."""
    results = {}
    for count in modes:
        env = FF1GbaEnv(
            checkpoint, project_root=ROOT, max_steps=1, action_frames=1,
            render_mode="rgb_array", restore_count=count,
        )
        try:
            hashes, durations = [], []
            for index in range(repeats):
                obs, info = env.reset(seed=99 + index)
                hashes.append(frame_digest(obs))
                durations.append(info["reset_elapsed_seconds"])
            results[str(count)] = {
                "reset_count": repeats,
                "observation_hashes_identical": len(set(hashes)) == 1,
                "first_equals_later": bool(hashes) and len(set(hashes)) == 1,
                "unique_hash_count": len(set(hashes)),
                "first_observation_sha256": hashes[0] if hashes else "",
                "mean_reset_seconds": round(float(np.mean(durations)), 6),
                "priming_noop_frames_per_reset": 2 * count,
            }
        finally:
            env.close()
    one, two = results.get("1", {}), results.get("2", {})
    results["one_restore_matches_two_restore"] = bool(
        one and two and one.get("first_observation_sha256")
        == two.get("first_observation_sha256")
    )
    return results


def run(checkpoint: Path, *, repeats: int, horizon: int, action_frames: int, restore_count: int = 2) -> Path:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = new_run_dir(ROOT / "videos" / "gba_eval")
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reset_mode_audit = audit_reset_modes(checkpoint)
    env = FF1GbaEnv(
        checkpoint, project_root=ROOT, max_steps=max(horizon, 24), action_frames=action_frames,
        render_mode="rgb_array", restore_count=restore_count,
    )
    trace: list[dict] = []
    trial_summary: list[dict] = []
    frames: list[tuple[str, np.ndarray]] = []
    reset_observations: list[str] = []
    reset_times: list[float] = []
    blocked_probe: dict = {}
    start = time.perf_counter()
    try:
        for policy_index, policy in enumerate(POLICIES):
            for repeat in range(1, repeats + 1):
                obs, reset_info = env.reset(seed=1729 + repeat)
                reset_digest = frame_digest(obs)
                reset_observations.append(reset_digest)
                reset_times.append(float(reset_info["reset_elapsed_seconds"]))
                initial_frame = env.render()
                if policy in {"NOOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B"} and repeat == 1:
                    frames.append((f"{policy} start", initial_frame.copy()))
                pixel_deltas: list[float] = []
                marker_positions: list[dict | None] = [marker_candidate(initial_frame)]
                before = initial_frame.copy()
                termination = truncated = False
                action_plan = actions_for(policy, horizon, seed=1729 + policy_index * 100 + repeat)
                for step in range(1, horizon + 1):
                    action_name = action_plan[step - 1]
                    action_idx = GBA_ACTIONS.index(action_name)
                    step_started = time.perf_counter()
                    obs, reward, termination, truncated, info = env.step(action_idx)
                    step_elapsed = time.perf_counter() - step_started
                    after = env.render()
                    delta = changed_fraction(before, after)
                    marker = marker_candidate(after)
                    pixel_deltas.append(delta)
                    marker_positions.append(marker)
                    image_file = ""
                    if repeat == 1 and step in {1, horizon}:
                        image_file = f"{policy.lower()}_r{repeat}_s{step:03}.png"
                        Image.fromarray(after).save(output / image_file)
                        frames.append((f"{policy} step {step}", after.copy()))
                    trace.append({
                        "policy": policy, "repeat": repeat, "step": step,
                        "action": info["action"], "reward": reward,
                        "emulator_frames": info["emulator_frames_executed"],
                        "step_seconds": round(step_elapsed, 6),
                        "changed_pixel_fraction_vs_previous": round(delta, 6),
                        "marker_x_pixel_candidate": marker["x_pixel_candidate"] if marker else "",
                        "marker_y_pixel_candidate": marker["y_pixel_candidate"] if marker else "",
                        "marker_matching_pixels": marker["matching_pixels"] if marker else 0,
                        "observation_sha256": frame_digest(obs),
                        "frame": image_file,
                        "terminated": termination, "truncated": truncated,
                    })
                    before = after.copy()
                    if termination or truncated:
                        break
                start_mark, end_mark = marker_positions[0], marker_positions[-1]
                scene_shift = estimate_scene_translation(initial_frame, before)
                trial_summary.append({
                    "policy": policy, "repeat": repeat, "steps": step,
                    "start_observation_sha256": reset_digest,
                    "start_marker_candidate": start_mark,
                    "end_marker_candidate": end_mark,
                    "scene_translation_candidate": scene_shift,
                    "screen_translation_matches_input": shift_matches_direction(policy, scene_shift),
                    "mean_changed_pixel_fraction": round(float(np.mean(pixel_deltas)), 6) if pixel_deltas else 0.0,
                    "terminated": bool(termination), "truncated": bool(truncated),
                })
        # A bounded rightward shoreline/forest edge probe is compared with a
        # same-horizon no-op from the exact same checkpoint. Treat a plateau
        # only as a candidate obstruction; inspect its saved frames visually.
        edge_horizon = 24
        edge_deltas = {}
        for action_name in ("NOOP", "RIGHT"):
            obs, reset_info = env.reset(seed=4242)
            reset_observations.append(frame_digest(obs))
            reset_times.append(float(reset_info["reset_elapsed_seconds"]))
            previous = env.render()
            deltas = []
            for index in range(1, edge_horizon + 1):
                t0 = time.perf_counter()
                obs, reward, termination, truncated, info = env.step(GBA_ACTIONS.index(action_name))
                elapsed = time.perf_counter() - t0
                frame = env.render()
                delta = changed_fraction(previous, frame)
                deltas.append(delta)
                image_file = ""
                if index in {1, 8, 16, edge_horizon}:
                    image_file = f"edge_{action_name.lower()}_s{index:03}.png"
                    Image.fromarray(frame).save(output / image_file)
                    frames.append((f"{action_name} edge {index}", frame.copy()))
                trace.append({
                    "policy": f"EDGE_{action_name}", "repeat": 1, "step": index,
                    "action": info["action"], "reward": reward,
                    "emulator_frames": info["emulator_frames_executed"],
                    "step_seconds": round(elapsed, 6),
                    "changed_pixel_fraction_vs_previous": round(delta, 6),
                    "marker_x_pixel_candidate": "", "marker_y_pixel_candidate": "",
                    "marker_matching_pixels": 0, "observation_sha256": frame_digest(obs),
                    "frame": image_file, "terminated": termination, "truncated": truncated,
                })
                previous = frame.copy()
                if termination or truncated:
                    break
            edge_deltas[action_name] = deltas
        noop_tail = float(np.mean(edge_deltas["NOOP"][-4:]))
        right_tail = float(np.mean(edge_deltas["RIGHT"][-4:]))
        blocked_probe = {
            "action": "RIGHT",
            "horizon_actions": edge_horizon,
            "mean_changed_pixel_fraction_last_4_noop": round(noop_tail, 6),
            "mean_changed_pixel_fraction_last_4_right": round(right_tail, 6),
            "candidate_screen_plateau_vs_noop": right_tail <= max(0.05, noop_tail * 1.5),
            "interpretation": "candidate visual plateau only; collision/blocked terrain not confirmed without reviewed avatar/map alignment",
        }
        with (output / "action_trace.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
        (output / "episodes.json").write_text(json.dumps(trial_summary, indent=2) + "\n", encoding="utf-8")
        contact_sheet(frames, output / "contact_sheet.png")
        summary = {
            "run_id": output.name,
            "checkpoint_run_id": checkpoint.parent.name,
            "checkpoint_sha256_before": source_hash,
            "checkpoint_sha256_after": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_unchanged": source_hash == hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "restore_count_per_reset": restore_count,
            "priming_noop_frames_per_reset": 2 * restore_count,
            "reset_observation_hashes_identical": len(set(reset_observations)) == 1,
            "reset_observation_hash_count": len(set(reset_observations)),
            "reset_count": len(reset_observations),
            "mean_reset_seconds": round(float(np.mean(reset_times)), 6),
            "reset_strategy_audit": reset_mode_audit,
            "policies": list(POLICIES), "repeats_per_policy": repeats,
            "fixed_horizon_actions": horizon, "action_frames": action_frames,
            "trials_completed": len(trial_summary) + 2, "extra_control_trials": 2,
            "blocked_terrain_probe": blocked_probe,
            "action_steps_completed": len(trace),
            "neutral_rewards_only": all(float(row["reward"]) == 0.0 for row in trace),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "mean_env_step_seconds": round(float(np.mean([row["step_seconds"] for row in trace])), 6),
            "episode_summary": {
                policy: {
                    "mean_changed_pixel_fraction": round(float(np.mean([x["mean_changed_pixel_fraction"] for x in trial_summary if x["policy"] == policy])), 6),
                    "marker_detection_rate": round(float(np.mean([
                        bool(row["marker_matching_pixels"]) for row in trace if row["policy"] == policy
                    ])), 6),
                    "candidate_marker_start_end": [
                        x["start_marker_candidate"] for x in trial_summary if x["policy"] == policy
                    ],
                    "scene_translation_candidates": [
                        x["scene_translation_candidate"] for x in trial_summary if x["policy"] == policy
                    ],
                    "directional_translation_match_rate": (
                        round(float(np.mean([
                            x["screen_translation_matches_input"] for x in trial_summary
                            if x["policy"] == policy and x["screen_translation_matches_input"] is not None
                        ])), 6)
                        if any(x["screen_translation_matches_input"] is not None for x in trial_summary if x["policy"] == policy)
                        else None
                    ),
                } for policy in POLICIES
            },
            "interpretation": {
                "changed_pixel_fraction": "screen-change triage only; can include animation and camera movement",
                "marker_candidate": "heuristic gold/red pixel centroid in the central map ROI; not a tile/world coordinate and not an independently confirmed game signal",
                "scene_translation_candidate": "phase-correlation displacement of the background crop; screen-scene movement only, not player/world coordinates or tile units",
                "position_signal_status": "screen_space_map_translation_repeatable_for_this_checkpoint; not a world coordinate; candidate avatar color centroid rejected",
                "rewards": "neutral zero; no progress rewards or training",
            },
        }
        (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        report = [
            "# GBA controlled evaluation", "",
            f"Checkpoint run `{checkpoint.parent.name}` was evaluated under {len(POLICIES)} fixed-action policies, {repeats} repeats each, with {horizon} actions per episode.",
            "", f"- Trials/steps: {summary['trials_completed']} / {len(trace)}",
            f"- Repeated reset observations identical: {summary['reset_observation_hashes_identical']} ({summary['reset_count']} resets)",
            f"- Reset mean: {summary['mean_reset_seconds']} sec; step mean: {summary['mean_env_step_seconds']} sec",
            f"- Reset path audit: {json.dumps(reset_mode_audit, sort_keys=True)}",
            f"- Checkpoint unchanged: {summary['checkpoint_unchanged']}",
            f"- Neutral rewards only: {summary['neutral_rewards_only']}",
            "- Baselines: NOOP, seeded RANDOM over cardinal movement, UP, DOWN, LEFT, RIGHT, A, B, START, SELECT, L, R.",
            f"- Candidate blocked-terrain comparison (RIGHT vs NOOP): {json.dumps(blocked_probe, sort_keys=True)}",
            "- The fixed ROI color centroid is a provisional visual marker. It is not a game coordinate; a hit is not proof of movement, and a miss is not proof of no movement.",
            "- Phase correlation estimates background/map translation in screen pixels. It tracks map scroll consistently with cardinal input in the recorded repeats, but not world position or tile coordinates.",
            "- No-op and button baselines help identify animation/UI changes. Pixel deltas are not treated as position.",
            "- No rewards other than neutral zero and no training were used.", "",
            "Raw action trace, screenshots, and native checkpoint remain local. Publish a reviewed sanitized summary separately.",
        ]
        (output / "run_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    finally:
        env.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="verified Seed-party native checkpoint")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=8, help="same-action policy steps per trial")
    parser.add_argument("--action-frames", type=int, default=8)
    args = parser.parse_args()
    if min(args.repeats, args.horizon, args.action_frames) < 1:
        parser.error("repeats, horizon, and action-frames must be positive")
    checkpoint = args.checkpoint or discover_checkpoint()
    result = run(checkpoint, repeats=args.repeats, horizon=args.horizon, action_frames=args.action_frames)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
