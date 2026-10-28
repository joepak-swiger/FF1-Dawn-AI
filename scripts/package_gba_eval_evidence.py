"""Create a compact GitHub-safe summary from a completed local GBA evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def make_compact_proof(run_dir: Path, target: Path) -> None:
    selections = (
        ("NOOP start", "noop_r1_s001.png"),
        ("UP movement", "up_r1_s008.png"),
        ("DOWN movement", "down_r1_s008.png"),
        ("LEFT movement", "left_r1_s008.png"),
        ("RIGHT movement", "right_r1_s008.png"),
        ("START menu", "start_r1_s008.png"),
        ("NOOP edge", "edge_noop_s024.png"),
        ("RIGHT edge", "edge_right_s024.png"),
    )
    missing = [name for _, name in selections if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing selected proof frames: {missing}")
    tile_w, tile_h, cols = 200, 145, 4
    sheet = Image.new("RGB", (cols * tile_w + 12, 2 * tile_h + 12), "#10182b")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for i, (label, filename) in enumerate(selections):
        x, y = 6 + (i % cols) * tile_w, 6 + (i // cols) * tile_h
        image = Image.open(run_dir / filename).convert("RGB").resize((192, 128), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y + 15))
        draw.text((x, y), label, fill="white", font=font)
    sheet.save(target, optimize=True)


def package(run_dir: Path, destination: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    baseline_trials = len(metrics.get("policies", [])) * int(metrics.get("repeats_per_policy", 0))
    extra_trials = int(metrics.get("extra_control_trials", 0))
    expected_trials = baseline_trials + extra_trials
    expected_steps = (
        baseline_trials * int(metrics.get("fixed_horizon_actions", 0))
        + extra_trials * int(metrics.get("blocked_terrain_probe", {}).get("horizon_actions", 0))
    )
    checks = {
        "checkpoint_unchanged": metrics.get("checkpoint_unchanged") is True,
        "repeatable_reset": metrics.get("reset_observation_hashes_identical") is True,
        "all_trials_completed": metrics.get("trials_completed") == expected_trials and expected_trials > 0,
        "all_steps_completed": metrics.get("action_steps_completed") == expected_steps,
        "neutral_rewards_only": metrics.get("neutral_rewards_only") is True,
    }
    if not all(checks.values()):
        raise ValueError(f"Refusing incomplete or non-repeatable evaluation: {checks}")
    target = (destination or ROOT / "docs" / "experiments" / "gba_eval" / run_dir.name).resolve()
    target.mkdir(parents=True, exist_ok=False)
    public = {
        "run_id": metrics.get("run_id"),
        "checkpoint_run_id": metrics.get("checkpoint_run_id"),
        "reset_count": metrics.get("reset_count"),
        "reset_observations_identical": metrics.get("reset_observation_hashes_identical"),
        "reset_strategy_audit": metrics.get("reset_strategy_audit"),
        "mean_reset_seconds": metrics.get("mean_reset_seconds"),
        "policies": metrics.get("policies"),
        "repeats_per_policy": metrics.get("repeats_per_policy"),
        "fixed_horizon_actions": metrics.get("fixed_horizon_actions"),
        "action_frames": metrics.get("action_frames"),
        "trials_completed": metrics.get("trials_completed"),
        "action_steps_completed": metrics.get("action_steps_completed"),
        "neutral_rewards_only": metrics.get("neutral_rewards_only"),
        "mean_env_step_seconds": metrics.get("mean_env_step_seconds"),
        "episode_summary": metrics.get("episode_summary"),
        "extra_control_trials": metrics.get("extra_control_trials"),
        "blocked_terrain_probe": metrics.get("blocked_terrain_probe"),
        "interpretation": metrics.get("interpretation"),
        "validation_checks": checks,
    }
    (target / "metrics.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    make_compact_proof(run_dir, target / "proof.png")
    shift_lines = []
    for action in ("NOOP", "UP", "DOWN", "LEFT", "RIGHT"):
        item = (public.get("episode_summary") or {}).get(action) or {}
        shifts = [x for x in item.get("scene_translation_candidates", []) if x]
        if shifts:
            dx = sum(x["dx_pixels_candidate"] for x in shifts) / len(shifts)
            dy = sum(x["dy_pixels_candidate"] for x in shifts) / len(shifts)
            match = item.get("directional_translation_match_rate")
            suffix = f"; cardinal sign match {match:.0%}" if match is not None else ""
            shift_lines.append(f"- {action}: mean rendered-scene shift ({dx:.2f}, {dy:.2f}) px{suffix}.")
    report = [
        "# Controlled GBA checkpoint evaluation", "",
        f"Run `{public['run_id']}` tested the official Seed-party native checkpoint from `{public['checkpoint_run_id']}`.",
        "", f"- Repeated checkpoint resets: {public['reset_count']}; identical observation hashes: {public['reset_observations_identical']}.",
        f"- Policies: {', '.join(public['policies'] or [])}; {public['repeats_per_policy']} repeats, {public['fixed_horizon_actions']} actions per trial.",
        f"- Completed trials/actions: {public['trials_completed']} / {public['action_steps_completed']}.",
        f"- Validation: {checks}.",
        "- Pixel-change fractions are only screen-change diagnostics. Phase correlation yields repeatable screen-space map translation in the expected opposite direction to cardinal input, which is a useful movement proxy but not player/world coordinates or tile units.",
        *shift_lines,
        "- The gold/red avatar color-centroid marker misses reset frames and is rejected as a progress signal.",
        "- Controlled idle, cardinal, button, and seeded-random trials start from native checkpoint reloads. A prolonged RIGHT-vs-NOOP map-edge probe is reported as a candidate visual plateau only, not confirmed collision evidence.",
        "- Rewards remained zero; no PPO training ran. Local native state, raw action trace, and full screenshots are excluded.",
        "", "The compact proof sheet samples idle, cardinal movement, the START menu, and the 24-action RIGHT-vs-NOOP edge probe.",
    ]
    (target / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    print(package(args.run_dir, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
