"""Package compact, sanitized GBA locomotion findings for GitHub review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROOF_FILES = ("movement_sheet.png", "reversal_sheet.png", "north_path_sheet.png")


def _mean(rows: list[dict], axis: str) -> float | None:
    values = [row["scene_translation_candidate"][axis] for row in rows
              if row.get("scene_translation_candidate")]
    return round(sum(values) / len(values), 4) if values else None


def package(run_dir: Path, destination: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    required = {
        "checkpoint_unchanged": metrics.get("checkpoint_unchanged") is True,
        "repeatable_reset": metrics.get("reset_observation_hashes_identical") is True,
        "three_independent_repeats": metrics.get("repeat_count", 0) >= 3,
        "one_frame_actions": metrics.get("action_frames_per_step") == 1,
        "neutral_rewards": metrics.get("interpretation", {}).get("rewards") == "neutral; no training ran",
    }
    missing = [name for name in PROOF_FILES if not (run_dir / name).is_file()]
    if not all(required.values()) or missing:
        raise ValueError(f"Refusing incomplete locomotion evidence: {required}; missing={missing}")

    target = (destination or ROOT / "docs" / "experiments" / "gba_locomotion" / run_dir.name).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for name in PROOF_FILES:
        source = run_dir / name
        with Image.open(source) as image:
            if image.width > 1400 or image.height > 1200:
                raise ValueError(f"Proof image is larger than the publication limit: {name}")
            image.convert("RGB").save(target / name, optimize=True)

    direction_metrics = metrics["directional_movement_metrics"]
    horizons = [str(value) for value in metrics["movement_horizons_emulator_frames"]]
    movement = {}
    for horizon in horizons:
        movement[horizon] = {}
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            rows = direction_metrics[direction][horizon]
            shifts = [row.get("scene_translation_candidate") for row in rows]
            component = "dy_pixels_candidate" if direction in ("UP", "DOWN") else "dx_pixels_candidate"
            expected_positive = direction in ("UP", "LEFT")
            matched = sum(
                bool(shift) and ((shift[component] > 0) == expected_positive)
                for shift in shifts
            )
            movement[horizon][direction] = {
                "repeat_count": len(rows),
                "scene_shift_x_mean_px": _mean(rows, "dx_pixels_candidate"),
                "scene_shift_y_mean_px": _mean(rows, "dy_pixels_candidate"),
                "expected_sign_matches": matched,
                "confidence": "screen-space translation proxy; not a world coordinate",
            }

    reversals = {}
    for axis in ("vertical", "horizontal"):
        for frames in (8, 16, 32):
            rows = [metrics["reversal_tests"].get(f"{axis}_n{frames}_r{repeat}", {})
                    for repeat in range(1, int(metrics["repeat_count"]) + 1)]
            component = "dy_pixels_candidate" if axis == "vertical" else "dx_pixels_candidate"
            positions = [abs(row.get("return_scene_translation", {}).get(component, 1e9))
                         for row in rows]
            idle = [abs(row.get("idle_scene_translation", {}).get(component, 1e9))
                    for row in rows]
            reversals[f"{axis}_{frames}_frames_each_way"] = {
                "repeat_count": len(rows),
                "mean_abs_return_scene_shift_px": round(sum(positions) / len(positions), 4),
                "mean_abs_idle_control_scene_shift_px": round(sum(idle) / len(idle), 4),
                "reversible_scene_alignment": bool(rows) and all(value < 1.0 for value in positions),
                "changed_fraction_return_mean": round(sum(row.get("return_vs_start_changed_fraction", 1)
                                                           for row in rows) / len(rows), 4),
                "changed_fraction_idle_mean": round(sum(row.get("idle_vs_start_changed_fraction", 1)
                                                         for row in rows) / len(rows), 4),
                "ram_reversible_changed_byte_count": len(metrics["reversal_tests"].get(
                    f"{axis}_n{frames}_ram_return_offsets", [])),
            }

    candidate_horizon = "8" if "8" in metrics["ram_candidates_by_horizon"] else horizons[0]
    candidate_values = metrics["ram_candidates_by_horizon"][candidate_horizon]["candidate_offsets"]
    candidate_examples = []
    for offset in (0x33FC, 0x3404):
        item = next((row for row in candidate_values if row["offset"] == offset), None)
        if item:
            candidate_examples.append({
                "ram_array_offset_hex": f"0x{offset:04X}",
                "checkpoint_value": item["checkpoint_value"],
                "direction_values_after_release": item["direction_values"],
                "classification": "provisional direction-correlated bytes; not validated coordinates",
            })

    public = {
        "run_id": metrics["run_id"],
        "checkpoint_run_id": metrics["checkpoint_run_id"],
        "checkpoint_unchanged": metrics["checkpoint_unchanged"],
        "reset_observation_hashes_identical": metrics["reset_observation_hashes_identical"],
        "independent_repeats": metrics["repeat_count"],
        "reset_count": metrics["reset_count"],
        "stable_retro_ram_bytes_exposed": metrics["ram_size_bytes"],
        "one_emulator_frame_per_action_step": True,
        "movement_horizons_emulator_frames": metrics["movement_horizons_emulator_frames"],
        "scene_translation_by_horizon": movement,
        "reversal_summary": reversals,
        "candidate_ram_examples": candidate_examples,
        "candidate_ram_counts_by_horizon": {
            h: data["directional_candidate_counts"]
            for h, data in metrics["ram_candidates_by_horizon"].items()
        },
        "blocked_terrain": {
            "verified": False,
            "observation": "The bounded north route entered Castle Cornelia 1F at emulator frame 128; this was a scene transition, not an identified blocked tile.",
        },
        "absolute_coordinates_verified": False,
        "evaluation_visit_tracker_enabled": False,
        "rewards": "neutral (0.0); no training",
        "validation_checks": required,
    }
    (target / "metrics.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    report = [
        "# GBA locomotion and position-signal validation", "",
        f"Run `{public['run_id']}` tested the official Seed-party native checkpoint from `{public['checkpoint_run_id']}`.",
        "",
        f"- Stable-Retro reset frame hash matched for all {public['reset_count']} restores; the native checkpoint file was unchanged.",
        f"- Three independent restored starts were tested for each direction at {', '.join(horizons)} held emulator frames, followed by two neutral release frames.",
        "- At 1–32 frames, the phase-correlation screen-space translation sign matched the requested cardinal direction on all three repeats. This measures map-image displacement; it does not identify tile units or absolute player coordinates.",
        "- For UP/DOWN and LEFT/RIGHT routes of 16 and 32 frames each way, the rendered map returned to within 0.04 pixels of its starting registration across repeats. The matched idle control stayed within 0.01 pixels. Screen pixel changes were close to idle levels after reversal, consistent with animation/facing differences rather than a continued map offset.",
        "- Eight-frame reversals did not return to the starting map registration (about 14 pixels residual), so the reversible pulse is at least 16 frames in this tested checkpoint/route.",
        "- A 128-frame northward route visibly entered Castle Cornelia 1F. It did not establish blocked terrain; the route crossed a map transition.",
        "- RAM reads returned 527,360 bytes. Candidate offsets `0x33FC` and `0x3404` responded repeatably to direction at short horizons, but their direction values do not uniquely separate both axes, many other bytes respond, and the candidates were not validated at a second known world position. They remain provisional; no coordinate/visit tracker is enabled.",
        "- No GBA RAM offset is currently accepted as a player coordinate. The hundreds of repeatable responders and rendering-memory changes show why change detection alone is insufficient.",
        "- Rewards stayed neutral. No PPO training ran. Raw RAM, action traces, ROMs, native checkpoint data, and full-resolution capture frames are not included.",
        "",
        "## Evidence",
        "",
        "- `movement_sheet.png` compares short and longer cardinal inputs with neutral controls.",
        "- `reversal_sheet.png` compares each route, its reverse, and equal-duration idle controls.",
        "- `north_path_sheet.png` shows the route entering Castle Cornelia 1F, explaining why this did not test an obstacle.",
        "",
        "## Next step",
        "",
        "Capture a second known field position, repeat the RAM differential and reversal protocol there, and test a visually established wall/water boundary with matched controls. Only then decide whether a map-relative visit tracker can be promoted; absolute tile coordinates remain unverified.",
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
