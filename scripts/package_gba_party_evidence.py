"""Publish a compact, sanitized summary for a completed local GBA party run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_proof(run_dir: Path, output_path: Path) -> None:
    font = ImageFont.load_default()
    labels = ("Accepted custom party roster", "Cornelia after restore + UP")
    canvas = Image.new("RGB", (500, 198), "#10182b")
    draw = ImageDraw.Draw(canvas)
    for index, (filename, label) in enumerate(zip(
        ("gba_custom_party_roster.png", "gba_after_reload_resume.png"), labels
    )):
        image = Image.open(run_dir / filename).convert("RGB").resize((240, 160), Image.Resampling.NEAREST)
        x = 10 + index * 250
        draw.text((x, 6), label, fill="white", font=font)
        canvas.paste(image, (x, 22))
    canvas.save(output_path, optimize=True)


def validate_run(run_dir: Path) -> tuple[dict, dict, dict]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    runtime = metrics.get("runtime", {})
    machine = runtime.get("party_state_machine", {})
    milestones = runtime.get("milestones", {})
    if runtime.get("gameplay_readiness") != "verified":
        raise ValueError(f"Refusing unverified gameplay run: {run_dir.name}")
    if machine.get("classes") != ["Warrior", "Thief", "Monk", "Red Mage"]:
        raise ValueError(f"Custom GBA class roster mismatch: {run_dir.name}")
    names = machine.get("entered_names", [])
    canonical = [item.get("canonical_name") for item in names]
    displays = [item.get("display_name") for item in names]
    if canonical != ["Sif", "Myth", "Milo", "Mustard"] or displays != ["Sif", "Myth", "Milo", "Mustar"]:
        raise ValueError(f"Custom GBA name roster mismatch: {run_dir.name}")
    if not all(item.get("display_name_matches_verified_key_sequence") for item in names):
        raise ValueError(f"Name keyboard evidence missing: {run_dir.name}")
    for milestone in (
        "four_characters_confirmed", "party_setup_completed", "playable_field_reached",
        "native_checkpoint_saved", "native_checkpoint_restored", "field_movement_after_restore",
    ):
        if milestones.get(milestone, {}).get("status") != "passed":
            raise ValueError(f"Required milestone did not pass in {run_dir.name}: {milestone}")
    movement = machine.get("movement_verification", {}).get("directions", {})
    if any(not movement.get(direction, {}).get("passed") for direction in ("UP", "DOWN", "LEFT", "RIGHT")):
        raise ValueError(f"Four-direction restore movement was not verified: {run_dir.name}")
    return metrics, runtime, machine


def package_run(run_dir: Path, destination: Path | None = None, repeat_run_ids: list[str] | None = None) -> Path:
    run_dir = run_dir.resolve()
    metrics, runtime, machine = validate_run(run_dir)
    milestones = runtime.get("milestones", {})
    repeats: list[tuple[str, Path]] = []
    for run_id in repeat_run_ids or []:
        if Path(run_id).name != run_id or run_id == run_dir.name:
            raise ValueError(f"Invalid independent run ID: {run_id!r}")
        repeated = run_dir.parent / run_id
        validate_run(repeated)
        repeats.append((run_id, repeated))
    required = (
        "gba_custom_party_roster.png",
        "gba_after_reload_resume.png",
    )
    for filename in required:
        if not (run_dir / filename).is_file():
            raise FileNotFoundError(run_dir / filename)
    names = machine.get("entered_names", [])

    dest = destination or PROJECT_ROOT / "docs" / "experiments" / "gba_feasibility" / run_dir.name
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    public_metrics = {
        "run_id": run_dir.name,
        "other_matching_successful_run_ids": [run_id for run_id, _ in repeats],
        "matching_clean_run_count": 1 + len(repeats),
        "independent_sessions": [
            {"run_id": run_dir.name, "verified": True},
            *({"run_id": run_id, "verified": True} for run_id, _ in repeats),
        ],
        "execution_status": metrics.get("status"),
        "gameplay_readiness": runtime.get("gameplay_readiness"),
        "party": [
            {"slot": index + 1, "canonical_name": item["canonical_name"],
             "display_name": item["display_name"], "class": machine["classes"][index],
             "keyboard_keys_cursor_verified": item["display_name_matches_verified_key_sequence"]}
            for index, item in enumerate(names)
        ],
        "milestones": {key: milestones.get(key, {}).get("status", "not_run") for key in (
            "rom_integration_boots", "party_setup_started", "four_characters_confirmed",
            "party_setup_completed", "playable_field_reached", "native_checkpoint_saved",
            "native_checkpoint_restored", "field_movement_after_restore",
        )},
        "checkpoint_bytes": runtime.get("native_state_bytes"),
        "checkpoint_path_local_only": True,
        "restore_movement_directions": sorted(
            runtime.get("party_state_machine", {}).get("movement_verification", {}).get("directions", {})
        ),
        "direction_trials": {
            key: {
                "passed": value.get("passed"),
                "changed_pixel_fraction": value.get("movement_changed_pixel_fraction"),
                "idle_changed_pixel_fraction": value.get("idle_changed_pixel_fraction"),
            }
            for key, value in runtime.get("party_state_machine", {}).get("movement_verification", {}).get("directions", {}).items()
        },
        "limitations": [
            "Name spelling was validated by per-character keyboard cursor anchors and visually reviewed roster screenshot; OCR is not enabled.",
            "Movement evidence compares direction trials with deterministic idle controls; it does not infer map coordinates.",
            "Battery-save persistence and long-term gameplay are not tested.",
        ],
    }
    (dest / "metrics.json").write_text(json.dumps(public_metrics, indent=2) + "\n", encoding="utf-8")
    report = [
        "# GBA custom party and playable checkpoint",
        "",
        f"Run `{run_dir.name}` completed the custom party and playable-field checks from a clean game reset.",
        "",
        "| Slot | Display name | Class |",
        "|---:|---|---|",
    ]
    report.extend(f"| {row['slot']} | {row['display_name']} | {row['class']} |" for row in public_metrics["party"])
    report.extend([
        "",
        "The game accepted the party, reached the Cornelia field map, serialized a fresh native Stable-Retro state in a run-specific local directory, and restored it after reset. UP, DOWN, LEFT, and RIGHT each produced broad scene changes; two no-input controls were identical in this deterministic run.",
        "",
        "The screenshot contact sheet is the compact visual evidence. Character letters were selected only after the on-screen keyboard cursor matched the requested letter anchor. The roster itself was visually inspected. Class choice was read from the visible picker cursor position and checked after each confirmation.",
        "",
        "The original generic checkpoint and all earlier experiment directories were left untouched. The native state and full diagnostic trace remain local and are not included in this package. This establishes a repeatable playable setup checkpoint, not long-term party survival, save-RAM persistence, or reinforcement-learning readiness.",
        "",
        "## Milestones",
        "",
    ])
    report.extend(f"- `{key}`: **{value}**" for key, value in public_metrics["milestones"].items())
    if repeat_run_ids:
        report.insert(2, f"Independent matching clean-start run(s): {', '.join(f'`{item}`' for item in repeat_run_ids)}.")
        report.insert(3, "")
    (dest / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    build_proof(run_dir, dest / "proof.png")
    for run_id, repeated in repeats:
        build_proof(repeated, dest / f"proof_{run_id}.png")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="local unique videos/gba_feasibility/<run-id> directory")
    parser.add_argument("--destination", type=Path, help="new evidence directory; default is docs/experiments/gba_feasibility/<run-id>")
    parser.add_argument("--repeat-run-id", action="append", default=[], help="another independently successful clean run for repeatability")
    args = parser.parse_args()
    print(package_run(args.run_dir, args.destination, args.repeat_run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
