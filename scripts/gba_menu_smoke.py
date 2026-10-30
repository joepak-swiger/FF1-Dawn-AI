#!/usr/bin/env python3
"""Run bounded read-only inventory and Sif equipment menu checks from reset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS  # noqa: E402
from ff1_dawn_ai.gba_menu import GbaMenuController, GbaMenuSceneClassifier  # noqa: E402
from scripts.gba_gym_smoke import discover_checkpoint  # noqa: E402


def _sha(frame) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()


def _new_output(root: Path) -> tuple[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = root / run_id
    output.mkdir(exist_ok=False)
    return run_id, output


def run(checkpoint: Path, operations: list[str], output_root: Path) -> Path:
    from PIL import Image, ImageDraw

    reference = ROOT / "docs/experiments/gba_menu_controller/reference"
    classifier = GbaMenuSceneClassifier({
        "main_menu": reference / "main_menu_items_cursor.png",
        "main_menu_equipment_cursor": reference / "main_menu_equipment_cursor.png",
        "empty_items": reference / "empty_items_menu.png",
        "equipment_character_select": reference / "equipment_character_select_sif.png",
        "sif_equipment": reference / "sif_equipment_menu.png",
    })
    run_id, output = _new_output(output_root)
    rows = []
    results = []
    failures = []
    proof_frames = []
    frame_counts = {operation: 0 for operation in operations}
    print(f"GBA menu smoke run: {output}")
    for operation in operations:
        env = FF1GbaEnv(checkpoint, project_root=ROOT, max_steps=400,
                        action_frames=1, render_mode="rgb_array", restore_count=2)
        try:
            _, reset_info = env.reset(seed=2026)
            initial = env.render().copy()
            Image.fromarray(initial).save(output / f"{operation}_00_reset.png")
            proof_frames.append((f"{operation}: reset", initial))

            def step(control: str, frames: int):
                env.action_frames = frames
                _, _, terminated, truncated, info = env.step(GBA_ACTIONS.index(control))
                frame = env.render().copy()
                rows.append({"operation": operation, "action": control, "frames": frames,
                             "frame_sha256": _sha(frame), "terminated": terminated,
                             "truncated": truncated,
                             "emulator_frames": info["emulator_frames_executed"]})
                frame_counts[operation] += 1
                Image.fromarray(frame).save(
                    output / f"{operation}_{frame_counts[operation]:03d}_{control.lower()}.png")
                if terminated or truncated:
                    raise RuntimeError(f"emulator ended during {operation}")
                proof_frames.append((f"{operation}: {control} x{frames}", frame))
                return frame

            controller = GbaMenuController(step, classifier)
            result = None
            try:
                result = (controller.inspect_inventory(initial) if operation == "inventory"
                          else controller.inspect_sif_equipment(initial))
                final_scene, final_score = classifier.classify(result.frame)
                if final_scene != result.scene:
                    raise RuntimeError(f"final scene changed before capture: {final_scene}")
                Image.fromarray(result.frame).save(output / f"{operation}_verified.png")
                proof_frames.append((f"{operation}: confirmed {final_scene}", result.frame))
                results.append({"operation": operation, "status": result.status,
                                "scene": result.scene, "scene_score": round(final_score, 6),
                                "input_frames": result.input_frames,
                                "reset_checkpoint_loaded": bool(reset_info.get("checkpoint_loaded")),
                                "reset_observation_sha256": reset_info.get("observation_sha256"),
                                "verified_result_sha256": _sha(result.frame)})
            except Exception as exc:
                last_frame = proof_frames[-1][1] if proof_frames else initial
                last_scene, last_score = classifier.classify(last_frame)
                failure = {"operation": operation, "error_type": type(exc).__name__,
                           "error": str(exc), "last_scene": last_scene,
                           "last_score": round(last_score, 6),
                           "input_frames": controller.input_frames}
                failures.append(failure)
                Image.fromarray(last_frame).save(output / f"{operation}_last_observed.png")
                results.append({"operation": operation, "status": "not_verified",
                                "scene": last_scene, "scene_score": round(last_score, 6),
                                "input_frames": controller.input_frames,
                                "reset_checkpoint_loaded": bool(reset_info.get("checkpoint_loaded")),
                                "error": f"{type(exc).__name__}: {exc}"})
            for trace_row in controller.trace:
                rows.append({"operation": operation, **trace_row,
                             "frames": trace_row.get("held_frames", 0),
                             "frame_sha256": "", "terminated": False,
                             "truncated": False, "emulator_frames": trace_row.get("held_frames", 0)})
        finally:
            env.close()

    with (output / "actions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("operation", "action", "frames", "frame_sha256", "terminated",
                  "truncated", "emulator_frames", "scene", "score", "phase")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    metrics = {"run_id": run_id, "status": "partial_failed" if failures else "completed",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "results": results, "failures": failures,
        "interpretation": {
            "inventory": "The exact empty Items screen was confirmed; no consumable was used.",
            "equipment": "The equipment path did not reach a confirmed Sif equipment screen; no gear was changed.",
            "limits": "Only reviewed Cornelia reset and menu anchors are recognized; unknown screens stop the controller.",
        }}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = [f"GBA read-only menu smoke — {run_id}", "",
              f"Checkpoint SHA-256: {metrics['checkpoint_sha256']}"]
    report.extend(f"- {row['operation']}: {row['status']} ({row['scene']}, score {row['scene_score']:.5f}, "
                  f"{row['input_frames']} input/release frames)" for row in results)
    report.extend(["", "No item was consumed and no equipment was modified.",
                   "The inventory template shows no items in this starting checkpoint.",
                   "These are read-only menu checks, not proof of potion use or shopping."])
    for failure in failures:
        report.extend(["", f"Unverified {failure['operation']}: {failure['error']}",
                       f"Last recognized scene: {failure['last_scene']} (score {failure['last_score']:.5f})."])
    (output / "run_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    width, height = 480, 320
    sheet = Image.new("RGB", (width * 2, height * ((len(proof_frames) + 1) // 2)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, frame) in enumerate(proof_frames):
        image = Image.fromarray(frame).resize((width, height), Image.Resampling.NEAREST)
        x, y = (index % 2) * width, (index // 2) * height
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + width - 1, y + 20), fill="black")
        draw.text((x + 6, y + 4), label[:68], fill="white")
    sheet.save(output / "contact_sheet.png")
    print(f"GBA menu smoke {metrics['status']}: {output}")
    if failures:
        print(f"Unverified operations: {', '.join(row['operation'] for row in failures)}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Verified local Seed-party checkpoint")
    parser.add_argument("--operations", nargs="+", choices=("inventory", "equipment"),
                        default=("inventory", "equipment"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "videos" / "gba_menu_smoke")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        checkpoint = args.checkpoint.resolve() if args.checkpoint else discover_checkpoint()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if args.dry_run:
            refs = ROOT / "docs/experiments/gba_menu_controller/reference"
            required = ("main_menu_items_cursor.png", "empty_items_menu.png",
                        "main_menu_equipment_cursor.png", "equipment_character_select_sif.png",
                        "sif_equipment_menu.png")
            missing = [name for name in required if not (refs / name).is_file()]
            if missing:
                raise FileNotFoundError(f"missing menu references: {missing}")
            print(f"Dry-run passed: checkpoint={checkpoint.stat().st_size} bytes; no emulator started")
            return 0
        output = run(checkpoint, args.operations, args.output_root)
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        if metrics["status"] != "completed":
            return 1
    except Exception as exc:
        print(f"GBA MENU SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
