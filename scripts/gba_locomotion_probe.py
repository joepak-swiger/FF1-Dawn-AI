"""Bounded live GBA movement, reversal, blocked-path, and RAM probe.

This is evaluation-only: it starts from the verified Seed-party checkpoint,
uses short held-control pulses, writes local raw samples under videos/, and
never changes rewards or trains a policy. RAM offsets remain unverified.
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
from ff1_dawn_ai.gba_position import analyze_ram_position_candidates, reversible_offsets  # noqa: E402
from scripts.gba_eval import changed_fraction, estimate_scene_translation  # noqa: E402
from scripts.gba_gym_smoke import discover_checkpoint  # noqa: E402


def make_run_dir(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    path = parent / stamp
    path.mkdir()
    return path


def digest(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def make_contact_sheet(items: list[tuple[str, str]], folder: Path, target: Path) -> None:
    columns, tile_w, tile_h = 4, 220, 168
    sheet = Image.new(
        "RGB", (columns * tile_w, ((len(items) + columns - 1) // columns) * tile_h),
        "#10182b",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, filename) in enumerate(items):
        x, y = (index % columns) * tile_w + 6, (index // columns) * tile_h
        image = Image.open(folder / filename).convert("RGB").resize(
            (208, 138), Image.Resampling.NEAREST
        )
        sheet.paste(image, (x, y + 22))
        draw.text((x, y + 4), label[:30], fill="white", font=font)
    sheet.save(target, optimize=True)


def run_probe(
    checkpoint: Path,
    *,
    repeats: int = 3,
    horizons: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    blocked_frames: int = 128,
    output_parent: Path | None = None,
) -> Path:
    if repeats < 3:
        raise ValueError("At least three independent repeats are required")
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("horizons must be positive")
    if blocked_frames < max(horizons):
        raise ValueError("blocked_frames must be at least the largest horizon")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = make_run_dir(output_parent or ROOT / "videos" / "gba_locomotion")
    env = FF1GbaEnv(
        checkpoint, project_root=ROOT, max_steps=blocked_frames + 70,
        action_frames=1, render_mode="rgb_array", restore_count=2,
    )
    baseline_ram: list[np.ndarray] = []
    direction_ram: dict[str, dict[int, list[np.ndarray]]] = {
        direction: {n: [] for n in horizons} for direction in ("UP", "DOWN", "LEFT", "RIGHT")
    }
    baseline_frames: list[np.ndarray] = []
    noop_ram: dict[int, list[np.ndarray]] = {n: [] for n in horizons}
    movement_metrics: dict[str, dict[str, list[dict]]] = {
        direction: {str(n): [] for n in horizons}
        for direction in ("UP", "DOWN", "LEFT", "RIGHT")
    }
    trace: list[dict] = []
    reset_hashes: list[str] = []
    start_time = time.perf_counter()

    def fresh() -> tuple[np.ndarray, np.ndarray]:
        obs, info = env.reset()
        image = env.render()
        ram = np.asarray(env.retro_env.get_ram(), dtype=np.uint8).copy()
        reset_hashes.append(digest(image))
        return image, ram

    def press(name: str) -> tuple[np.ndarray, np.ndarray]:
        obs, _, terminated, truncated, _info = env.step(GBA_ACTIONS.index(name))
        if terminated or truncated:
            raise RuntimeError(f"unexpected episode end during {name}")
        return env.render(), np.asarray(env.retro_env.get_ram(), dtype=np.uint8).copy()

    try:
        for repeat in range(repeats):
            image, ram = fresh()
            baseline_frames.append(image.copy())
            baseline_ram.append(ram)
        # Each duration begins from a fresh native restore. Release two frames
        # before sampling so a held-key register is not mistaken for position.
        for horizon in horizons:
            for repeat in range(repeats):
                image, _ = fresh()
                for _ in range(horizon + 2):
                    image, ram = press("NOOP")
                noop_ram[horizon].append(ram.copy())
                Image.fromarray(image).save(output / f"noop_r{repeat + 1}_f{horizon:03}.png")
            for direction in direction_ram:
                for repeat in range(repeats):
                    start_frame, _start_ram = fresh()
                    image = start_frame
                    for _ in range(horizon):
                        image, _ = press(direction)
                    for _ in range(2):
                        image, ram = press("NOOP")
                    direction_ram[direction][horizon].append(ram.copy())
                    Image.fromarray(image).save(
                        output / f"{direction.lower()}_r{repeat + 1}_f{horizon:03}.png"
                    )
                    movement_metrics[direction][str(horizon)].append({
                        "scene_translation_candidate": estimate_scene_translation(start_frame, image),
                        "changed_pixel_fraction_vs_reset": changed_fraction(start_frame, image),
                        "screen_hash_equals_reset": digest(start_frame) == digest(image),
                    })
                    trace.append({
                        "protocol": "held_direction_then_release", "repeat": repeat + 1,
                        "direction": direction, "held_emulator_frames": horizon,
                        "release_noop_frames": 2, "screen_sha256": digest(image),
                    })

        return_tests: dict[str, dict] = {}
        for outbound, inbound, label in (
            ("UP", "DOWN", "vertical"), ("LEFT", "RIGHT", "horizontal")
        ):
            for horizon in (8, 16, 32):
                starts, outs, backs, idles = [], [], [], []
                for repeat in range(repeats):
                    start_frame, start_ram = fresh()
                    image, out_ram = start_frame, start_ram
                    for _ in range(horizon):
                        image, out_ram = press(outbound)
                    outbound_frame = image.copy()
                    for _ in range(2):
                        outbound_frame, out_ram = press("NOOP")
                    outbound_ram = out_ram.copy()
                    for _ in range(horizon):
                        image, back_ram = press(inbound)
                    for _ in range(2):
                        image, back_ram = press("NOOP")
                    returned_frame = image.copy()
                    # Matched control has the same elapsed frame count.
                    idle_frame, idle_ram = fresh()
                    for _ in range(2 * horizon + 4):
                        idle_frame, idle_ram = press("NOOP")
                    starts.append(start_ram)
                    outs.append(outbound_ram)
                    backs.append(back_ram)
                    idles.append(idle_ram)
                    if repeat == 0:
                        for suffix, proof in (("start", start_frame), ("out", outbound_frame),
                                              ("return", returned_frame), ("noop", idle_frame)):
                            Image.fromarray(proof).save(
                                output / f"{label}_n{horizon}_{suffix}.png"
                            )
                    return_tests[f"{label}_n{horizon}_r{repeat + 1}"] = {
                        "outbound_scene_translation": estimate_scene_translation(start_frame, outbound_frame),
                        "return_scene_translation": estimate_scene_translation(start_frame, returned_frame),
                        "idle_scene_translation": estimate_scene_translation(start_frame, idle_frame),
                        "return_vs_start_changed_fraction": changed_fraction(start_frame, returned_frame),
                        "idle_vs_start_changed_fraction": changed_fraction(start_frame, idle_frame),
                        "screen_return_hash_match": digest(start_frame) == digest(returned_frame),
                    }
                return_tests[f"{label}_n{horizon}_ram_return_offsets"] = reversible_offsets(
                    starts, outs, backs, idles
                )

        # A prolonged northward route is a bounded obstacle candidate only.
        # No collision is inferred unless the rendered evidence is independently reviewed.
        for repeat in range(repeats):
            image, _ = fresh()
            for frame_index in range(1, blocked_frames + 1):
                image, _ram = press("UP")
                if frame_index in (32, 64, 96, blocked_frames):
                    Image.fromarray(image).save(
                        output / f"north_path_r{repeat + 1}_f{frame_index:03}.png"
                    )
                trace.append({"protocol": "bounded_north_path", "repeat": repeat + 1,
                              "direction": "UP", "emulator_frame": frame_index,
                              "screen_sha256": digest(image)})

        castle_followup: dict[str, list[dict]] = {str(n): [] for n in (8, 16, 32, 64)}
        for horizon in (8, 16, 32, 64):
            for mode in ("UP", "NOOP"):
                for repeat in range(repeats):
                    image, _ = fresh()
                    for _ in range(blocked_frames):
                        image, _ = press("UP")
                    for _ in range(2):
                        image, _ = press("NOOP")
                    transition_frame = image.copy()
                    for _ in range(horizon):
                        image, _ = press(mode)
                    for _ in range(2):
                        image, _ = press("NOOP")
                    metric = {
                        "mode": mode,
                        "repeat": repeat + 1,
                        "scene_translation_candidate": estimate_scene_translation(transition_frame, image),
                        "changed_pixel_fraction_vs_transition_frame": changed_fraction(transition_frame, image),
                    }
                    castle_followup[str(horizon)].append(metric)
                    Image.fromarray(image).save(
                        output / f"castle_{mode.lower()}_n{horizon}_r{repeat + 1}.png"
                    )

        selected_horizon = 32 if 32 in horizons else max(horizons)
        candidates_by_horizon = {}
        for horizon in horizons:
            controls = {"NOOP": noop_ram[horizon]}
            controls.update({name: data[horizon] for name, data in direction_ram.items()})
            candidates_by_horizon[str(horizon)] = analyze_ram_position_candidates(
                baseline_ram, controls, min_repeats=3
            )
        ram_candidates = candidates_by_horizon[str(selected_horizon)]
        checkpoint_after_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metrics = {
            "run_id": output.name,
            "checkpoint_run_id": checkpoint.parent.name,
            "checkpoint_unchanged": checkpoint_hash == checkpoint_after_hash,
            "checkpoint_sha256": checkpoint_hash,
            "repeat_count": repeats,
            "reset_observation_hashes_identical": len(set(reset_hashes)) == 1,
            "reset_count": len(reset_hashes),
            "action_frames_per_step": 1,
            "movement_horizons_emulator_frames": list(horizons),
            "reversal_horizons_emulator_frames_each_way": [8, 16, 32],
            "blocked_path_direction": "UP",
            "blocked_path_bounded_frames": blocked_frames,
            "post_transition_followup": castle_followup,
            "ram_size_bytes": ram_candidates["ram_size"],
            "ram_candidate_method_horizon": selected_horizon,
            "ram_position_candidates": ram_candidates,
            "ram_candidates_by_horizon": candidates_by_horizon,
            "directional_movement_metrics": movement_metrics,
            "reversal_tests": return_tests,
            "interpretation": {
                "cardinal_visual_response": "candidate evidence; compare with elapsed-frame no-op and inspect the saved images",
                "reversal": "only exact repeatable RAM offsets are listed; image hashes include animation and are not a position oracle",
                "blocked_terrain": "not automatically confirmed; review north_path screenshots and matched no-op controls",
                "ram_offsets": "unverified byte offsets; no coordinate meaning assigned",
                "position_tracker": "not enabled unless a coordinate signal is independently validated",
                "rewards": "neutral; no training ran",
            },
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            "raw_ram_archive": "ram_samples.npz (local, ignored)",
        }
        np.savez_compressed(
            output / "ram_samples.npz",
            baseline=np.stack(baseline_ram),
            noop=np.stack(noop_ram[selected_horizon]),
            **{f"{name.lower()}_{selected_horizon}": np.stack(values)
               for name, values in direction_ram.items()},
        )
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        (output / "action_trace.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in trace),
            encoding="utf-8",
        )
        movement_items = [("Reset / no-op", "noop_r1_f008.png")]
        for horizon in (8, 16, 32, 64):
            if horizon in horizons:
                movement_items.extend(
                    (f"{direction} {horizon} frames", f"{direction.lower()}_r1_f{horizon:03}.png")
                    for direction in ("UP", "DOWN", "LEFT", "RIGHT")
                )
        make_contact_sheet(movement_items, output, output / "movement_sheet.png")
        reversal_items = [
            (f"{axis} {horizon} {suffix}", f"{axis}_n{horizon}_{suffix}.png")
            for axis in ("vertical", "horizontal")
            for horizon in (8, 16, 32)
            for suffix in ("start", "out", "return", "noop")
        ]
        make_contact_sheet(reversal_items, output, output / "reversal_sheet.png")
        north_items = [(f"North route {horizon}", f"north_path_r1_f{horizon:03}.png")
                       for horizon in (32, 64, 96, blocked_frames)]
        north_items.extend((label, filename) for label, filename in (
            ("Castle UP 64", "castle_up_n64_r1.png"),
            ("Castle no-op 64", "castle_noop_n64_r1.png"),
        ))
        make_contact_sheet(north_items, output, output / "north_path_sheet.png")
        report = [
            "# GBA controlled locomotion probe", "",
            f"Run `{output.name}` used the verified Seed-party checkpoint from `{checkpoint.parent.name}`.",
            f"- Reset image hashes identical: {metrics['reset_observation_hashes_identical']} ({len(reset_hashes)} restores).",
            f"- Held cardinal inputs sampled at {', '.join(map(str, horizons))} emulator frames, with {repeats} independent resets each.",
            f"- RAM exposed by Stable-Retro: {ram_candidates['ram_size']} bytes; horizon {selected_horizon} candidate byte counts: {ram_candidates['directional_candidate_counts']}.",
            f"- Direction-responsive bytes passing repeat/no-op filters at {selected_horizon} frames: {ram_candidates['directional_candidate_counts']}; the report retains only the leading 200 offsets per horizon, with no coordinate meaning assigned.",
            "- Reversal trials compare UP/DOWN and LEFT/RIGHT paths against elapsed-frame no-op controls at 8, 16, and 32 frames per leg.",
            "- A 128-frame bounded northward path was captured for obstacle review; plateau is not equated with blocked terrain.",
            "- This run is evaluation-only. Rewards remain zero and no policy training was run.",
            "- Raw RAM, detailed action traces, and full-size frames are local under the ignored videos/ directory.",
        ]
        (output / "run_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
        return output
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="verified native Seed-party state")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--blocked-frames", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="validate runtime/checkpoint without stepping")
    args = parser.parse_args()
    checkpoint = args.checkpoint or discover_checkpoint()
    if args.dry_run:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(json.dumps({"runtime_ready": True, "checkpoint": str(checkpoint),
                          "checkpoint_bytes": checkpoint.stat().st_size}, indent=2))
        return 0
    result = run_probe(checkpoint, repeats=args.repeats, blocked_frames=args.blocked_frames)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
