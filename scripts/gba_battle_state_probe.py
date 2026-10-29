#!/usr/bin/env python3
"""Compare one-button continuations from a saved local GBA battle state.

This diagnostic forks an ignored native state into independent emulator
instances. It is not a controller and does not infer battle outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(checkpoint: Path, buttons: list[str], settle_frames: int, action_frames: int) -> Path:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS
    from ff1_dawn_ai.gba_landmark import SceneAnchorClassifier
    from ff1_dawn_ai.gba_autoplay import BattleCommandDetector, BattleTargetDetector
    from scripts.gba_castle_learning_smoke import FIELD_VARIANTS, REFERENCES

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not set(buttons) <= {"A", "B", "NOOP", "UP", "DOWN", "LEFT", "RIGHT"}:
        raise ValueError("probe buttons are limited to GBA directions, A, B, and NOOP")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out = ROOT / "videos" / "gba_battle_probe" / stamp
    out.mkdir(parents=True)
    classifier = SceneAnchorClassifier(REFERENCES, field_reference_variants=FIELD_VARIANTS)
    ui_reference = ROOT / "docs/experiments/gba_autoplay/battle_ui_reference"
    command = BattleCommandDetector(str(ui_reference / "attack_command_positive_left.png"))
    target = BattleTargetDetector(str(ui_reference / "goblin_target_positive_left.png"))
    reports = []
    for button in buttons:
        env = FF1GbaEnv(checkpoint, project_root=ROOT, max_steps=settle_frames + action_frames + 4,
                        action_frames=action_frames, render_mode="rgb_array", restore_count=2)
        frames = []
        try:
            obs, reset_info = env.reset(seed=1729)
            before = env.render()
            frames.append(Image.fromarray(before))
            obs, reward, terminated, truncated, info = env.step(GBA_ACTIONS.index(button))
            immediate = env.render()
            frames.append(Image.fromarray(immediate))
            selected: list[tuple[int, np.ndarray]] = []
            for frame_no in range(1, settle_frames + 1):
                obs, reward, terminated, truncated, info = env.step(GBA_ACTIONS.index("NOOP"))
                frame = env.render()
                if frame_no in {1, 2, 8, 16, 32, 60, 120, 180, 240, 300, settle_frames}:
                    selected.append((frame_no, frame.copy()))
                if terminated or truncated:
                    break
            for _, frame in selected:
                frames.append(Image.fromarray(frame))
            labels = []
            for index, image in enumerate(frames):
                frame = np.asarray(image)
                scene = classifier.classify(frame)
                sample_frame = 0 if index == 0 else 1 if index == 1 else selected[index - 2][0]
                labels.append({"sample": index, "emulator_frame_after_initial_action": sample_frame,
                               "scene_candidate": scene.label,
                               "scene_status": scene.status,
                               "battle_command_score": round(command.score(frame), 5),
                               "target_panel_score": round(target.score(frame), 5)})
            canvas = Image.new("RGB", (240, (160 + 18) * len(frames)), "#101828")
            draw = ImageDraw.Draw(canvas)
            for index, image in enumerate(frames):
                canvas.paste(image.resize((240, 160), Image.Resampling.NEAREST), (0, index * 178 + 18))
                draw.text((3, index * 178 + 2), f"{button} branch sample {index}", fill="white",
                          font=ImageFont.load_default())
            canvas.save(out / f"{button.lower()}_contact_sheet.png", optimize=True)
            endpoint_path = out / f"{button.lower()}_endpoint.state"
            endpoint_path.write_bytes(env.retro_env.em.get_state())
            start_array = np.asarray(frames[0])
            end_array = np.asarray(frames[-1])
            pixel_delta_fraction = float(np.any(start_array != end_array, axis=2).mean())
            reports.append({"button": button, "reset_hash": reset_info.get("observation_sha256"),
                            "reset_state_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            "action_frames_held": action_frames,
                            "endpoint_state_local_only": endpoint_path.name,
                            "endpoint_state_sha256": hashlib.sha256(endpoint_path.read_bytes()).hexdigest(),
                            "start_to_end_pixel_delta_fraction": round(pixel_delta_fraction, 6),
                            "post_action_samples": labels,
                            "battle_outcome": "unknown", "reward": 0.0})
        finally:
            env.close()
    metrics = {"run_id": stamp, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               "settle_frames": settle_frames, "action_frames_held": action_frames,
               "branches": reports, "optimizer_timesteps": 0}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    report = ["# GBA battle-state continuation probe", f"Run ID: {stamp}",
              f"Checkpoint SHA-256: {metrics['checkpoint_sha256']}",
              f"Branches: {', '.join(buttons)}; initial input held {action_frames} frame(s); neutral settling cap: {settle_frames} frames.",
              "Each branch restored the same local native checkpoint in a separate emulator instance.",
              "Scene/anchor scores and pixel deltas are diagnostic only; a changed screen does not prove player displacement or a battle outcome.",
              "Native states and full frames remain local."]
    (out / "run_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Battle-state probe output: {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--buttons", nargs="+", choices=("A", "B", "NOOP", "UP", "DOWN", "LEFT", "RIGHT"),
                        default=("A", "B"))
    parser.add_argument("--settle-frames", type=int, choices=range(1, 601), default=360)
    parser.add_argument("--action-frames", type=int, choices=range(1, 17), default=8)
    args = parser.parse_args(argv)
    try:
        run(args.checkpoint, list(args.buttons), args.settle_frames, args.action_frames)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
