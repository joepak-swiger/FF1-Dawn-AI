#!/usr/bin/env python3
"""Fork a local GBA battle state and record bounded input/menu screenshots.

This diagnostic never writes back to the source state or official checkpoint.
It is for verifying menu actions before they are used by a controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ff1_dawn_ai.gba_env import FF1GbaEnv, GBA_ACTIONS  # noqa: E402
from scripts.gba_gym_smoke import discover_checkpoint  # noqa: E402


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(state_path: Path, actions: list[str], output_root: Path, *, save_states: bool = False,
        action_frames: int = 1, frame_sequence: list[int] | None = None,
        release_frames: int = 1) -> Path:
    from PIL import Image
    from ff1_dawn_ai.gba_autoplay import BattleCommandDetector

    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    invalid = [action for action in actions if action not in GBA_ACTIONS]
    if invalid:
        raise ValueError(f"unknown controls: {invalid}")
    if len(actions) > 100:
        raise ValueError("probe is limited to 100 one-frame inputs")
    if not 1 <= action_frames <= 60:
        raise ValueError("action_frames must be between 1 and 60")
    if frame_sequence is not None and (len(frame_sequence) != len(actions)
                                       or any(not 1 <= count <= 60 for count in frame_sequence)):
        raise ValueError("frame_sequence must provide 1..60 frames for each action")
    if not 0 <= release_frames <= 8:
        raise ValueError("release_frames must be between 0 and 8")
    per_action_frames = frame_sequence or [action_frames] * len(actions)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    state = state_path.read_bytes()
    anchor_dir = ROOT / "docs" / "experiments" / "gba_autoplay" / "battle_ui_reference"
    anchor_detectors = {
        name: BattleCommandDetector(str(anchor_dir / filename), threshold=0.95)
        for name, filename in (
            ("attack_command", "attack_command_positive_left.png"),
            ("goblin_target", "goblin_target_positive_left.png"),
            ("magic_menu", "magic_menu_negative_left.png"),
        ) if (anchor_dir / filename).is_file()
    }
    checkpoint = discover_checkpoint()
    env = FF1GbaEnv(checkpoint, project_root=ROOT, max_steps=len(actions) + 8,
                    action_frames=action_frames, render_mode="rgb_array", restore_count=2)
    observations = []
    try:
        env.reset(seed=2026)
        env.retro_env.em.set_state(state)
        # Stable-Retro needs the same two-frame framebuffer sync used by the
        # project's native-state replay checks before a frame is evidence.
        for _ in range(2):
            env.step(GBA_ACTIONS.index("NOOP"))
        initial = env.render().copy()
        Image.fromarray(initial).save(output / "00_before.png")
        observations.append({"index": 0, "action": "before", "sha256": _sha256(initial.tobytes()),
                             "anchor_scores": {name: round(detector.score(initial), 6)
                                               for name, detector in anchor_detectors.items()}})
        for index, (action, held_frames) in enumerate(zip(actions, per_action_frames), start=1):
            env.action_frames = held_frames
            env.step(GBA_ACTIONS.index(action))
            immediate = env.render().copy()
            Image.fromarray(immediate).save(output / f"{index:02d}_after_action_{action.lower()}.png")
            for _ in range(release_frames):
                env.action_frames = 1
                env.step(GBA_ACTIONS.index("NOOP"))
            frame = env.render().copy()
            Image.fromarray(frame).save(output / f"{index:02d}_after_{action.lower()}.png")
            if save_states:
                (output / f"{index:02d}_after_{action.lower()}.state").write_bytes(
                    env.retro_env.em.get_state()
                )
            observations.append({"index": index, "action": action,
                                 "sha256_immediate": _sha256(immediate.tobytes()),
                                 "sha256": _sha256(frame.tobytes()),
                                 "release_sync_frames": release_frames,
                                 "anchor_scores": {name: round(detector.score(frame), 6)
                                                   for name, detector in anchor_detectors.items()}})
    finally:
        env.close()
    report = {
        "run_id": run_id,
        "status": "completed",
        "source_state_sha256": _sha256(state),
        "input_count": len(actions),
        "frames_per_input": per_action_frames,
        "release_frames_after_input": release_frames,
        "actions": actions,
        "observations": observations,
        "native_states_saved": bool(save_states),
        "evidence": "local screenshot sequence from an in-memory state fork; no gameplay result inferred",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Combat menu probe: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path, help="local Stable-Retro .state captured during battle")
    parser.add_argument("--actions", nargs="+", required=True, choices=GBA_ACTIONS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "videos" / "gba_combat_menu_probe")
    parser.add_argument("--save-states", action="store_true",
                        help="Also save local-only native states after each input for follow-up branching")
    parser.add_argument("--action-frames", type=int, default=1,
                        help="Frames to hold each control; bounded to 1..60")
    parser.add_argument("--frame-sequence", type=int, nargs="+",
                        help="Optional per-input frame counts, one value for each --actions entry")
    parser.add_argument("--release-frames", type=int, default=1,
                        help="Released NOOP frames after each input before the post-action evidence frame")
    args = parser.parse_args(argv)
    try:
        run(args.state, args.actions, args.output_root, save_states=args.save_states,
            action_frames=args.action_frames, frame_sequence=args.frame_sequence,
            release_frames=args.release_frames)
    except Exception as exc:
        print(f"COMBAT MENU PROBE FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
