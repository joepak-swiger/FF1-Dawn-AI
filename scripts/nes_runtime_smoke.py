"""Quick NES core, input, RAM, screenshot, and in-memory state round-trip check."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from ff1_dawn_ai.nes_env import FF1NesEnv


def run_smoke(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    env = None
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "game": "FinalFantasy-Nes",
        "status": "failed",
    }
    try:
        env = FF1NesEnv(project_root=PROJECT_ROOT, max_steps=4, action_freq=16, print_buttons=False)
        obs, _ = env.reset(seed=1)
        core = env.retro_env
        initial_frame = np.asarray(core.render())
        if initial_frame.ndim != 3 or initial_frame.shape[2] < 3:
            raise RuntimeError(f"Unexpected RGB frame shape: {initial_frame.shape}")

        get_ram = getattr(core, "get_ram", None)
        if not callable(get_ram):
            raise RuntimeError("Stable-Retro environment does not expose get_ram()")
        ram = np.asarray(get_ram(), dtype=np.uint8).reshape(-1)
        if ram.size == 0:
            raise RuntimeError("Stable-Retro returned an empty RAM array")

        saved_state = core.em.get_state()
        if not saved_state:
            raise RuntimeError("Stable-Retro returned an empty native emulator state")
        _, _, terminated, truncated, info = env.step(4)  # FF1NesEnv action 4 is A.
        if info.get("button") != "A":
            raise RuntimeError(f"A-button action was not dispatched as expected: {info.get('button')!r}")
        if terminated or truncated:
            raise RuntimeError("The short A-button smoke action unexpectedly ended the episode")

        core.em.set_state(saved_state)
        restored_frame = np.asarray(core.render())
        if restored_frame.shape != initial_frame.shape or not np.array_equal(restored_frame, initial_frame):
            raise RuntimeError("Frame after native state reload differs from the saved starting frame")

        proof_path = output_dir / "nes_smoke_boot.png"
        Image.fromarray(restored_frame[:, :, :3].astype(np.uint8)).save(proof_path)
        report.update({
            "status": "passed",
            "observation_shape": list(np.asarray(obs).shape),
            "frame_shape": list(initial_frame.shape),
            "frame_sha256": hashlib.sha256(initial_frame.tobytes()).hexdigest(),
            "ram_bytes": int(ram.size),
            "native_state_bytes": len(saved_state),
            "input_action": "A",
            "native_reload_frame_match": True,
            "proof_image": str(proof_path),
        })
        return report
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if env is not None:
            env.close()
        (output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New, unique directory for the screenshot and smoke report")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    try:
        report = run_smoke(output_dir)
    except Exception as exc:
        print(f"NES runtime smoke test FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Smoke report, if created: {output_dir / 'smoke_report.json'}", file=sys.stderr)
        return 1
    print(f"NES runtime smoke test passed: RAM={report['ram_bytes']} bytes; native state round-trip matched.")
    print(f"Proof: {report['proof_image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
