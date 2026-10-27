"""Inspect local GBA state formats or run an isolated Stable-Retro state smoke."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ff1_dawn_ai.gba_feasibility import inspect_file  # noqa: E402

GAME = "FinalFantasyDawnOfSouls-GbAdvance"
DEFAULT_FILES = (
    PROJECT_ROOT / "states" / "ff1_after_party_grass_cornelia.ss0",
    PROJECT_ROOT / "states" / "ff1_after_party_menu.ss0",
    PROJECT_ROOT / "states" / "roundtrip_test.state",
    PROJECT_ROOT / "states" / "ff1_after_party.sav",
)


def unique_output_dir(requested: Path | None) -> Path:
    root = requested or PROJECT_ROOT / "videos" / "gba_feasibility"
    root = root if root.is_absolute() else PROJECT_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    if requested:
        if any(root.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {root}")
        return root
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = root / run_id
    output.mkdir()
    return output


def static_inspection(paths: list[Path]) -> dict:
    inspected = []
    for path in paths:
        if path.exists():
            inspected.append(inspect_file(path))
        else:
            inspected.append({"name": path.name, "missing": True})
    return {"status": "completed", "artifacts": inspected}


def _step(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, _, terminated, truncated, _ = result
        return np.asarray(obs), bool(terminated or truncated)
    obs, _, done, _ = result
    return np.asarray(obs), bool(done)


def _run(env, button_index: int | None, frames: int) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.int8)
    if button_index is not None:
        action[button_index] = 1
    obs = None
    for _ in range(frames):
        obs, done = _step(env, action)
        if done:
            raise RuntimeError("GBA integration ended unexpectedly during smoke input")
    return obs


def runtime_smoke(output_dir: Path, automate_party_setup: bool = False) -> dict:
    """Generate a checkpoint from this Stable-Retro core and verify resume."""
    try:
        import stable_retro
    except ImportError as exc:
        raise RuntimeError("stable_retro is unavailable in this Python environment") from exc

    integration_root = PROJECT_ROOT / "custom_integrations"
    rom_path = integration_root / GAME / "rom.gba"
    if not rom_path.is_file():
        raise FileNotFoundError(f"Missing local GBA ROM: {rom_path}")

    stable_retro.data.Integrations.add_custom_path(str(integration_root))
    env = stable_retro.make(
        game=GAME,
        state=stable_retro.State.NONE,
        inttype=stable_retro.data.Integrations.ALL,
        render_mode="rgb_array",
    )
    report = {
        "game": GAME,
        "status": "failed",
        "state_origin": "serialized directly from this Stable-Retro core",
        "external_mgba_state_loaded": False,
        "integration_files_modified": False,
    }
    try:
        reset = env.reset()
        obs = np.asarray(reset[0] if isinstance(reset, tuple) else reset)
        if obs.ndim != 3 or obs.shape[2] < 3:
            raise RuntimeError(f"Unexpected reset observation shape: {obs.shape}")

        button_names = list(getattr(env, "buttons", []))
        indices = {str(name).upper(): index for index, name in enumerate(button_names) if name}
        missing = [name for name in ("A", "START") if name not in indices]
        if missing:
            raise RuntimeError(f"Stable-Retro integration does not expose buttons: {missing}")

        def tap(name: str) -> np.ndarray:
            _run(env, indices[name], 8)
            return _run(env, None, 18)

        # Deterministic boot/menu input. The optional setup macro reuses the
        # previously explored route, but its screenshot still needs visual
        # confirmation before it can be called a playable post-party state.
        obs = _run(env, None, 360)
        if automate_party_setup:
            for _ in range(20):
                tap("START")
                obs = tap("A")
            for name in ("SELECT", "DOWN", "SELECT", "DOWN", "SELECT", "DOWN", "SELECT"):
                if name not in indices:
                    raise RuntimeError(f"Party setup route requires button not exposed by core: {name}")
                obs = tap(name)
            obs = _run(env, None, 60)
            checkpoint_image = output_dir / "gba_candidate_party_state.png"
        else:
            obs = tap("START")
            obs = tap("A")
            checkpoint_image = output_dir / "gba_before_checkpoint.png"
        Image.fromarray(obs[:, :, :3].astype(np.uint8)).save(checkpoint_image)

        state = env.em.get_state()
        if not state:
            raise RuntimeError("Stable-Retro returned an empty native emulator state")
        (output_dir / "stable_retro_core.state").write_bytes(state)

        # Continue from the generated checkpoint, reset the same emulator,
        # reload it, and replay identical inputs. Matching frames establish a
        # real native-core round trip without importing an mGBA savestate.
        expected = _run(env, indices["A"], 8)
        expected = _run(env, None, 24)
        env.reset()
        env.em.set_state(state)
        actual = _run(env, indices["A"], 8)
        actual = _run(env, None, 24)
        matched = expected.shape == actual.shape and np.array_equal(expected, actual)
        Image.fromarray(actual[:, :, :3].astype(np.uint8)).save(output_dir / "gba_after_reload_resume.png")
        if not matched:
            raise RuntimeError("Native Stable-Retro state failed deterministic reset/reload/resume comparison")

        report.update({
            "status": "passed",
            "observation_shape": list(obs.shape),
            "action_space_shape": list(env.action_space.shape),
            "button_names": button_names,
            "tested_inputs": ["START", "A"],
            "party_setup_macro_attempted": bool(automate_party_setup),
            "native_state_bytes": len(state),
            "reset_reload_resume_frame_match": True,
            "checkpoint_path": str(output_dir / "stable_retro_core.state"),
            "screenshots": [
                str(checkpoint_image),
                str(output_dir / "gba_after_reload_resume.png"),
            ],
            "party_creation_verified": False,
            "candidate_party_screenshot": str(checkpoint_image) if automate_party_setup else None,
            "battery_save_verified": False,
        })
        return report
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true", help="also run a short live Stable-Retro boot/input/native-state round trip")
    parser.add_argument("--automate-party-setup", action="store_true", help="replay the previously explored START/A and name-entry macro before making the native checkpoint; visually verify the saved screenshot")
    parser.add_argument("--output-dir", type=Path, help="new empty report directory; default is a unique videos/gba_feasibility/<UTC-id>")
    parser.add_argument("--files", nargs="*", type=Path, help="state/save artifacts for read-only format inspection")
    args = parser.parse_args()
    if args.automate_party_setup and not args.runtime:
        parser.error("--automate-party-setup requires --runtime")

    output_dir = None
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_requested": bool(args.runtime),
        "status": "failed",
        "format_inspection": {"status": "not_started", "artifacts": []},
    }
    try:
        output_dir = unique_output_dir(args.output_dir)
        report.update({
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runtime_requested": bool(args.runtime),
            "status": "completed",
            "format_inspection": static_inspection(args.files or list(DEFAULT_FILES)),
        })
        if args.runtime:
            report["runtime"] = runtime_smoke(output_dir, args.automate_party_setup)
            report["status"] = report["runtime"]["status"]
        (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report_text = [
            "FF1 Dawn of Souls GBA feasibility probe",
            f"Status: {report['status']}",
            "Format inspection: read-only; no ROM or integration files modified.",
        ]
        for item in report["format_inspection"]["artifacts"]:
            if item.get("missing"):
                report_text.append(f"Missing artifact: {item['name']}")
            else:
                chunk_summary = ", ".join(
                    f"{chunk['type']}:{chunk['payload_bytes']}" for chunk in item.get("png_chunks", [])
                ) or "no PNG chunks"
                report_text.append(f"{item['name']}: {item['size_bytes']} bytes; PNG={item['is_png_container']}; {chunk_summary}")
        if args.runtime:
            runtime = report["runtime"]
            report_text.append(f"Native core reset/reload/resume frame match: {runtime.get('reset_reload_resume_frame_match', False)}")
            report_text.append("Party creation and battery-save loading are not covered by this short smoke.")
        (output_dir / "run_report.txt").write_text("\n".join(report_text) + "\n", encoding="utf-8")
        print(f"GBA feasibility report: {output_dir / 'run_report.txt'}")
        print(f"Metrics: {output_dir / 'metrics.json'}")
        return 0 if report["status"] == "completed" or report["status"] == "passed" else 1
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        if output_dir is not None:
            (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            (output_dir / "run_report.txt").write_text(
                f"FF1 Dawn of Souls GBA feasibility probe\nStatus: failed\nFailure: {report['failure']}\n",
                encoding="utf-8",
            )
            print(f"Failure report: {output_dir / 'run_report.txt'}", file=sys.stderr)
        print(f"GBA feasibility probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
