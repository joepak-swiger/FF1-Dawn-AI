"""Inspect local GBA state formats or run an isolated Stable-Retro state smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def _save_trace_frame(output_dir: Path, rows: list[dict], step: int, button: str, phase: str,
                      obs: np.ndarray, previous: np.ndarray | None) -> None:
    """Save one post-input image and a durable trace row for party-route diagnosis."""
    image = np.asarray(obs)[:, :, :3].astype(np.uint8)
    name = f"party_trace_{step:03d}_{button.lower()}_{phase}.png"
    Image.fromarray(image).save(output_dir / name)
    fraction = None
    if previous is not None and previous.shape == image.shape:
        fraction = float(np.mean(np.any(previous != image, axis=2)))
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    known_screens = {item["image_sha256"]: item["screen_id"] for item in rows}
    screen_id = known_screens.get(image_hash)
    if screen_id is None:
        screen_id = f"screen_{len(known_screens) + 1:03d}"
    row = {
        "step": step,
        "button": button,
        "phase": phase,
        "changed_pixel_fraction": "" if fraction is None else f"{fraction:.6f}",
        "screen_id": screen_id,
        "image_sha256": image_hash,
        "screenshot": name,
    }
    rows.append(row)
    trace_path = output_dir / "party_setup_trace.csv"
    write_header = not trace_path.exists()
    with trace_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _summarize_party_trace(output_dir: Path, rows: list[dict]) -> dict:
    """Create compact ordered contact sheets and exact-frame repeat statistics.

    Screen IDs indicate exact screenshot equality only; they do not name or
    classify game screens and cannot prove an input caused a transition.
    """
    released = [row for row in rows if row["phase"] in {"boot_idle", "released"}]
    repeated = []
    for previous, current in zip(released, released[1:]):
        if previous["image_sha256"] == current["image_sha256"]:
            repeated.append({"previous_step": previous["step"], "step": current["step"], "button": current["button"]})
    tile_w, thumb_w, thumb_h, caption_h, columns = 160, 144, 96, 30, 5
    sheets = []
    for sheet_index, offset in enumerate(range(0, len(released), 25), start=1):
        group = released[offset:offset + 25]
        row_count = math.ceil(len(group) / columns)
        canvas = Image.new("RGB", (columns * tile_w, row_count * (thumb_h + caption_h)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, event in enumerate(group):
            with Image.open(output_dir / event["screenshot"]) as source:
                thumb = source.convert("RGB")
                thumb.thumbnail((thumb_w, thumb_h))
            x, y = (index % columns) * tile_w, (index // columns) * (thumb_h + caption_h)
            canvas.paste(thumb, (x + (tile_w - thumb.width) // 2, y))
            delta = event["changed_pixel_fraction"] or "first"
            caption = f"{event['step']} {event['button']} {event['screen_id']}"
            draw.text((x + 2, y + thumb_h + 2), caption[:26], fill="black", font=ImageFont.load_default())
            draw.text((x + 2, y + thumb_h + 16), f"pixel change {delta}"[:25], fill="black", font=ImageFont.load_default())
        canvas.quantize(colors=48, method=Image.Quantize.MEDIANCUT).save(
            output_dir / f"party_setup_contact_sheet_{sheet_index:02d}.png", optimize=True
        )
        sheets.append(f"party_setup_contact_sheet_{sheet_index:02d}.png")
    unique_screens = len({row["image_sha256"] for row in released})
    summary = {
        "ordered_action_screens": sum(row["phase"] == "released" for row in rows),
        "boot_idle_screens": sum(row["phase"] == "boot_idle" for row in rows),
        "unique_exact_frame_signatures": unique_screens,
        "consecutive_exact_duplicate_screens": repeated,
        "contact_sheets": sheets,
        "screen_semantics_classified": False,
        "identical_frames_prove_stall": False,
    }
    (output_dir / "party_setup_trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _initial_milestones() -> dict:
    return {
        "rom_integration_boots": {"status": "not_run", "evidence": "No runtime launched."},
        "input_controls_function": {"status": "not_verified", "evidence": "Mapped buttons and accepted action calls do not alone prove in-game control."},
        "party_creation_works": {"status": "not_verified", "evidence": "Requires visual confirmation of the resulting party screen."},
        "playable_scene_reached": {"status": "not_verified", "evidence": "A title, logo, or character-selection frame is not a playable checkpoint."},
        "native_state_serialized": {"status": "not_run", "evidence": "No runtime state created."},
        "state_restored_after_reset": {"status": "not_run", "evidence": "No reset/reload attempt run."},
        "gameplay_resumes_correctly": {"status": "not_verified", "evidence": "Requires a confirmed playable scene and a verified resumed game action."},
    }


def _repeated_pair_boundary_frame(obs: np.ndarray, seen: dict[str, int], pair_index: int) -> tuple[bool, str]:
    """Detect an exact screenshot cycle at a completed START/A pair boundary.

    This is a conservative stop signal only. It does not classify the screen
    or prove that the game is stalled; it prevents replaying the same macro
    cycle indefinitely when an exact visual state reappears.
    """
    signature = hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest()
    previous = seen.get(signature)
    if previous is not None:
        return True, f"pair {pair_index} returned the exact frame already seen after pair {previous}"
    seen[signature] = pair_index
    return False, ""


def _merge_runtime_results(report: dict, runtime: dict) -> None:
    """Keep top-level status and milestones consistent with the live probe."""
    report["runtime"] = runtime
    report["status"] = runtime.get("status", "failed")
    report["milestones"] = runtime.get("milestones", report.get("milestones", _initial_milestones()))
    gameplay_milestones = ("party_creation_works", "playable_scene_reached", "gameplay_resumes_correctly")
    if all(report["milestones"].get(key, {}).get("status") == "passed" for key in gameplay_milestones):
        report["gameplay_readiness"] = "verified"
    else:
        report["gameplay_readiness"] = "not_verified"


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
        "milestones": _initial_milestones(),
    }
    trace_rows: list[dict] = []
    trace_previous: np.ndarray | None = None
    trace_step = 0
    if automate_party_setup:
        report["party_setup_trace_csv"] = str(output_dir / "party_setup_trace.csv")
    try:
        reset = env.reset()
        obs = np.asarray(reset[0] if isinstance(reset, tuple) else reset)
        if obs.ndim != 3 or obs.shape[2] < 3:
            raise RuntimeError(f"Unexpected reset observation shape: {obs.shape}")
        report["milestones"]["rom_integration_boots"] = {
            "status": "partially_verified",
            "evidence": f"Stable-Retro reset returned RGB observation {list(obs.shape)}; this confirms core initialization, but the image is not automatically classified as a game boot/title screen.",
        }

        button_names = list(getattr(env, "buttons", []))
        indices = {str(name).upper(): index for index, name in enumerate(button_names) if name}
        missing = [name for name in ("A", "START") if name not in indices]
        if missing:
            raise RuntimeError(f"Stable-Retro integration does not expose buttons: {missing}")
        report["button_mapping"] = {name: indices[name] for name in indices}
        report["milestones"]["input_controls_function"] = {
            "status": "partially_verified",
            "evidence": "Required A/START (and route buttons when requested) are mapped and action calls returned observations; in-game navigation effect is not independently confirmed.",
        }

        def tap(name: str) -> np.ndarray:
            nonlocal trace_previous, trace_step
            pressed = _run(env, indices[name], 8)
            if automate_party_setup:
                _save_trace_frame(output_dir, trace_rows, trace_step, name, "pressed", pressed, trace_previous)
                trace_previous = pressed
            released = _run(env, None, 18)
            if automate_party_setup:
                _save_trace_frame(output_dir, trace_rows, trace_step, name, "released", released, trace_previous)
                trace_previous = released
                trace_step += 1
            return released

        # Deterministic boot/menu input. The optional setup macro reuses the
        # previously explored route, but its screenshot still needs visual
        # confirmation before it can be called a playable post-party state.
        obs = _run(env, None, 360)
        if automate_party_setup:
            _save_trace_frame(output_dir, trace_rows, trace_step, "NONE", "boot_idle", obs, None)
            trace_previous = obs
            trace_step += 1
        if automate_party_setup:
            seen_pair_frames: dict[str, int] = {}
            loop_detected = False
            loop_reason = ""
            pairs_completed = 0
            for pair_index in range(1, 21):
                tap("START")
                obs = tap("A")
                pairs_completed += 1
                loop_detected, loop_reason = _repeated_pair_boundary_frame(obs, seen_pair_frames, pair_index)
                if loop_detected:
                    break
            if not loop_detected:
                for name in ("SELECT", "DOWN", "SELECT", "DOWN", "SELECT", "DOWN", "SELECT"):
                    if name not in indices:
                        raise RuntimeError(f"Party setup route requires button not exposed by core: {name}")
                    obs = tap(name)
            obs = _run(env, None, 60)
            checkpoint_image = output_dir / "gba_candidate_party_state.png"
            report["party_setup_pairs_completed"] = pairs_completed
            report["party_setup_loop_detected"] = loop_detected
            report["party_setup_stop_reason"] = loop_reason or "configured route completed"
        else:
            obs = tap("START")
            obs = tap("A")
            checkpoint_image = output_dir / "gba_before_checkpoint.png"
        Image.fromarray(obs[:, :, :3].astype(np.uint8)).save(checkpoint_image)

        state = env.em.get_state()
        if not state:
            raise RuntimeError("Stable-Retro returned an empty native emulator state")
        state_path = output_dir / "stable_retro_core.state"
        bytes_written = state_path.write_bytes(state)
        if bytes_written != len(state) or state_path.read_bytes() != state:
            raise RuntimeError("Native Stable-Retro state did not round-trip to disk byte-for-byte")
        report["milestones"]["native_state_serialized"] = {
            "status": "passed",
            "evidence": f"Active Stable-Retro core serialized {len(state)} bytes and the local file matched byte-for-byte.",
        }

        # Continue from the generated checkpoint, reset the same emulator,
        # reload it, and replay identical inputs. Matching frames establish a
        # real native-core round trip without importing an mGBA savestate.
        expected = _run(env, indices["A"], 8)
        expected = _run(env, None, 24)
        report["milestones"]["state_restored_after_reset"] = {
            "status": "in_progress",
            "evidence": "Reset and native state reload started; deterministic continuation is being checked.",
        }
        try:
            env.reset()
            env.em.set_state(state)
            actual = _run(env, indices["A"], 8)
            actual = _run(env, None, 24)
        except Exception as restore_error:
            report["milestones"]["state_restored_after_reset"] = {
                "status": "failed",
                "evidence": f"Reset/reload/replay raised {type(restore_error).__name__}: {restore_error}",
            }
            raise
        matched = expected.shape == actual.shape and np.array_equal(expected, actual)
        Image.fromarray(actual[:, :, :3].astype(np.uint8)).save(output_dir / "gba_after_reload_resume.png")
        report["milestones"]["state_restored_after_reset"] = {
            "status": "passed" if matched else "failed",
            "evidence": "After env.reset and native set_state, replayed A/idle frames were compared byte-for-byte with the pre-reset continuation.",
        }
        if not matched:
            raise RuntimeError("Native Stable-Retro state failed deterministic reset/reload/resume comparison")

        trace_analysis = _summarize_party_trace(output_dir, trace_rows) if automate_party_setup else None
        report.update({
            "status": "completed",
            "gameplay_readiness": "not_verified",
            "observation_shape": list(obs.shape),
            "action_space_shape": list(env.action_space.shape),
            "button_names": button_names,
            "tested_inputs": ["START", "A"],
            "party_setup_macro_attempted": bool(automate_party_setup),
            "party_setup_pairs_completed": report.get("party_setup_pairs_completed", 0),
            "party_setup_loop_detected": report.get("party_setup_loop_detected", False),
            "party_setup_stop_reason": report.get("party_setup_stop_reason"),
            "party_setup_trace_rows": len(trace_rows),
            "party_setup_trace_csv": str(output_dir / "party_setup_trace.csv") if automate_party_setup else None,
            "party_setup_trace_screenshots": len(trace_rows),
            "party_setup_trace_summary": "party_setup_trace_summary.json" if automate_party_setup else None,
            "party_setup_trace_analysis": trace_analysis,
            "party_setup_contact_sheets": trace_analysis["contact_sheets"] if trace_analysis else [],
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
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.close()
        if automate_party_setup and trace_rows and "party_setup_trace_summary" not in report:
            try:
                report["party_setup_trace_summary"] = "party_setup_trace_summary.json"
                report["party_setup_trace_analysis"] = _summarize_party_trace(output_dir, trace_rows)
            except Exception as trace_error:
                report["trace_summary_failure"] = f"{type(trace_error).__name__}: {trace_error}"
        if automate_party_setup:
            report["party_setup_trace_screenshots"] = len(trace_rows)
            trace_path = output_dir / "party_setup_trace.csv"
            report["party_setup_trace_csv"] = str(trace_path) if trace_path.is_file() else None
        (output_dir / "runtime_progress.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


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
        "milestones": _initial_milestones(),
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
            _merge_runtime_results(report, runtime_smoke(output_dir, args.automate_party_setup))
        (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report_text = [
            "FF1 Dawn of Souls GBA feasibility probe",
            f"Execution status: {report['status']}",
            f"Gameplay readiness: {report.get('gameplay_readiness', 'not_verified')}",
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
            report_text.append(f"Party creation/playability: {report.get('gameplay_readiness', 'not_verified')}.")
            if runtime.get("party_setup_macro_attempted"):
                report_text.append(f"Party macro: {runtime.get('party_setup_pairs_completed', 0)} START/A pairs; loop detected={runtime.get('party_setup_loop_detected', False)}; stop reason={runtime.get('party_setup_stop_reason')}.")
            report_text.append("Battery-save loading is not verified by this probe.")
            if runtime.get("party_setup_trace_csv"):
                report_text.append(f"Party macro trace: {runtime['party_setup_trace_csv']} ({runtime.get('party_setup_trace_rows', 0)} screenshots/rows; inspect each input in order).")
                report_text.append(f"Party macro contact sheets: {', '.join(runtime.get('party_setup_contact_sheets', [])) or 'not generated'}.")
        report_text.extend(["", "Milestones:"])
        for name, item in report.get("milestones", {}).items():
            report_text.append(f"- {name}: {item['status']} — {item['evidence']}")
        report_text.append("- gameplay_resumes_correctly is not established by a frame match unless the checkpoint is separately confirmed playable.")
        (output_dir / "run_report.txt").write_text("\n".join(report_text) + "\n", encoding="utf-8")
        print(f"GBA feasibility report: {output_dir / 'run_report.txt'}")
        print(f"Metrics: {output_dir / 'metrics.json'}")
        return 0 if report["status"] == "completed" else 1
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        if output_dir is not None:
            progress_path = output_dir / "runtime_progress.json"
            if progress_path.is_file():
                try:
                    report["runtime"] = json.loads(progress_path.read_text(encoding="utf-8"))
                    report["milestones"] = report["runtime"].get("milestones", report["milestones"])
                except (OSError, json.JSONDecodeError):
                    pass
            (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            failure_lines = [
                "FF1 Dawn of Souls GBA feasibility probe",
                "Status: failed",
                f"Failure: {report['failure']}",
                "",
                "Milestones:",
            ]
            failure_lines.extend(f"- {name}: {item['status']} — {item['evidence']}" for name, item in report["milestones"].items())
            runtime = report.get("runtime", {})
            for key, label in (
                ("party_setup_trace_csv", "Party trace CSV"),
                ("party_setup_trace_summary", "Party trace summary"),
                ("candidate_party_screenshot", "Candidate checkpoint screenshot"),
            ):
                if runtime.get(key):
                    failure_lines.append(f"{label}: {runtime[key]}")
            if runtime.get("party_setup_contact_sheets"):
                failure_lines.append(f"Party contact sheets: {', '.join(runtime['party_setup_contact_sheets'])}")
            (output_dir / "run_report.txt").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
            print(f"Failure report: {output_dir / 'run_report.txt'}", file=sys.stderr)
        print(f"GBA feasibility probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
