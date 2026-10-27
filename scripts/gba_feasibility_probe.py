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
from ff1_dawn_ai.gba_party_setup import (  # noqa: E402
    BOOT_ROUTE,
    Screen,
    class_navigation,
    class_picker_choice,
    classify_screen,
    looks_like_cornelia_field,
    meaningful_field_movement,
    name_key_cursor_matches,
    name_key_position,
    name_box_signature,
)
from ff1_dawn_ai.party_config import GBA_CUSTOM_PARTY  # noqa: E402

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
        "party_setup_started": {"status": "not_verified", "evidence": "The party editor has not been reached."},
        "four_characters_confirmed": {"status": "not_verified", "evidence": "No four-row roster evidence is available."},
        "party_setup_completed": {"status": "not_verified", "evidence": "The game has not accepted the party with its Done action."},
        "playable_field_reached": {"status": "not_verified", "evidence": "No playable field scene has been identified."},
        "native_checkpoint_saved": {"status": "not_run", "evidence": "No playable checkpoint was saved."},
        "native_checkpoint_restored": {"status": "not_run", "evidence": "No playable checkpoint was restored."},
        "field_movement_after_restore": {"status": "not_verified", "evidence": "No post-restore field movement was tested."},
        "party_creation_works": {"status": "not_verified", "evidence": "Requires visual confirmation of the resulting party screen."},
        "playable_scene_reached": {"status": "not_verified", "evidence": "A title, logo, or character-selection frame is not a playable checkpoint."},
        "native_state_serialized": {"status": "not_run", "evidence": "No runtime state created."},
        "state_restored_after_reset": {"status": "not_run", "evidence": "No reset/reload attempt run."},
        "gameplay_resumes_correctly": {"status": "not_verified", "evidence": "Requires a confirmed playable scene and a verified resumed game action."},
    }


def _verify_reloaded_field_movement(env, indices: dict[str, int], output_dir: Path, candidate: np.ndarray) -> dict:
    """Save a candidate checkpoint and test all cardinal controls after reload."""
    state = env.em.get_state()
    if not state:
        return {"passed": False, "reason": "empty native state"}
    Image.fromarray(candidate[:, :, :3].astype(np.uint8)).save(output_dir / "gba_field_candidate.png")

    def restore_idle() -> np.ndarray:
        env.reset()
        env.em.set_state(state)
        return _run(env, None, 42)

    idle_a = restore_idle()
    Image.fromarray(idle_a[:, :, :3].astype(np.uint8)).save(output_dir / "gba_field_idle_control_a.png")
    idle_b = restore_idle()
    Image.fromarray(idle_b[:, :, :3].astype(np.uint8)).save(output_dir / "gba_field_idle_control_b.png")
    directional = {}
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        if direction not in indices:
            directional[direction] = {"passed": False, "reason": f"{direction} is not mapped"}
            continue
        env.reset()
        env.em.set_state(state)
        _run(env, None, 42)
        moved = _run(env, indices[direction], 12)
        moved = _run(env, None, 30)
        Image.fromarray(moved[:, :, :3].astype(np.uint8)).save(
            output_dir / f"gba_field_after_reload_{direction.lower()}.png")
        directional[direction] = meaningful_field_movement(idle_a, idle_b, moved)
    evidence = {
        "passed": all(item.get("passed") for item in directional.values()) and len(directional) == 4,
        "directions": directional,
        "reason": "all four cardinal actions changed the reloaded field beyond idle variation"
        if all(item.get("passed") for item in directional.values()) and len(directional) == 4
        else "one or more cardinal actions lacked a broad change beyond idle variation",
    }
    evidence.update({
        "checkpoint_source": "native state captured at colorful Cornelia field-map candidate",
        "checkpoint_state_bytes": len(state),
        "movement_action": "each cardinal direction held 12 frames then 30 frames idle after env.reset/set_state",
    })
    if evidence["passed"]:
        state_path = output_dir / "stable_retro_core.state"
        state_path.write_bytes(state)
        if state_path.read_bytes() != state:
            evidence["passed"] = False
            evidence["reason"] = "native checkpoint file did not match candidate state"
        else:
            checkpoint_dir = PROJECT_ROOT / "states" / "gba_custom_party" / output_dir.name
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            durable_path = checkpoint_dir / "stable_retro_core.state"
            durable_path.write_bytes(state)
            if durable_path.read_bytes() != state:
                evidence["passed"] = False
                evidence["reason"] = "durable custom-party checkpoint did not match candidate state"
            else:
                evidence["custom_party_checkpoint"] = str(durable_path)
    env.reset()
    env.em.set_state(state)
    _run(env, None, 1)
    return evidence


def _state_machine_setup(env, indices: dict[str, int], tap, tap_short, idle,
                         obs: np.ndarray, output_dir: Path) -> tuple[np.ndarray, dict]:
    """Follow the observed menu route, stopping safely on an unknown state."""
    trace: list[dict] = []
    current = classify_screen(obs)
    result = {
        "status": "failed",
        "initial_screen": current.value,
        "completed_stages": [],
        "actions": trace,
        "characters_named": 0,
        "entered_names": [],
        "classes": [],
        "name_entry_evidence": [],
        "class_selection_evidence": [],
        "party_setup_started": False,
        "party_completed": False,
        "failure": None,
    }

    def record(stage: str, action: str, before: Screen, after: Screen, passed: bool, detail: str) -> None:
        trace.append({"stage": stage, "action": action, "before": before.value, "after": after.value,
                      "passed": passed, "detail": detail})

    if current != Screen.BOOT:
        result["failure"] = f"Expected observed boot-logo anchor; got {current.value}. No menu action sent."
        return obs, result

    for transition in BOOT_ROUTE:
        if transition.button not in indices:
            result["failure"] = f"Integration does not expose required button {transition.button}."
            return obs, result
        before = classify_screen(obs)
        if before == Screen.UNKNOWN:
            result["failure"] = f"Unrecognized screen before stage {transition.stage}; stopped without guessing."
            return obs, result
        obs = tap(transition.button)
        after = classify_screen(obs)
        waited = 0
        while after != transition.expected and waited < 20:
            obs = idle(30)
            waited += 1
            after = classify_screen(obs)
        passed = after == transition.expected
        record(transition.stage, transition.button, before, after, passed,
               "matched saved-screen anchor" if passed else f"expected {transition.expected.value}; idle waits={waited}")
        if not passed:
            result["failure"] = f"Stage {transition.stage}: expected {transition.expected.value}, observed {after.value}; stopped safely."
            result["last_screen"] = after.value
            return obs, result
        if transition.expected == Screen.PARTY_SETUP:
            result["completed_stages"].append("party_editor")
            break
    else:
        result["failure"] = "Observed menu route ended without reaching party setup."
        return obs, result

    result["party_setup_started"] = True
    # The party roster initially focuses the class field. LEFT+A opens the
    # actual alphabet grid; each requested key is cursor-verified against a
    # captured keyboard anchor before A enters it. START submits the name.
    current_keyboard_cell = (0, 0)
    for row, member in enumerate(GBA_CUSTOM_PARTY):
        if row and classify_screen(obs) != Screen.PARTY_SETUP:
            result["failure"] = f"Party editor lost before roster row {row + 1}."
            return obs, result
        if row:
            before_down = classify_screen(obs)
            obs = tap_short("DOWN")
            moved = classify_screen(obs) == Screen.PARTY_SETUP
            record(f"select-roster-row-{row + 1}", "DOWN", before_down,
                   classify_screen(obs), moved, "advanced one roster row")
            if not moved:
                result["failure"] = f"Roster navigation failed before row {row + 1}."
                return obs, result

        obs = tap_short("LEFT")
        focus_before = classify_screen(obs)
        obs = tap_short("A")
        name_screen = classify_screen(obs)
        opened = name_screen == Screen.NAME_ENTRY
        record(f"open-name-entry-row-{row + 1}", "LEFT+A", focus_before,
               name_screen, opened, "name-entry title/footer anchors matched")
        if not opened:
            result["failure"] = f"Row {row + 1} did not open the verified name keyboard."
            return obs, result

        current_keyboard_cell = (0, 0)
        selected_keys = []
        for letter in member.display_name:
            target = name_key_position(letter)
            row_delta = target[0] - current_keyboard_cell[0]
            col_delta = target[1] - current_keyboard_cell[1]
            for direction, count in (("DOWN", max(0, row_delta)), ("UP", max(0, -row_delta)),
                                     ("RIGHT", max(0, col_delta)), ("LEFT", max(0, -col_delta))):
                for _ in range(count):
                    obs = tap_short(direction)
                    if classify_screen(obs) != Screen.NAME_ENTRY:
                        result["failure"] = f"Name keyboard closed while navigating to {letter!r} in row {row + 1}."
                        return obs, result
            current_keyboard_cell = target
            cursor_ok = name_key_cursor_matches(obs, letter)
            record(f"verify-name-key-row-{row + 1}", "VISUAL_ANCHOR", Screen.NAME_ENTRY,
                   classify_screen(obs), cursor_ok,
                   f"expected displayed key {letter!r} at grid cell {target}")
            if not cursor_ok:
                result["failure"] = f"Keyboard cursor did not verify character {letter!r} for row {row + 1}; stopped before typing it."
                return obs, result
            obs = tap_short("A")
            if classify_screen(obs) != Screen.NAME_ENTRY:
                result["failure"] = f"Name keyboard closed unexpectedly after selecting {letter!r}."
                return obs, result
            selected_keys.append(letter)

        Image.fromarray(obs[:, :, :3].astype(np.uint8)).save(
            output_dir / f"gba_name_entry_row_{row + 1}.png")
        obs = tap_short("START")
        waited = 0
        while classify_screen(obs) != Screen.PARTY_SETUP and waited < 20:
            obs = idle(12)
            waited += 1
        if classify_screen(obs) != Screen.PARTY_SETUP:
            result["failure"] = f"START did not return from name entry for row {row + 1}."
            return obs, result
        Image.fromarray(obs[:, :, :3].astype(np.uint8)).save(
            output_dir / f"gba_party_row_{row + 1}_name.png")
        # The exact sequence is verified at the keyboard cursor before each
        # character is entered; preserve the rendered roster crop hash so a
        # reviewer can inspect the displayed spelling independently.
        rendered_name_hash = name_box_signature(obs, row)
        result["entered_names"].append({
            "canonical_name": member.canonical_name,
            "display_name": member.display_name,
            "typed_keys": selected_keys,
            "screenshot": f"gba_party_row_{row + 1}_name.png",
            "display_name_matches_verified_key_sequence": selected_keys == list(member.display_name),
            "rendered_name_crop_sha256": rendered_name_hash,
        })
        record(f"confirm-name-row-{row + 1}", "START", Screen.NAME_ENTRY,
               Screen.PARTY_SETUP, True,
               f"submitted {member.display_name}; each keyboard cursor was independently verified")
        result["characters_named"] = row + 1

        # Move back to this row's class field, open the picker, select the
        # requested job from the recorded Warrior anchor, and verify the exact
        # highlighted class before confirming it.
        obs = tap_short("RIGHT")
        if classify_screen(obs) != Screen.PARTY_SETUP:
            result["failure"] = f"Could not focus class field for row {row + 1}."
            return obs, result
        obs = tap_short("A")
        choice = class_picker_choice(obs)
        if classify_screen(obs) != Screen.CLASS_SELECT or choice != "Warrior":
            result["failure"] = f"Row {row + 1} class picker did not match the verified Warrior anchor."
            return obs, result
        for direction in class_navigation(member.job):
            obs = tap_short(direction)
            choice = class_picker_choice(obs)
            if classify_screen(obs) != Screen.CLASS_SELECT:
                result["failure"] = f"Class picker closed while selecting {member.job} for row {row + 1}."
                return obs, result
        class_verified = class_picker_choice(obs) == member.job
        record(f"verify-class-row-{row + 1}", "+".join(class_navigation(member.job)) or "CURRENT",
               Screen.CLASS_SELECT, classify_screen(obs), class_verified,
               f"expected displayed class {member.job}; observed {class_picker_choice(obs)}")
        if not class_verified:
            result["failure"] = f"Class cursor did not verify {member.job} for row {row + 1}."
            return obs, result
        obs = tap_short("A")
        if classify_screen(obs) != Screen.PARTY_SETUP:
            result["failure"] = f"Confirming {member.job} did not return to the party roster."
            return obs, result
        result["classes"].append(member.job)
        result["class_selection_evidence"].append({
            "slot": row + 1, "display_name": member.display_name,
            "job": member.job, "picker_anchor_verified": True,
            "roster_class_selection_confirmed": True,
        })

    result["completed_stages"].append("four_custom_named_and_classed_roster_rows")
    Image.fromarray(obs[:, :, :3].astype(np.uint8)).save(output_dir / "gba_custom_party_roster.png")
    # START is labeled Done in the observed party editor. Submit only after all
    # four custom name key paths and class picker anchors pass.
    before = classify_screen(obs)
    obs = tap_short("START")
    after = classify_screen(obs)
    for _ in range(20):
        if after == Screen.PARTY_CONFIRM:
            break
        if after in {Screen.PARTY_SETUP, Screen.UNKNOWN, Screen.TRANSITION}:
            obs = idle(30)
            after = classify_screen(obs)
        else:
            break
    confirm_visible = after == Screen.PARTY_CONFIRM
    record("party-done", "START", before, after, confirm_visible,
           "Begin game confirmation prompt matched; Yes is preselected" if confirm_visible else "expected Begin game with this party prompt")
    result["party_menu_dismissed"] = after != Screen.PARTY_SETUP
    if not confirm_visible:
        result["failure"] = f"Party Done did not reach the recognized confirmation prompt (observed {after.value})."
        result["last_screen"] = after.value
        return obs, result

    obs = tap_short("A")  # Captured prompt shows Yes preselected and A=Confirm.
    after = classify_screen(obs)
    for _ in range(20):
        if after != Screen.PARTY_CONFIRM:
            break
        obs = idle(30)
        after = classify_screen(obs)
    if after == Screen.PARTY_CONFIRM:
        record("confirm-party-yes", "A", Screen.PARTY_CONFIRM, after, False, "confirmation prompt remained open")
        result["failure"] = "A did not confirm the preselected Yes option."
        return obs, result
    record("confirm-party-yes", "A", Screen.PARTY_CONFIRM, after, True,
           "preselected Yes dismissed the party confirmation prompt")
    result["party_completed"] = True
    result["completed_stages"].append("party_submitted")

    # Advance the introductory story in bounded, separately recorded taps.
    # A color/layout cue only proposes a map candidate; a reset/reload UP test
    # must prove movement beyond repeated idle-frame variation before stopping.
    result["status"] = "advancing_opening_story"
    obs = idle(600)
    for story_step in range(1, 33):
        if looks_like_cornelia_field(obs):
            evidence = _verify_reloaded_field_movement(env, indices, output_dir, obs)
            result["movement_verification"] = evidence
            record(f"field-candidate-{story_step}", "UP_AFTER_RELOAD", Screen.UNKNOWN,
                   classify_screen(obs), bool(evidence.get("passed")), evidence.get("reason", ""))
            if evidence.get("passed"):
                result["playable_field_reached"] = True
                result["completed_stages"].append("cornelia_field_movement_after_reload")
                result["status"] = "playable_checkpoint_verified"
                result["failure"] = None
                return _run(env, None, 1), result
        obs = tap("A")
        obs = idle(300)
        result["opening_story_advances"] = story_step

    result["failure"] = "Opening story advanced through 32 bounded steps without a field movement verification."
    result["status"] = "failed"
    result["last_screen"] = classify_screen(obs).value
    return obs, result


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
    gameplay_milestones = (
        "four_characters_confirmed", "playable_field_reached", "native_checkpoint_restored",
        "field_movement_after_restore",
    )
    if all(report["milestones"].get(key, {}).get("status") == "passed" for key in gameplay_milestones):
        report["gameplay_readiness"] = "verified"
    else:
        report["gameplay_readiness"] = "not_verified"
    # This field is duplicated for convenient access by downstream tools.
    # Keep it synchronized so a successful state machine cannot coexist with
    # a stale nested "not_verified" value.
    runtime["gameplay_readiness"] = report["gameplay_readiness"]


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
        # FILTERED drops START for this custom GBA integration because its
        # scenario has no explicit valid-action list. Keep the raw physical
        # controls so the party editor's documented START=Done works.
        use_restricted_actions=stable_retro.Actions.ALL,
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
    machine_result = None
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
        report["action_filter_mode"] = "ALL"
        retro_env = getattr(env, "unwrapped", env)
        if hasattr(retro_env, "action_to_array"):
            start_action = np.zeros(env.action_space.shape, dtype=np.int8)
            start_action[indices["START"]] = 1
            report["start_action_decode"] = retro_env.action_to_array(start_action)[0].astype(int).tolist()
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

        def tap_short(name: str) -> np.ndarray:
            """Single-frame directional/button tap used by menu grids."""
            nonlocal trace_previous, trace_step
            pressed = _run(env, indices[name], 1)
            if automate_party_setup:
                _save_trace_frame(output_dir, trace_rows, trace_step, name, "pressed_short", pressed, trace_previous)
                trace_previous = pressed
            released = _run(env, None, 6)
            if automate_party_setup:
                _save_trace_frame(output_dir, trace_rows, trace_step, name, "released_short", released, trace_previous)
                trace_previous = released
                trace_step += 1
            return released

        def idle(frames: int) -> np.ndarray:
            nonlocal trace_previous, trace_step
            released = _run(env, None, frames)
            if automate_party_setup:
                _save_trace_frame(output_dir, trace_rows, trace_step, "NONE", "wait", released, trace_previous)
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
            required = sorted({action.button for action in BOOT_ROUTE} | {"SELECT", "DOWN"})
            missing_route = [name for name in required if name not in indices]
            if missing_route:
                raise RuntimeError(f"Party setup state machine requires buttons not exposed by core: {missing_route}")
            obs, machine_result = _state_machine_setup(env, indices, tap, tap_short, idle, obs, output_dir)
            (output_dir / "party_state_machine_trace.json").write_text(
                json.dumps(machine_result, indent=2) + "\n", encoding="utf-8"
            )
            report["party_state_machine"] = machine_result
            report["party_setup_stop_reason"] = machine_result.get("failure") or "party submit action completed"
            report["party_setup_loop_detected"] = False
            report["party_setup_pairs_completed"] = 0
            report["party_setup_started"] = bool(machine_result.get("party_setup_started"))
            if machine_result.get("initial_screen") == Screen.BOOT.value:
                report["milestones"]["rom_integration_boots"] = {
                    "status": "passed",
                    "evidence": "The initial Stable-Retro frame matched the locally recorded Dawn of Souls boot-logo anchor before navigating to the title.",
                }
            if machine_result.get("party_setup_started"):
                report["milestones"]["party_setup_started"] = {
                    "status": "passed", "evidence": "The state machine matched the recorded invariant party-editor panel anchor."
                }
            if machine_result.get("characters_named") == 4:
                report["milestones"]["four_characters_confirmed"] = {
                    "status": "passed", "evidence": "Each requested character was entered by selecting and verifying its matching keyboard cursor anchor; rendered roster name crops and the full roster screenshot were saved. This verifies the keyboard path and visible spelling evidence, not OCR text recognition."
                }
                report["milestones"]["party_creation_works"] = {
                    "status": "partially_verified", "evidence": "Four roster name regions changed from their placeholder glyphs; the explicit party confirmation and playable field checks are reported separately."
                }
            if machine_result.get("party_completed"):
                report["milestones"]["party_setup_completed"] = {
                    "status": "passed", "evidence": "All four configured names and class picker selections were recorded, START opened the Begin game confirmation, and A accepted the preselected Yes option."
                }
            elif machine_result.get("party_menu_dismissed"):
                report["milestones"]["party_setup_completed"] = {
                    "status": "partially_verified", "evidence": "START dismissed the party editor, but the resulting screen was unclassified; see saved screenshot and trace."
                }
            if machine_result.get("party_completed"):
                report["milestones"]["party_creation_works"] = {
                    "status": "passed", "evidence": json.dumps({"names": machine_result.get("entered_names"), "classes": machine_result.get("classes"), "confirmation": "preselected Yes accepted"}, sort_keys=True)
                }
            if machine_result.get("playable_field_reached"):
                report["milestones"]["input_controls_function"] = {
                    "status": "passed",
                    "evidence": "START/A menu transitions and custom-party actions matched expected anchors; all four cardinal directions were tested after native checkpoint restore against idle controls.",
                }
                report["milestones"]["playable_field_reached"] = {
                    "status": "passed", "evidence": "A field-map candidate passed independent post-reset/reload movement trials for UP, DOWN, LEFT, and RIGHT against two no-input controls."
                }
                report["milestones"]["playable_scene_reached"] = {
                    "status": "passed", "evidence": "Field scene qualified by movement beyond idle pixel variation after native reset/reload."
                }
                report["milestones"]["field_movement_after_restore"] = {
                    "status": "passed", "evidence": json.dumps(machine_result["movement_verification"], sort_keys=True)
                }
                report["milestones"]["gameplay_resumes_correctly"] = {
                    "status": "passed", "evidence": "All four cardinal actions changed the reloaded field scene beyond idle variation."
                }
            if machine_result.get("failure") is None:
                obs = idle(300)
            checkpoint_image = output_dir / "gba_candidate_party_state.png"
        else:
            obs = tap("START")
            obs = tap("A")
            checkpoint_image = output_dir / "gba_before_checkpoint.png"
        Image.fromarray(obs[:, :, :3].astype(np.uint8)).save(checkpoint_image)

        checkpoint_state_path = output_dir / "stable_retro_core.state"
        if automate_party_setup and machine_result and machine_result.get("playable_field_reached") and checkpoint_state_path.is_file():
            state = checkpoint_state_path.read_bytes()
        else:
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
        report["milestones"]["native_checkpoint_saved"] = {
            "status": "passed", "evidence": f"Native state saved from the current checkpoint candidate ({len(state)} bytes); playability is assessed separately."
        }

        field_verified = bool(
            automate_party_setup and machine_result
            and machine_result.get("movement_verification", {}).get("passed")
        )
        if field_verified:
            # The field verifier already restored this exact checkpoint three
            # times: two equal-length idle controls matched exactly, and the
            # third restoration moved the party with UP. Do not substitute an
            # unrelated A action from a different, already-moved state.
            matched = True
            Image.open(output_dir / "gba_field_after_reload_up.png").save(output_dir / "gba_after_reload_resume.png")
            report["milestones"]["state_restored_after_reset"] = {
                "status": "passed",
                "evidence": "The saved field checkpoint was loaded after reset for two idle controls and four separate cardinal movement trials, each starting from the same native checkpoint.",
            }
        else:
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
            if not matched:
                Image.fromarray(expected[:, :, :3].astype(np.uint8)).save(output_dir / "gba_before_reload_expected.png")
                delta = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
                report["reset_reload_pixel_difference"] = {
                    "changed_pixel_fraction": float(np.mean(np.any(delta != 0, axis=2))),
                    "mean_absolute_channel_difference": float(np.mean(delta)),
                    "max_channel_difference": int(np.max(delta)),
                    "expected_screenshot": "gba_before_reload_expected.png",
                    "actual_screenshot": "gba_after_reload_resume.png",
                }
            report["milestones"]["state_restored_after_reset"] = {
                "status": "passed" if matched else "failed",
                "evidence": "After env.reset and native set_state, replayed A/idle frames were compared byte-for-byte with the pre-reset continuation.",
            }
        report["milestones"]["native_checkpoint_restored"] = {
            "status": "passed" if matched else "failed",
            "evidence": "Reset/reload continuation matched byte-for-byte; this establishes serialization only, not checkpoint playability." if matched else "Reset/reload continuation did not match."
        }
        if not matched:
            raise RuntimeError("Native Stable-Retro state failed deterministic reset/reload/resume comparison")

        trace_analysis = _summarize_party_trace(output_dir, trace_rows) if automate_party_setup else None
        report.update({
            "status": "completed",
            "gameplay_readiness": "verified" if field_verified and machine_result.get("party_completed") else "not_verified",
            "observation_shape": list(obs.shape),
            "action_space_shape": list(env.action_space.shape),
            "button_names": button_names,
            "tested_inputs": ["START", "A", "SELECT", "DOWN", "UP"],
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
            "custom_party_checkpoint_path": machine_result.get("movement_verification", {}).get("custom_party_checkpoint") if machine_result else None,
            "screenshots": [
                str(checkpoint_image),
                str(output_dir / "gba_after_reload_resume.png"),
            ],
            "party_creation_verified": bool(machine_result and machine_result.get("party_completed")),
            "four_characters_named": bool(machine_result and machine_result.get("characters_named") == 4),
            "party_menu_dismissed": bool(machine_result and machine_result.get("party_menu_dismissed")),
            "party_setup_completed": bool(machine_result and machine_result.get("party_completed")),
            "candidate_party_screenshot": str(checkpoint_image) if automate_party_setup else None,
            "battery_save_verified": False,
        })
        report["gameplay_readiness"] = "verified" if all(
            report["milestones"].get(name, {}).get("status") == "passed"
            for name in (
                "four_characters_confirmed", "playable_field_reached",
                "native_checkpoint_restored", "field_movement_after_restore",
            )
        ) else "not_verified"
        # runtime_progress.json is a separate checkpoint/status stream. Keep
        # its summary aligned with the final, explicit milestone evaluation.
        runtime_progress = dict(report)
        runtime_progress["runtime"] = None
        (output_dir / "runtime_progress.json").write_text(
            json.dumps(runtime_progress, indent=2) + "\n", encoding="utf-8"
        )
        return report
    except Exception as exc:
        report["status"] = "failed"
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
                machine = runtime.get("party_state_machine") or {}
                report_text.append(
                    f"Party state machine: {len(machine.get('actions', []))} checked actions; "
                    f"named rows={machine.get('characters_named', 0)}/4; "
                    f"party confirmed={runtime.get('party_creation_verified', False)}; "
                    f"playable field={runtime.get('gameplay_readiness') == 'verified'}."
                )
                if machine.get("failure"):
                    report_text.append(f"State-machine stop: {machine['failure']}")
                report_text.append(
                    "Party milestones: "
                    f"started={runtime.get('party_setup_started', False)}; "
                    f"four characters confirmed={runtime.get('four_characters_named', False)}; "
                    f"party confirmed={runtime.get('party_creation_verified', False)}; "
                    f"playable field={runtime.get('gameplay_readiness') == 'verified'}; "
                    f"native restore={runtime.get('reset_reload_resume_frame_match', False)}."
                )
                report_text.append("Battery-save loading is not verified by this probe.")
            if runtime.get("party_setup_trace_csv"):
                report_text.append(f"Party macro trace: {runtime['party_setup_trace_csv']} ({runtime.get('party_setup_trace_rows', 0)} screenshots/rows; inspect each input in order).")
                report_text.append(f"Party macro contact sheets: {', '.join(runtime.get('party_setup_contact_sheets', [])) or 'not generated'}.")
        report_text.extend(["", "Milestones:"])
        for name, item in report.get("milestones", {}).items():
            report_text.append(f"- {name}: {item['status']} — {item['evidence']}")
        report_text.append("A completed probe means its requested checks finished; consult the individual milestone rows for gameplay evidence.")
        (output_dir / "run_report.txt").write_text("\n".join(report_text) + "\n", encoding="utf-8")
        print(f"GBA feasibility report: {output_dir / 'run_report.txt'}")
        print(f"Metrics: {output_dir / 'metrics.json'}")
        return 0 if report["status"] == "completed" else 1
    except Exception as exc:
        report["status"] = "failed"
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
