"""Capture the locally observed DoS name-entry screen without changing saves.

This short diagnostic always starts a fresh new game in the registered
Stable-Retro core, navigates the saved menu route, and captures before/after
opening the first roster name. It writes only to a unique ignored videos dir.
"""

from __future__ import annotations

import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gba_feasibility_probe import GAME, _run  # noqa: E402
from ff1_dawn_ai.gba_party_setup import BOOT_ROUTE, Screen, class_picker_choice, classify_screen


def main() -> int:
    import stable_retro

    root = PROJECT_ROOT / "videos" / "gba_feasibility" / "name_keyboard_probe"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    output = root / run_id
    output.mkdir(parents=True, exist_ok=False)
    stable_retro.data.Integrations.add_custom_path(str(PROJECT_ROOT / "custom_integrations"))
    env = stable_retro.make(
        game=GAME,
        state=stable_retro.State.NONE,
        inttype=stable_retro.data.Integrations.ALL,
        use_restricted_actions=stable_retro.Actions.ALL,
        render_mode="rgb_array",
    )
    trace = []

    def tap(name: str) -> np.ndarray:
        index = [str(button).upper() for button in env.buttons].index(name)
        _run(env, index, 8)
        return _run(env, None, 18)

    try:
        reset = env.reset()
        obs = np.asarray(reset[0] if isinstance(reset, tuple) else reset)
        obs = _run(env, None, 360)
        for transition in BOOT_ROUTE:
            before = classify_screen(obs)
            obs = tap(transition.button)
            after = classify_screen(obs)
            waited = 0
            while after != transition.expected and waited < 20:
                obs = _run(env, None, 30)
                waited += 1
                after = classify_screen(obs)
            trace.append({"action": transition.button, "before": before.value,
                          "after": after.value, "waits": waited})
            if after != transition.expected:
                break
        Image.fromarray(obs[:, :, :3].astype(np.uint8)).save(output / "party_editor_before_name.png")
        if classify_screen(obs) == Screen.PARTY_SETUP:
            party_state = env.em.get_state()
            obs = tap("LEFT")
            trace.append({"action": "LEFT", "purpose": "move roster focus from class to name",
                          "after": classify_screen(obs).value})
            Image.fromarray(obs[:, :, :3].astype(np.uint8)).save(output / "party_editor_name_focus.png")
            obs = tap("A")
            trace.append({"action": "A", "purpose": "open first character name entry",
                          "after": classify_screen(obs).value})
            Image.fromarray(obs[:, :, :3].astype(np.uint8)).save(output / "name_entry_screen.png")
            keyboard_state = env.em.get_state()
            key_targets = {
                "S": (1, 8), "i": (3, 8), "f": (3, 5), "M": (1, 2),
                "y": (5, 4), "t": (4, 9), "h": (3, 7), "l": (4, 1),
                "o": (4, 4), "u": (5, 0), "s": (4, 8), "a": (3, 0),
                "r": (4, 7), "e": (3, 4), "d": (3, 3),
            }
            target_sheet = Image.new("RGB", (4 * 240, 4 * 182), (20, 20, 20))
            target_draw = ImageDraw.Draw(target_sheet)
            target_hashes = {}
            for index, (letter, (target_row, target_col)) in enumerate(key_targets.items()):
                env.reset()
                env.em.set_state(keyboard_state)
                key_frame = _run(env, None, 1)
                for direction, count in (("DOWN", target_row), ("UP", 0),
                                         ("RIGHT", target_col), ("LEFT", 0)):
                    button = [str(value).upper() for value in env.buttons].index(direction)
                    for _ in range(count):
                        _run(env, button, 1)
                        key_frame = _run(env, None, 6)
                target_name = f"key_target_{ord(letter):02x}.png"
                Image.fromarray(key_frame[:, :, :3].astype(np.uint8)).save(output / target_name)
                target_hashes[letter] = hashlib.sha256(
                    np.ascontiguousarray(key_frame[35:135, 98:240, :3]).tobytes()
                ).hexdigest()
                thumb = Image.fromarray(key_frame[:, :, :3].astype(np.uint8))
                x, y = (index % 4) * 240, (index // 4) * 182
                target_sheet.paste(thumb, (x, y + 22))
                target_draw.text((x + 4, y + 4), f"{letter} {target_row},{target_col}", fill=(255, 255, 255))
            target_sheet.save(output / "target_keys_contact_sheet.png")
            (output / "target_key_hashes.json").write_text(
                json.dumps(target_hashes, indent=2) + "\n", encoding="utf-8")
            for direction in ("UP", "DOWN"):
                index = [str(button).upper() for button in env.buttons].index(direction)
                for frames in (1, 2, 4, 8):
                    env.reset()
                    env.em.set_state(keyboard_state)
                    _run(env, index, frames)
                    trial = _run(env, None, 18)
                    Image.fromarray(trial[:, :, :3].astype(np.uint8)).save(
                        output / f"name_entry_{direction.lower()}_{frames}f.png")
            for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                env.reset()
                env.em.set_state(keyboard_state)
                index = [str(button).upper() for button in env.buttons].index(direction)
                _run(env, index, 8)
                moved = _run(env, None, 18)
                Image.fromarray(moved[:, :, :3].astype(np.uint8)).save(
                    output / f"name_entry_after_{direction.lower()}.png")
                trace.append({"action": direction, "purpose": "single cursor step from A",
                              "after": classify_screen(moved).value})
            thumbs = []
            env.reset()
            env.em.set_state(keyboard_state)
            frame = _run(env, None, 1)
            for count in range(37):
                if count:
                    frame = tap("RIGHT")
                name = f"cursor_right_{count:02d}.png"
                Image.fromarray(frame[:, :, :3].astype(np.uint8)).save(output / name)
                thumb = Image.fromarray(frame[:, :, :3].astype(np.uint8)).resize((240, 160))
                thumbs.append((name, thumb))
            sheet = Image.new("RGB", (5 * 240, 8 * 182), (20, 20, 20))
            draw = ImageDraw.Draw(sheet)
            for i, (name, thumb) in enumerate(thumbs):
                x, y = (i % 5) * 240, (i // 5) * 182
                sheet.paste(thumb, (x, y + 22))
                draw.text((x + 4, y + 4), name, fill=(255, 255, 255))
            sheet.save(output / "cursor_right_contact_sheet.png")
            # Exercise a known sample name through the observed flat keyboard
            # row order. This is a diagnostic only; it creates no battery save.
            env.reset()
            env.em.set_state(keyboard_state)
            _run(env, None, 1)
            current = (0, 0)

            def short_tap(name: str) -> np.ndarray:
                index = [str(button).upper() for button in env.buttons].index(name)
                _run(env, index, 1)
                return _run(env, None, 6)

            def move_to(target: tuple[int, int]) -> None:
                nonlocal current, frame
                row, col = target
                for direction, count in (("DOWN", max(0, row - current[0])),
                                         ("UP", max(0, current[0] - row)),
                                         ("RIGHT", max(0, col - current[1])),
                                         ("LEFT", max(0, current[1] - col))):
                    for _ in range(count):
                        frame = short_tap(direction)
                current = target

            for letter, target in (("S", (1, 8)), ("i", (3, 8)), ("f", (3, 5))):
                move_to(target)
                Image.fromarray(frame[:, :, :3].astype(np.uint8)).save(
                    output / f"sif_cursor_{letter}.png")
                frame = short_tap("A")
                trace.append({"action": "A", "purpose": f"select {letter}", "cursor_cell": current})
            Image.fromarray(frame[:, :, :3].astype(np.uint8)).save(output / "sif_name_entry.png")
            frame = short_tap("START")
            Image.fromarray(frame[:, :, :3].astype(np.uint8)).save(output / "sif_roster.png")
            env.reset()
            env.em.set_state(party_state)
            frame = _run(env, None, 1)
            for _ in range(2):
                frame = short_tap("DOWN")
            Image.fromarray(frame[:, :, :3].astype(np.uint8)).save(output / "row3_focus.png")
            frame = short_tap("A")
            trace.append({"class_screen": classify_screen(frame).value,
                          "class_choice": class_picker_choice(frame)})
            Image.fromarray(frame[:, :, :3].astype(np.uint8)).save(output / "row3_class_picker.png")
            class_state = env.em.get_state()
            for label, sequence in (("warrior", ()), ("thief", ("RIGHT",)),
                                    ("monk", ("DOWN",)), ("red_mage", ("RIGHT", "DOWN"))):
                env.reset()
                env.em.set_state(class_state)
                frame = _run(env, None, 1)
                for direction in sequence:
                    frame = short_tap(direction)
                region = np.ascontiguousarray(frame[29:108, 100:240, :3])
                digest = hashlib.sha256(region.tobytes()).hexdigest()
                Image.fromarray(frame[:, :, :3].astype(np.uint8)).save(
                    output / f"class_choice_{label}.png")
                trace.append({"class_choice_probe": label, "sequence": sequence,
                              "recognized": class_picker_choice(frame), "crop_sha256": digest})
        (output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        print(output)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
