"""Evidence-driven menu recognition and party setup for Dawn of Souls.

The exact-frame anchors below come from two repeatable local Stable-Retro
traces. They are deliberately conservative: an unfamiliar frame is unknown,
not guessed. The party panel anchor is a cropped, invariant UI region so name
and cursor changes do not invalidate it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from PIL import Image


class Screen(StrEnum):
    UNKNOWN = "unknown"
    BOOT = "boot_logo"
    DOS_TITLE = "dawn_of_souls_title"
    GAME_SELECT = "game_selection"
    TRANSITION = "transition"
    FF1_TITLE = "ff1_title"
    NEW_GAME = "new_game_prompt"
    PARTY_TRANSITION = "party_transition"
    PARTY_SETUP = "party_setup"
    PARTY_CONFIRM = "party_confirmation"
    CLASS_SELECT = "class_selection"
    NAME_ENTRY = "name_entry"


# SHA-256 of full RGB frames saved after released actions in the local v1.0
# trace. Exact signatures are a conservative whitelist, not general OCR.
_FULL_FRAME_STATES = {
    "9338bd11ed5f11407c1415b4081f81f5bc736171af5d0057cfb5b752e1a1c756": Screen.BOOT,
    "ff85b572bd4cecac9ff3acaa89a8c60352faef5399f8959d3c08be2997d3b660": Screen.BOOT,
    "a1f00ccd4c00135c75646919e4144d4e4cb2618dd239bb834786e5500a274cfb": Screen.DOS_TITLE,
    "355c87a51ff74e7c0922479425789ce76bab5078b07a7d3e402a268fba84c57d": Screen.DOS_TITLE,
    "a3045451ea0baf529f453ca4ca87bf01e2adb12c663b567397935d4877065442": Screen.DOS_TITLE,
    "ffdca0b66df985f9e0fd7640d51633599fbd84ff224d96c9cfea136ae014a817": Screen.TRANSITION,
    "c60484d885073f924e75556b3a6937514811575f7466cbf9897878531ab8bb70": Screen.GAME_SELECT,
    "361d3103e80729c925cb9caee94fc03500e37f4fb53a34e9b197a374cdaa731c": Screen.GAME_SELECT,
    "ee662a6c665d308789f0e9251c1a96530479c109158775f263e7bf27940df79b": Screen.GAME_SELECT,
    "819815a43eb468f45e6d2f74e62b34785d0c024c8a9e1b8b9710deabf3e96bd2": Screen.TRANSITION,
    "fa2b10f75452e3ef65e9e2d7c81853743eec8103594715af61161d984cb78019": Screen.FF1_TITLE,
    "badc3bba9f018c26de8f6adf2630ca840c5b25f2611a2258594b356bcc4dbd5c": Screen.FF1_TITLE,
    "bf17d9ad6f70492b765797c12b10718242fbdcd4c0c3c6c9efa62b104268cc1b": Screen.FF1_TITLE,
    "9dbe8c3fa330f7a172471150984c2141806ac66f03e7f85d6b4a191753dc9200": Screen.NEW_GAME,
    "e8945ae01d89c46c99a8889640d4203a09d5775bd1d52a3e4bcc4569c473d34e": Screen.PARTY_TRANSITION,
    "f3d47687dfb103155a2fa9168f5d3e28c87c4c2f742b703e3ed999c69e65f244": Screen.PARTY_TRANSITION,
}

# Invariant upper-right instruction panel hashes measured from the party editor
# and class picker. Hashing this region ignores the changing party names.
_PARTY_PANEL_SHA256 = "d54115b271798cff25168cedef392afc7367ccaf10fa115d4a5012a34cfc496e"
_CLASS_HEADER_SHA256 = "dd449c3356296e62cf988c8b8303b2b91351a0b9922f6aae6a6bbfb0b3b59725"
_CLASS_PICKER_CURSOR_BOXES = {
    "Warrior": (105, 46), "Thief": (168, 46),
    "Monk": (105, 75), "Red Mage": (168, 75),
}
_PARTY_CONFIRM_PANEL_SHA256 = "7ac10055b41a6e6580bdf7e349ef09ed73bd3a872de12fd7c3a54b74bff9df89"
_NAME_ENTRY_TITLE_SHA256 = "513a1df39bc114621fc5129f887a4e4d354251d88283e526f880f505ce582011"
_NAME_ENTRY_FOOTER_SHA256 = "e140b3061fd2985e3ccdb6a17c8011ca7686e7f3d4d926d8c478bc708c64daa8"

# The game keyboard has irregular row lengths. Letter coordinates and the
# cursor anchors below were measured from the actual name-entry screen.
NAME_KEY_ROWS = ("ABCDEFGHIJ", "KLMNOPQRST", "UVWXYZ", "abcdefghij", "klmnopqrst", "uvwxyz")
NAME_KEY_POSITIONS = {letter: (row, column) for row, keys in enumerate(NAME_KEY_ROWS) for column, letter in enumerate(keys)}
NAME_KEY_CURSOR_PHASH = {
    "S": "caa3115eeaa35156baa9544cff33044efb92885ffe502a4bcdf0aaa9cc215555",
    "i": "ca83115eeaa31556baa9544cff33044efb92885ffd28226dcf522a7bcc035555",
    "f": "caa3115eeaa311daaaa9564cff3304cedbb6885f7d182269cd64aa6f4c035555",
    "M": "caa3115ceaab155ebaa9544cfb33044cfbb28c5ffc382a69cd302aeb4c435555",
    "y": "c883155aeaa311d6aaa9544cfb3304ccfb968c5bfd0c3a6dc7303aedcc435555",
    "t": "c8a3115ceaa31156aaa91448ff3304ecfbb68d5bfd182aedcdb4226bcc435555",
    "h": "caa3155ecaa3515eaaa9544cfb3304ceff92885ffb682a69c59022bbcc435555",
    "l": "caa3115eeaa31156aaa97448ff33044eff96885ffd303a0fdd00a2f9cc435555",
    "o": "c883115acaa311deeaa91658ff3304cefbb2885ffd18226bdd302aedcc435555",
    "u": "caa3115eeaa31156aaa95448fb3304ccffb2885fff302a69c4362ae9de435555",
    "s": "ca83155ceaa35156baa9544cfb3304ecfdb6885fff10322dd5a02a5bcc435555",
    "a": "caa31558eaa3115ebaa95648fb33044cffb2885ffc3c226fcde02ae9cc035555",
    "r": "caa3115eeba31556baa9544cff33046cfbb28c57fe102a69cd6032bbcc035555",
    "e": "c883115eeaa31156aaa91448ff3304cef9968c5fff30226bddc4baedcc035555",
    "d": "ca83115ceaa31156aaa9744cff33044cff968c5ffc703a6bc714a23dcc435555",
}


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def classify_screen(frame: np.ndarray) -> Screen:
    """Classify only locally observed menu anchors; never infer unknown UI."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        return Screen.UNKNOWN
    rgb = np.ascontiguousarray(image[:, :, :3])
    exact = _FULL_FRAME_STATES.get(_sha256(rgb))
    if exact is not None:
        return exact
    if rgb.shape[:2] == (160, 240):
        panel = _sha256(rgb[0:90, 160:240, :])
        if panel == _PARTY_PANEL_SHA256:
            return Screen.PARTY_SETUP
        if panel == _PARTY_CONFIRM_PANEL_SHA256:
            return Screen.PARTY_CONFIRM
        class_header = _sha256(rgb[0:24, 160:240, :])
        if class_header == _CLASS_HEADER_SHA256:
            return Screen.CLASS_SELECT
        if (
            _sha256(rgb[0:24, 98:240, :]) == _NAME_ENTRY_TITLE_SHA256
            and _sha256(rgb[141:160, 98:240, :]) == _NAME_ENTRY_FOOTER_SHA256
        ):
            return Screen.NAME_ENTRY
    return Screen.UNKNOWN


@dataclass(frozen=True)
class MenuAction:
    button: str
    expected: Screen
    stage: str


# Observed title prompts are now reachable because raw START is enabled. The
# old 16-input path was needed only while Stable-Retro FILTERED erased START.
# Each transition has a checked screenshot family and a bounded transition wait.
BOOT_ROUTE = (
    MenuAction("START", Screen.DOS_TITLE, "open-dawn-of-souls-title"),
    MenuAction("START", Screen.GAME_SELECT, "open-game-selection"),
    MenuAction("A", Screen.FF1_TITLE, "select-final-fantasy-i"),
    MenuAction("START", Screen.NEW_GAME, "open-final-fantasy-menu"),
    MenuAction("A", Screen.PARTY_SETUP, "start-new-game-and-open-party"),
)


# Name glyph boxes from the real 240x160 party roster screenshots. Class rows
# are left intact (the observed defaults are Warrior/Thief/White/Black Mage).
NAME_BOXES = ((50, 11, 112, 27), (50, 43, 112, 60), (50, 84, 112, 101), (50, 125, 112, 142))


def name_box_signature(frame: np.ndarray, row: int) -> str:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[0] != 160 or image.shape[1] != 240 or row not in range(4):
        raise ValueError("Expected a 240x160 GBA RGB frame and party row 0..3")
    x0, y0, x1, y1 = NAME_BOXES[row]
    return _sha256(image[y0:y1, x0:x1, :3])


def name_changed(before: np.ndarray, after: np.ndarray, row: int, min_pixels: int = 4) -> bool:
    """Check the chosen name glyph box changed after Auto-Name was pressed."""
    x0, y0, x1, y1 = NAME_BOXES[row]
    a = np.asarray(before)[y0:y1, x0:x1, :3]
    b = np.asarray(after)[y0:y1, x0:x1, :3]
    return a.shape == b.shape and int(np.count_nonzero(np.any(a != b, axis=2))) >= min_pixels


def class_picker_choice(frame: np.ndarray) -> str | None:
    """Return a locally anchored class picker selection; otherwise unknown."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[:2] != (160, 240) or classify_screen(image) != Screen.CLASS_SELECT:
        return None
    # The selection cursor is a white hand immediately above the option label.
    # Its location is more stable than whole-panel hashes across roster rows.
    scores = {}
    for job, (x, y) in _CLASS_PICKER_CURSOR_BOXES.items():
        patch = image[y:y + 12, x:x + 12, :3].astype(np.int16)
        neutral_bright = (patch.min(axis=2) > 170) & ((patch.max(axis=2) - patch.min(axis=2)) < 60)
        scores[job] = int(np.count_nonzero(neutral_bright))
    choice = max(scores, key=scores.get)
    runner_up = max(value for job, value in scores.items() if job != choice)
    return choice if scores[choice] >= 6 and scores[choice] - runner_up >= 4 else None


def name_key_cursor_matches(frame: np.ndarray, letter: str) -> bool:
    """Verify the requested key cursor by a tolerant, discriminative visual anchor.

    Low-frequency perceptual hashes tolerate small frame-phase/palette changes
    while retaining a margin over the other requested key locations. A raw
    SHA comparison was too brittle when the cursor blink differed by a few
    frames between an anchor capture and an independent fresh start.
    """
    if letter not in NAME_KEY_CURSOR_PHASH or classify_screen(frame) != Screen.NAME_ENTRY:
        return False
    image = np.asarray(frame)
    crop = Image.fromarray(image[35:135, 98:240, :3]).convert("L")
    pixels = np.asarray(crop.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)
    n = np.arange(64, dtype=np.float32)
    k = np.arange(16, dtype=np.float32)[:, None]
    basis = np.cos(np.pi * (2 * n + 1) * k / (2 * 64)).astype(np.float32)
    basis[0] *= np.sqrt(1 / 64)
    basis[1:] *= np.sqrt(2 / 64)
    coefficients = basis @ pixels @ basis.T
    flattened = coefficients.ravel()
    median = np.median(flattened[1:])
    signature = sum((int(value > median) << index) for index, value in enumerate(flattened))
    anchors = {key: int(value, 16) for key, value in NAME_KEY_CURSOR_PHASH.items()}
    distances = {key: (signature ^ value).bit_count() for key, value in anchors.items()}
    expected_distance = distances[letter]
    next_distance = min(distance for key, distance in distances.items() if key != letter)
    return expected_distance <= 16 and next_distance - expected_distance >= 6


def name_key_position(letter: str) -> tuple[int, int]:
    try:
        return NAME_KEY_POSITIONS[letter]
    except KeyError as exc:
        raise ValueError(f"Unsupported GBA name-entry character: {letter!r}") from exc


def class_navigation(job: str) -> tuple[str, ...]:
    """Observed direction sequence from the class picker's Warrior anchor."""
    paths = {"Warrior": (), "Thief": ("RIGHT",), "Monk": ("DOWN",), "Red Mage": ("RIGHT", "DOWN")}
    try:
        return paths[job]
    except KeyError as exc:
        raise ValueError(f"Unsupported Dawn of Souls class: {job!r}") from exc


def looks_like_cornelia_field(frame: np.ndarray) -> bool:
    """Triage a colorful field-map layout; movement check is still required."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[:2] != (160, 240) or classify_screen(image) != Screen.UNKNOWN:
        return False
    rgb = image[:, :, :3].astype(np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    green_fraction = np.mean((green > red * 1.15) & (green > blue * 1.04) & (green > 40))
    blue_fraction = np.mean((blue > red * 1.1) & (blue > green * 1.05) & (blue > 40))
    black_fraction = np.mean(np.max(rgb, axis=2) < 25)
    return bool(green_fraction > 0.60 and blue_fraction > 0.05 and black_fraction < 0.05)


def meaningful_field_movement(idle_a: np.ndarray, idle_b: np.ndarray, moved: np.ndarray) -> dict:
    """Compare like-length no-input and directional continuations from one state."""
    a, b, c = (np.asarray(item)[:, :, :3] for item in (idle_a, idle_b, moved))
    if a.shape != b.shape or a.shape != c.shape:
        return {"passed": False, "reason": "observation shapes differ"}
    idle_delta = np.any(a != b, axis=2)
    movement_delta = np.any(b != c, axis=2)
    ys, xs = np.where(movement_delta)
    idle_fraction = float(np.mean(idle_delta))
    moved_fraction = float(np.mean(movement_delta))
    bbox = None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    min_fraction = max(0.03, idle_fraction * 5.0)
    passed = bool(bbox and moved_fraction > min_fraction and bbox[2] - bbox[0] >= 24 and bbox[3] - bbox[1] >= 24)
    return {
        "passed": passed,
        "idle_changed_pixel_fraction": idle_fraction,
        "movement_changed_pixel_fraction": moved_fraction,
        "movement_vs_idle_threshold": min_fraction,
        "movement_bbox": bbox,
        "reason": "movement differs beyond idle variation" if passed else "no broad field change beyond idle variation",
    }

