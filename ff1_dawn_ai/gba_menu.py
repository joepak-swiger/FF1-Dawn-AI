"""Conservative, read-only GBA menu navigation for reviewed checkpoint views.

This module recognizes only the exact Cornelia checkpoint menu layouts that
have been captured and reviewed locally. Unknown or changed inventory/gear
screens stop the routine; it never confirms an item use or equipment change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


MENU_SCENES = ("main_menu", "main_menu_equipment_cursor", "empty_items",
               "equipment_character_select", "sif_equipment")


class GbaMenuSceneClassifier:
    """Whole-frame template classifier for reviewed static menu screenshots."""

    def __init__(self, references: dict[str, str | Path], *, threshold: float = 0.975):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        missing = set(MENU_SCENES) - set(references)
        if missing:
            raise ValueError(f"missing reviewed menu references: {sorted(missing)}")
        self.threshold = float(threshold)
        self.references: dict[str, np.ndarray] = {}
        for name in MENU_SCENES:
            with Image.open(references[name]) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if frame.shape != (160, 240, 3):
                raise ValueError(f"{name} reference must be a 240x160 RGB image")
            self.references[name] = frame

    def scores(self, frame: np.ndarray) -> dict[str, float]:
        image = np.asarray(frame, dtype=np.uint8)
        if image.shape != (160, 240, 3):
            return {name: 0.0 for name in MENU_SCENES}
        source = image.astype(np.int16)
        return {name: 1.0 - float(np.abs(source - ref.astype(np.int16)).mean()) / 255.0
                for name, ref in self.references.items()}

    def classify(self, frame: np.ndarray) -> tuple[str, float]:
        scores = self.scores(frame)
        name = max(scores, key=scores.get)
        score = scores[name]
        return (name, score) if score >= self.threshold else ("unknown", score)


@dataclass
class MenuOperationResult:
    operation: str
    status: str
    scene: str
    frame: np.ndarray
    input_frames: int
    score: float


class GbaMenuController:
    """Bounded menu state machine that only opens read-only views.

    ``step`` must execute one named GBA control for the requested frames and
    return the resulting RGB frame. The controller explicitly releases each
    button and waits for two matching observations before reporting success.
    """

    def __init__(self, step: Callable[[str, int], np.ndarray],
                 classifier: GbaMenuSceneClassifier, *, max_wait_frames: int = 48,
                 max_operation_frames: int = 256):
        if max_wait_frames < 2 or max_operation_frames < 32:
            raise ValueError("menu frame limits are too small")
        self.step = step
        self.classifier = classifier
        self.max_wait_frames = int(max_wait_frames)
        self.max_operation_frames = int(max_operation_frames)
        self.input_frames = 0
        self.trace: list[dict] = []

    def _input(self, button: str, frames: int, *, release_frames: int = 1) -> np.ndarray:
        if frames < 1 or button not in {"NOOP", "UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START"}:
            raise ValueError("unsupported menu control or frame count")
        if not 0 <= release_frames <= 8:
            raise ValueError("release_frames must be between 0 and 8")
        release_cost = release_frames if button != "NOOP" else 0
        if self.input_frames + frames + release_cost > self.max_operation_frames:
            raise RuntimeError("menu operation exceeded its emulator-frame budget")
        frame = self.step(button, frames)
        self.input_frames += frames
        self.trace.append({"action": button, "held_frames": frames,
                           "scene": self.classifier.classify(frame)[0], "phase": "input"})
        if button != "NOOP":
            for _ in range(release_frames):
                frame = self.step("NOOP", 1)
                self.input_frames += 1
                self.trace.append({"action": "NOOP", "held_frames": 1,
                                   "scene": self.classifier.classify(frame)[0], "phase": "release"})
        return np.asarray(frame, dtype=np.uint8)

    def _wait(self, expected: str, frame: np.ndarray) -> tuple[np.ndarray, float]:
        streak = 0
        score = 0.0
        current = np.asarray(frame, dtype=np.uint8)
        for _ in range(self.max_wait_frames):
            scene, score = self.classifier.classify(current)
            streak = streak + 1 if scene == expected else 0
            self.trace.append({"action": "OBSERVE", "held_frames": 0,
                               "scene": scene, "score": round(score, 6), "phase": "confirm"})
            if streak >= 2:
                return current, score
            current = self._input("NOOP", 1)
        last_scene, last_score = self.classifier.classify(current)
        raise RuntimeError(f"expected two consecutive {expected} observations; last scene was "
                           f"{last_scene} (score={last_score:.5f})")

    def _open_main_menu(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        frame = self._input("START", 8)
        return self._wait("main_menu", frame)

    def _select_equipment_row(self, frame: np.ndarray) -> np.ndarray:
        # The first d-pad edge can be swallowed while the menu finishes opening.
        # Read the cursor anchor after each bounded tap and stop as soon as the
        # Equipment row is independently visible. Never assume a fixed tap count.
        for _ in range(3):
            scene, _ = self.classifier.classify(frame)
            if scene == "main_menu_equipment_cursor":
                return frame
            if scene not in {"main_menu", "unknown"}:
                raise RuntimeError(f"unexpected scene while selecting Equipment: {scene}")
            frame = self._input("DOWN", 1, release_frames=2)
            frame, _ = self._wait_any(("main_menu", "main_menu_equipment_cursor"), frame)
        scene, _ = self.classifier.classify(frame)
        if scene != "main_menu_equipment_cursor":
            raise RuntimeError("Equipment cursor was not visibly selected after three bounded taps")
        return frame

    def _wait_any(self, expected: tuple[str, ...], frame: np.ndarray) -> tuple[np.ndarray, float]:
        streak_scene = ""
        streak = 0
        score = 0.0
        current = np.asarray(frame, dtype=np.uint8)
        for _ in range(self.max_wait_frames):
            scene, score = self.classifier.classify(current)
            if scene in expected:
                streak = streak + 1 if scene == streak_scene else 1
                streak_scene = scene
                self.trace.append({"action": "OBSERVE", "held_frames": 0,
                                   "scene": scene, "score": round(score, 6), "phase": "confirm"})
                if streak >= 2:
                    return current, score
            else:
                streak = 0
                self.trace.append({"action": "OBSERVE", "held_frames": 0,
                                   "scene": scene, "score": round(score, 6), "phase": "confirm"})
            current = self._input("NOOP", 1)
        scene, score = self.classifier.classify(current)
        raise RuntimeError(f"expected one of {expected}; last scene was {scene} (score={score:.5f})")

    def inspect_inventory(self, frame: np.ndarray) -> MenuOperationResult:
        self.input_frames = 0
        self.trace = []
        frame, _ = self._open_main_menu(frame)
        frame = self._input("A", 16)  # fresh checkpoint defaults to Items
        frame, score = self._wait("empty_items", frame)
        return MenuOperationResult("inspect_inventory", "empty_inventory_confirmed",
                                   "empty_items", frame, self.input_frames, score)

    def inspect_sif_equipment(self, frame: np.ndarray) -> MenuOperationResult:
        self.input_frames = 0
        self.trace = []
        frame, _ = self._open_main_menu(frame)
        frame = self._select_equipment_row(frame)
        frame = self._input("A", 16)
        frame, _ = self._wait("equipment_character_select", frame)
        frame = self._input("A", 16)  # default character row is Sif
        frame, score = self._wait("sif_equipment", frame)
        return MenuOperationResult("inspect_sif_equipment", "read_only_equipment_confirmed",
                                   "sif_equipment", frame, self.input_frames, score)
