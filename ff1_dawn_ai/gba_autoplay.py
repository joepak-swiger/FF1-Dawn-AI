"""Small deterministic controllers for bounded GBA autoplay demonstrations.

The baseline here is an explicit route script, not a trained policy. Game
scenes that lack an independently reviewed detector remain unknown; in
particular this module never infers a battle outcome from a screen change.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image


BATTLE_COMMAND_ROI = (0, 108, 100, 158)


class BattleCommandDetector:
    """Match the left battle menu with the cursor specifically on Attack.

    The left panel distinguishes Attack selection from a Magic submenu and
    the Goblin target panel. Evidence currently covers three fresh Goblin
    encounters, not every formation or class/menu configuration.
    """

    def __init__(self, reference: str, *, threshold: float = 0.90):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        with Image.open(reference) as image:
            self.reference = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if self.reference.shape != (50, 100, 3):
            raise ValueError(f"battle left-panel anchor must be 100x50 RGB, got {self.reference.shape}")
        self.threshold = float(threshold)

    def score(self, frame: np.ndarray) -> float:
        image = np.asarray(frame, dtype=np.uint8)
        x0, y0, x1, y1 = BATTLE_COMMAND_ROI
        if image.ndim != 3 or image.shape[0] < y1 or image.shape[1] < x1 or image.shape[2] < 3:
            return 0.0
        crop = image[y0:y1, x0:x1, :3]
        return 1.0 - float(np.abs(crop.astype(np.int16) - self.reference.astype(np.int16)).mean()) / 255.0

    def is_candidate(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold


class BattleConfirmationGate:
    """Require consecutive observations of the battle command panel."""

    def __init__(self, detector: BattleCommandDetector, consecutive_frames: int = 2):
        if consecutive_frames < 2:
            raise ValueError("battle confirmation needs at least two observations")
        self.detector = detector
        self.consecutive_frames = int(consecutive_frames)
        self.reset()

    def reset(self) -> None:
        self._streak = 0
        self.confirmed = False
        self.last_score = 0.0

    def update(self, frame: np.ndarray) -> bool:
        self.last_score = self.detector.score(frame)
        self._streak = self._streak + 1 if self.last_score >= self.detector.threshold else 0
        if self._streak >= self.consecutive_frames:
            self.confirmed = True
        return self.confirmed


class BattleTargetDetector(BattleCommandDetector):
    """Match the reviewed Goblin target list, distinct from Magic and Attack."""


class BattleExpResultDetector:
    """Match the exact reviewed ``Earned 6 EXP.`` result banner.

    This is narrow evidence from one real Goblin encounter. It verifies that
    the game displayed this specific experience award; it is not a general
    battle-state or variable-EXP parser.
    """

    def __init__(self, reference: str, *, threshold: float = 0.985):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        with Image.open(reference) as image:
            self.reference = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if self.reference.shape != (16, 240, 3):
            raise ValueError(f"EXP banner reference must be 240x16 RGB, got {self.reference.shape}")
        self.threshold = float(threshold)

    def score(self, frame: np.ndarray) -> float:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3 or image.shape[0] < 16 or image.shape[1] < 240 or image.shape[2] < 3:
            return 0.0
        crop = image[:16, :240, :3]
        return 1.0 - float(np.abs(crop.astype(np.int16) - self.reference.astype(np.int16)).mean()) / 255.0

    def is_candidate(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold


class BattleGilResultDetector(BattleExpResultDetector):
    """Match the exact reviewed ``Obtained 24 gil.`` follow-up banner."""


class CorneliaPostBattleMapDetector:
    """Match one reviewed Cornelia-area world-map composition after rewards.

    The anchor is a single camera view for a return check. It is not a general
    field classifier, coordinate estimate, or movement/progress signal.
    """

    ROI = (20, 24, 180, 72)

    def __init__(self, reference: str, *, threshold: float = 0.995):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        with Image.open(reference) as image:
            self.reference = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if self.reference.shape != (48, 160, 3):
            raise ValueError(f"Cornelia map anchor must be 160x48 RGB, got {self.reference.shape}")
        self.threshold = float(threshold)

    def score(self, frame: np.ndarray) -> float:
        image = np.asarray(frame, dtype=np.uint8)
        x0, y0, x1, y1 = self.ROI
        if image.ndim != 3 or image.shape[0] < y1 or image.shape[1] < x1 or image.shape[2] < 3:
            return 0.0
        crop = image[y0:y1, x0:x1, :3]
        return 1.0 - float(np.abs(crop.astype(np.int16) - self.reference.astype(np.int16)).mean()) / 255.0

    def is_candidate(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold


@dataclass
class ScriptedCastleExcursion:
    """Repeat the previously verified north-entry / south-exit route.

    The controller uses the reviewed Castle floor-title and exterior anchors
    only to decide when to reverse direction. Route macros are intentional and
    should be reported as scripted behavior.
    """

    max_approach_actions: int = 160
    max_return_actions: int = 112
    phase: str = "approach_castle"
    approach_actions: int = 0
    return_actions: int = 0
    castle_reached: bool = False
    field_return_confirmed: bool = False
    stop_reason: str | None = None

    def observe(self, confirmed_scene: str | None) -> None:
        if confirmed_scene == "castle_cornelia_1f":
            self.castle_reached = True
            if self.phase == "approach_castle":
                self.phase = "return_to_field"
        elif confirmed_scene == "cornelia_exterior_field" and self.phase == "return_to_field":
            self.field_return_confirmed = True
            self.phase = "complete"

    def next_action(self) -> str | None:
        if self.phase == "complete" or self.stop_reason:
            return None
        if self.phase == "approach_castle":
            if self.approach_actions >= self.max_approach_actions:
                self.stop_reason = "castle_entry_timeout"
                return None
            self.approach_actions += 1
            return "UP"
        if self.phase == "return_to_field":
            if self.return_actions >= self.max_return_actions:
                self.stop_reason = "field_return_timeout"
                return None
            self.return_actions += 1
            return "DOWN"
        self.stop_reason = "invalid_controller_phase"
        return None


class SeededRandomExplorer:
    """Reproducible cardinal movement baseline; does not issue confirm keys."""

    ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._random = random.Random(self.seed)

    def next_action(self) -> str:
        return self._random.choice(self.ACTIONS)


def conservative_battle_outcome(*, independently_confirmed_result: str | None = None) -> str:
    """Return an outcome only when supplied by independent verified evidence.

    No such GBA battle/result anchor has been reviewed yet. Callers should
    therefore omit the argument and preserve ``unknown`` rather than guessing
    that leaving an unclassified screen means victory, escape, or defeat.
    """

    allowed = {"victory", "escape", "defeat"}
    if independently_confirmed_result in allowed:
        return str(independently_confirmed_result)
    return "unknown"
