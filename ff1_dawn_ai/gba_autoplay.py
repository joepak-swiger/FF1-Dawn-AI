"""Small deterministic controllers for bounded GBA autoplay demonstrations.

The baseline here is an explicit route script, not a trained policy. Game
scenes that lack an independently reviewed detector remain unknown; in
particular this module never infers a battle outcome from a screen change.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

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
        # This is a current sustained observation, not a sticky historical
        # fact. Callers must latch events separately and reset at lifecycle
        # boundaries; stale battle UI may never trigger a later encounter.
        self.confirmed = self._streak >= self.consecutive_frames
        return self.confirmed


def should_retry_attack_selection(*, attempt: int, attack_panel_confirmed: bool,
                                  target_panel_confirmed: bool,
                                  magic_panel_confirmed: bool) -> bool:
    """Permit one safe re-press only when the reviewed Attack panel persisted.

    This handles dropped/early A inputs without guessing from a transition or
    replaying an action after the target panel has appeared. Magic recovery has
    its own verified B path and therefore is never treated as an Attack retry.
    """
    return (attempt == 0 and attack_panel_confirmed
            and not target_panel_confirmed and not magic_panel_confirmed)


@dataclass
class AutonomousSessionCoordinator:
    """Small auditable lifecycle tracker for continuous autoplay sessions.

    Evidence is attached to distinct encounter IDs. Reward banners are
    idempotent per encounter, so EXP and gil messages cannot inflate victory
    counts. This class does not infer scenes or gameplay outcomes itself.
    """

    phase: str = "field_navigate"
    battles_entered: int = 0
    battles_finished: int = 0
    victories: int = 0
    escapes: int = 0
    defeats: int = 0
    unknown_outcomes: int = 0
    map_views: int = 0
    movement_resumed: int = 0
    current_battle_id: int | None = None
    encounters: list[dict] = field(default_factory=list)

    def battle_detected(self, step: int) -> int:
        if self.phase not in {"field_navigate", "field_recovery"}:
            raise RuntimeError(f"cannot start battle from phase {self.phase}")
        self.battles_entered += 1
        self.current_battle_id = self.battles_entered
        self.phase = "battle_commands"
        self.encounters.append({"battle_id": self.current_battle_id,
                                "entry_step": step, "outcome": "pending",
                                "exp_confirmed": False, "gil_confirmed": False,
                                "exp_amount": None, "gil_amount": None,
                                "level_ups": [],
                                "finished": False,
                                "map_view_observed": False, "movement_resumed": False})
        return self.current_battle_id

    def _current(self) -> dict:
        if self.current_battle_id is None:
            raise RuntimeError("no active battle")
        return self.encounters[self.current_battle_id - 1]

    def reward_confirmed(self, kind: str, amount: int | None = None) -> bool:
        if kind not in {"exp", "gil"}:
            raise ValueError(f"unsupported reward kind: {kind}")
        row = self._current()
        key = f"{kind}_confirmed"
        if row[key]:
            return False
        row[key] = True
        row[f"{kind}_amount"] = amount
        if kind == "exp" and row["outcome"] == "pending":
            row["outcome"] = "victory"
            self.victories += 1
        return True

    def finish_battle(self, outcome: str) -> None:
        if outcome not in {"victory", "escape", "defeat", "unknown"}:
            raise ValueError(f"invalid battle outcome: {outcome}")
        row = self._current()
        if row["outcome"] == "pending":
            row["outcome"] = outcome
            if outcome == "victory":
                self.victories += 1
            elif outcome == "escape":
                self.escapes += 1
            elif outcome == "defeat":
                self.defeats += 1
            else:
                self.unknown_outcomes += 1
        elif outcome != row["outcome"]:
            raise RuntimeError("encounter outcome is already latched")
        if row["outcome"] != "pending" and not row["finished"]:
            row["finished"] = True
            self.battles_finished += 1
            self.phase = "acknowledge"

    def level_up_confirmed(self, character: str) -> bool:
        row = self._current()
        if character in row["level_ups"]:
            return False
        row["level_ups"].append(character)
        return True

    def map_view_observed(self) -> None:
        row = self._current()
        if not row["map_view_observed"]:
            row["map_view_observed"] = True
            self.map_views += 1
        self.phase = "field_recovery"

    def movement_observed(self, *, confirmed: bool) -> None:
        if not confirmed:
            return
        row = self._current()
        if not row["movement_resumed"]:
            row["movement_resumed"] = True
            self.movement_resumed += 1
        self.phase = "field_navigate"

    def new_encounter(self, step: int) -> int:
        return self.battle_detected(step)


def compare_matched_movement_frames(
    idle_a: np.ndarray, idle_b: np.ndarray,
    forward_a: np.ndarray, forward_b: np.ndarray,
    opposite_a: np.ndarray, opposite_b: np.ndarray,
    *, min_control_difference: float = 1.5,
) -> dict[str, float | bool]:
    """Conservatively compare repeated cardinal probes with matched idle.

    Pixel differences are screen-space evidence, never tile coordinates. A
    positive requires deterministic repeats, a stable idle control, opposite
    controls to differ, and at least one direction to differ materially from
    idle. The direction that actually differs is returned for the resume
    action, so a blocked direction cannot hide an independently observed move.
    Screens should only be sampled after a separately confirmed playable map.
    """
    frames = [np.asarray(frame, dtype=np.uint8) for frame in
              (idle_a, idle_b, forward_a, forward_b, opposite_a, opposite_b)]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("matched movement frames must have identical shapes")

    def mad(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())

    idle_noise = mad(frames[0], frames[1])
    forward_repeat = mad(frames[2], frames[3])
    opposite_repeat = mad(frames[4], frames[5])
    forward_vs_idle = mad(frames[2], frames[0])
    opposite_vs_idle = mad(frames[4], frames[0])
    forward_vs_opposite = mad(frames[2], frames[4])
    stable = idle_noise <= 0.10 and forward_repeat <= 0.10 and opposite_repeat <= 0.10
    forward_moves = forward_vs_idle >= min_control_difference
    opposite_moves = opposite_vs_idle >= min_control_difference
    controls_distinguishable = forward_vs_opposite >= min_control_difference
    confirmed = bool(stable and controls_distinguishable and (forward_moves or opposite_moves))
    movement_action = ("forward" if forward_moves else "opposite") if confirmed else "none"
    return {
        "idle_repeat_mae": idle_noise,
        "forward_repeat_mae": forward_repeat,
        "opposite_repeat_mae": opposite_repeat,
        "forward_vs_idle_mae": forward_vs_idle,
        "opposite_vs_idle_mae": opposite_vs_idle,
        "forward_vs_opposite_mae": forward_vs_opposite,
        "matched_controls_stable": stable,
        "movement_signal_confirmed": confirmed,
        "movement_signal_action": movement_action,
    }


class BattleTargetDetector(BattleCommandDetector):
    """Match the reviewed Goblin target list, distinct from Magic and Attack.

    Evidence includes multiple counts of the same Goblin species. This anchor
    does not establish support for other enemy formations or target layouts.
    """

    def __init__(self, reference: str, *, alternative_references: Sequence[str] = (),
                 threshold: float = 0.95):
        super().__init__(reference, threshold=threshold)
        self.alternative_references = []
        for path in alternative_references:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if pixels.shape != self.reference.shape:
                raise ValueError(
                    f"alternate Goblin target anchor must match {self.reference.shape}, got {pixels.shape}"
                )
            self.alternative_references.append(pixels)

    def score(self, frame: np.ndarray) -> float:
        image = np.asarray(frame, dtype=np.uint8)
        x0, y0, x1, y1 = BATTLE_COMMAND_ROI
        if image.ndim != 3 or image.shape[0] < y1 or image.shape[1] < x1 or image.shape[2] < 3:
            return 0.0
        crop = image[y0:y1, x0:x1, :3].astype(np.int16)
        references = [self.reference, *self.alternative_references]
        return max(1.0 - float(np.abs(crop - reference.astype(np.int16)).mean()) / 255.0
                   for reference in references)


class _BattleAmountBannerDetector:
    """Recognize a fixed result phrase and exact observed numeric glyphs.

    Phrase confirmation masks the amount pixels, while amount recognition
    compares only the reviewed digit box against observed-value templates.
    New numeric values remain unparsed until a real template is reviewed.
    """

    def __init__(self, references: dict[int, str], *, amount_roi: tuple[int, int, int, int],
                 threshold: float = 0.985, amount_threshold: float = 0.995):
        if not references or not 0.0 < threshold <= 1.0 or not 0.0 < amount_threshold <= 1.0:
            raise ValueError("banner templates and thresholds must be valid")
        self.references = {}
        for amount, path in references.items():
            with Image.open(path) as image:
                banner = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if banner.shape != (16, 240, 3):
                raise ValueError(f"result banner must be 240x16 RGB, got {banner.shape}")
            self.references[int(amount)] = banner
        self.amount_roi = amount_roi
        self.threshold = float(threshold)
        self.amount_threshold = float(amount_threshold)
        self._phrase_mask = np.ones((16, 240), dtype=bool)
        x0, y0, x1, y1 = amount_roi
        self._phrase_mask[y0:y1, x0:x1] = False

    def _banner(self, frame: np.ndarray) -> np.ndarray | None:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3 or image.shape[0] < 16 or image.shape[1] < 240 or image.shape[2] < 3:
            return None
        return image[:16, :240, :3]

    def score(self, frame: np.ndarray) -> float:
        banner = self._banner(frame)
        if banner is None:
            return 0.0
        return max(1.0 - float(np.abs(banner.astype(np.int16) - reference.astype(np.int16))
                                [self._phrase_mask].mean()) / 255.0
                   for reference in self.references.values())

    def is_candidate(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold

    def amount_scores(self, frame: np.ndarray) -> dict[int, float]:
        banner = self._banner(frame)
        if banner is None or not self.is_candidate(frame):
            return {}
        x0, y0, x1, y1 = self.amount_roi
        crop = banner[y0:y1, x0:x1]
        return {amount: 1.0 - float(np.abs(crop.astype(np.int16) - ref[y0:y1, x0:x1].astype(np.int16)).mean()) / 255.0
                for amount, ref in self.references.items()}

    def recognized_amount(self, frame: np.ndarray) -> int | None:
        scores = self.amount_scores(frame)
        if not scores:
            return None
        amount, score = max(scores.items(), key=lambda item: item[1])
        return amount if score >= self.amount_threshold else None


class BattleExpResultDetector(_BattleAmountBannerDetector):
    """Recognize observed ``Earned [amount] EXP.`` results.

    Only the real 4-EXP and 6-EXP values have numeric templates. The fixed
    message can confirm a victory while an unseen amount remains ``None``.
    """

    def __init__(self, reference: str, *, alternative_references: dict[int, str] | None = None,
                 threshold: float = 0.985, amount_threshold: float = 0.995):
        refs = {6: reference, **(alternative_references or {})}
        super().__init__(refs, amount_roi=(122, 0, 132, 16), threshold=threshold,
                         amount_threshold=amount_threshold)


class BattleGilResultDetector(_BattleAmountBannerDetector):
    """Recognize observed ``Obtained [amount] gil.`` result messages."""

    def __init__(self, reference: str, *, alternative_references: dict[int, str] | None = None,
                 threshold: float = 0.985, amount_threshold: float = 0.995):
        refs = {24: reference, **(alternative_references or {})}
        super().__init__(refs, amount_roi=(128, 0, 142, 16), threshold=threshold,
                         amount_threshold=amount_threshold)


class BattleLevelUpDetector:
    """Match the exact observed ``Sif gained a level!`` top banner."""

    def __init__(self, reference: str, *, threshold: float = 0.999):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        with Image.open(reference) as image:
            self.reference = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if self.reference.shape != (16, 240, 3):
            raise ValueError(f"level-up banner reference must be 240x16 RGB, got {self.reference.shape}")
        self.threshold = float(threshold)

    def score(self, frame: np.ndarray) -> float:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3 or image.shape[0] < 16 or image.shape[1] < 240 or image.shape[2] < 3:
            return 0.0
        crop = image[:16, :240, :3]
        return 1.0 - float(np.abs(crop.astype(np.int16) - self.reference.astype(np.int16)).mean()) / 255.0

    def is_candidate(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold


class BattleHpIncreaseDetector(BattleLevelUpDetector):
    """Recognize the one reviewed ``HP increased by 27.`` result banner.

    This is an exact local evidence template, not a general HP parser. Other
    values remain unknown until their screenshots are independently reviewed.
    """

    def __init__(self, reference: str, *, threshold: float = 0.999):
        super().__init__(reference, threshold=threshold)


class BattleStrengthIncreaseDetector(BattleLevelUpDetector):
    """Recognize the exact reviewed ``Strength increased.`` banner."""

    def __init__(self, reference: str, *, threshold: float = 0.999):
        super().__init__(reference, threshold=threshold)


class BattlePostVictoryMessageDetector:
    """Recognize a message overlay over the battle HUD after proven victory.

    This intentionally does not read or classify the message text. Callers may
    use it only after an exact EXP victory and a reviewed level-up banner have
    started the result-message sequence. Two-frame confirmation and a strict
    acknowledgement cap are required at the call site.
    """

    TOP_BANNER_ROI = (0, 0, 240, 16)
    BATTLE_PANEL_ROI = (0, 120, 100, 158)
    threshold = 0.5

    def score(self, frame: np.ndarray) -> float:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3 or image.shape[0] < 158 or image.shape[1] < 240 or image.shape[2] < 3:
            return 0.0
        matched = (CorneliaPostBattleMapDetector._has_blue_panel(image, self.TOP_BANNER_ROI)
                   and CorneliaPostBattleMapDetector._has_blue_panel(image, self.BATTLE_PANEL_ROI))
        return 1.0 if matched else 0.0

    def is_candidate(self, frame: np.ndarray) -> bool:
        return self.score(frame) >= self.threshold

    @staticmethod
    def message_signature(frame: np.ndarray) -> str:
        """Hash a white-glyph mask, ignoring animated blue panel colors."""
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3 or image.shape[0] < 14 or image.shape[1] < 205 or image.shape[2] < 3:
            return ""
        text = image[2:14, 35:205, :3].astype(np.uint16)
        glyph_mask = np.min(text, axis=2) >= 185
        return hashlib.sha256(glyph_mask.tobytes()).hexdigest()


class CorneliaPostBattleMapDetector:
    """Match reviewed Cornelia-area world-map compositions after rewards.

    Anchors are independently reviewed post-result views. A small bounded
    translation search tolerates camera settling between observations; it is
    not a general field classifier, coordinate estimate, or movement signal.
    """

    ROI = (20, 24, 180, 72)
    # Cornelia exterior anchors have no battle command panel in the lower-left
    # HUD region. The postbattle result messages have a full-width blue banner
    # at the top. These empirical UI guards prevent the field-colored battle
    # background from being mistaken for a playable overworld.
    TOP_BANNER_ROI = (0, 0, 240, 16)
    BATTLE_PANEL_ROI = (0, 120, 100, 158)

    def __init__(self, reference: str | Sequence[str], *, threshold: float = 0.995,
                 camera_tolerance: tuple[int, int] = (4, 4)):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if (len(camera_tolerance) != 2 or any(not isinstance(value, int) or value < 0
                                               for value in camera_tolerance)):
            raise ValueError("camera_tolerance must be two non-negative integer pixel bounds")
        references = (reference,) if isinstance(reference, (str, Path)) else tuple(reference)
        if not references:
            raise ValueError("at least one Cornelia map anchor is required")
        self.references = []
        for path in references:
            with Image.open(path) as image:
                anchor = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if anchor.shape != (48, 160, 3):
                raise ValueError(f"Cornelia map anchor must be 160x48 RGB, got {anchor.shape}")
            self.references.append(anchor)
        self.threshold = float(threshold)
        self.camera_tolerance = tuple(camera_tolerance)

    @staticmethod
    def _similarity(crop: np.ndarray, reference: np.ndarray) -> float:
        return 1.0 - float(np.abs(crop.astype(np.int16) - reference.astype(np.int16)).mean()) / 255.0

    @staticmethod
    def _has_blue_panel(image: np.ndarray, roi: tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = roi
        crop = image[y0:y1, x0:x1, :3].astype(np.int16)
        blue = ((crop[:, :, 2] > crop[:, :, 0] * 1.3)
                & (crop[:, :, 2] > crop[:, :, 1] * 1.1)
                & (crop[:, :, 2] > 45))
        if float(blue.mean()) <= 0.35:
            return False
        # Cornelia's ocean is bright cyan-blue; the reviewed GBA text panels
        # are much darker and have a lower blue-channel mean.
        return float(crop[:, :, 2][blue].mean()) < 220.0

    def score(self, frame: np.ndarray) -> float:
        image = np.asarray(frame, dtype=np.uint8)
        x0, y0, x1, y1 = self.ROI
        if (image.ndim != 3 or image.shape[0] < y1 or image.shape[1] < x1
                or image.shape[2] < 3 or image.shape[0] < self.BATTLE_PANEL_ROI[3]
                or image.shape[1] < self.TOP_BANNER_ROI[2]):
            return 0.0
        # A map-texture crop alone matches the outdoor backdrop behind battle
        # result UI. Require the top message banner and lower-left battle panel
        # to be absent before even considering the reviewed map anchors.
        if (self._has_blue_panel(image, self.TOP_BANNER_ROI)
                or self._has_blue_panel(image, self.BATTLE_PANEL_ROI)):
            return 0.0
        tolerance_x, tolerance_y = self.camera_tolerance
        # Coarse 2-pixel offsets keep this narrow post-result check cheap while
        # allowing small camera settling. The large crop and high threshold
        # still require a close match to a reviewed Cornelia composition.
        offsets_x = range(-tolerance_x, tolerance_x + 1, 2)
        offsets_y = range(-tolerance_y, tolerance_y + 1, 2)
        best = 0.0
        for reference in self.references:
            for dy in offsets_y:
                sy0, sy1 = y0 + dy, y1 + dy
                if sy0 < 0 or sy1 > image.shape[0]:
                    continue
                for dx in offsets_x:
                    sx0, sx1 = x0 + dx, x1 + dx
                    if sx0 < 0 or sx1 > image.shape[1]:
                        continue
                    best = max(best, self._similarity(image[sy0:sy1, sx0:sx1, :3], reference))
                    if best >= 0.9999:
                        return best
        return best

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
