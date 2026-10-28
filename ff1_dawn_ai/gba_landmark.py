"""Conservative scene anchors and temporal confirmation for GBA FF1.

The visual anchors are reviewed example screens. They are intended for
bounded Cornelia experiments, not as a general-purpose game-state oracle.
Raw single-frame matches are candidates only; stateful confirmation requires
consecutive consistent observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


TITLE_ROI = (55, 7, 185, 27)
SCENE_LABELS = ("cornelia_exterior_field", "cornelia_town", "castle_cornelia_1f", "menu")


@dataclass(frozen=True)
class LandmarkResult:
    label: str
    confidence: float
    status: str

    @property
    def matched(self) -> bool:
        """Whether a single frame matched an anchor (not temporal confirmation)."""
        return self.status == "candidate"

def _read_rgb(reference: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(reference, (str, Path)):
        with Image.open(reference) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        frame = np.asarray(reference, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"expected RGB frame, got {frame.shape}")
    return np.ascontiguousarray(frame[:, :, :3])


class SceneAnchorClassifier:
    """Recognize four reviewed Cornelia screen anchors and preserve uncertainty.

    Castle and town use the game's title banner. The exterior uses separate
    fixed map details sampled from the visually reviewed outdoor frame. The
    menu uses multiple panel crops from the rendered party/status menu. A
    screen that misses these specific anchors remains ``unknown``; it is never
    assigned to field simply because the Castle title is absent.
    """

    def __init__(self, references: Mapping[str, str | Path | np.ndarray], *,
                 title_threshold: float = 0.985, town_title_threshold: float = 0.94,
                 field_reference_variants: tuple[str | Path | np.ndarray, ...] = (),
                 field_threshold: float = 0.94,
                 menu_threshold: float = 0.94):
        required = set(SCENE_LABELS)
        if set(references) != required:
            raise ValueError(f"references must contain exactly {sorted(required)}")
        for threshold in (title_threshold, town_title_threshold, field_threshold, menu_threshold):
            if not 0.0 < threshold <= 1.0:
                raise ValueError("all thresholds must be in (0, 1]")
        self.references = {key: _read_rgb(value) for key, value in references.items()}
        self._field_references = [self.references["cornelia_exterior_field"]]
        self._field_references.extend(_read_rgb(value) for value in field_reference_variants)
        for label, frame in self.references.items():
            if frame.shape[0] < TITLE_ROI[3] or frame.shape[1] < TITLE_ROI[2]:
                raise ValueError(f"{label} reference is too small: {frame.shape}")
        self.title_threshold = title_threshold
        self.town_title_threshold = town_title_threshold
        self.field_threshold = field_threshold
        self.menu_threshold = menu_threshold
        x0, y0, x1, y1 = TITLE_ROI
        self._title = {label: frame[y0:y1, x0:x1].copy()
                       for label, frame in self.references.items()
                       if label in ("castle_cornelia_1f", "cornelia_town")}
        # Three map landmarks avoid the player sprite and use distinct parts of
        # the exterior: forest canopy, central stone approach, and shoreline.
        self._field_rois = ((8, 5, 55, 34), (93, 88, 142, 119), (190, 125, 235, 154))
        self._menu_rois = ((0, 0, 166, 45), (174, 4, 236, 79), (174, 82, 236, 154))

    @staticmethod
    def _similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return 0.0
        return 1.0 - float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) / 255.0

    def _crop_score(self, frame: np.ndarray, reference: np.ndarray,
                    rois: tuple[tuple[int, int, int, int], ...]) -> float:
        values = []
        for x0, y0, x1, y1 in rois:
            if frame.shape[0] < y1 or frame.shape[1] < x1:
                return 0.0
            values.append(self._similarity(frame[y0:y1, x0:x1], reference[y0:y1, x0:x1]))
        return float(min(values))

    def _field_score(self, frame: np.ndarray) -> float:
        """Match three outdoor landmarks with a shared vertical camera shift.

        Prior live exits showed the same Cornelia map scrolling vertically by
        one pixel per emulator frame during the transition. The bounded
        translation search is limited to those map templates and is not used
        to infer position or coordinates.
        """
        best = 0.0
        for reference in self._field_references:
            for dy in range(-32, 33):
                scores = []
                for x0, y0, x1, y1 in self._field_rois:
                    sy0, sy1 = y0 + dy, y1 + dy
                    if sy0 < 0 or sy1 > frame.shape[0] or x1 > frame.shape[1]:
                        scores = []
                        break
                    scores.append(self._similarity(frame[sy0:sy1, x0:x1],
                                                   reference[y0:y1, x0:x1]))
                if len(scores) == len(self._field_rois):
                    best = max(best, min(scores))
                    if best >= 0.999:
                        return best
        return best

    def classify(self, frame: np.ndarray) -> LandmarkResult:
        image = _read_rgb(frame)
        if image.shape[0] < TITLE_ROI[3] or image.shape[1] < TITLE_ROI[2]:
            return LandmarkResult("unknown", 0.0, "uncertain")
        x0, y0, x1, y1 = TITLE_ROI
        title = image[y0:y1, x0:x1]
        title_scores = {label: self._similarity(title, self._title[label])
                        for label in self._title}
        title_label = max(title_scores, key=title_scores.get)
        title_score = title_scores[title_label]
        title_threshold = self.town_title_threshold if title_label == "cornelia_town" else self.title_threshold
        if title_score >= title_threshold:
            return LandmarkResult(title_label, title_score, "candidate")

        field_score = self._field_score(image)
        if field_score >= self.field_threshold:
            return LandmarkResult("cornelia_exterior_field", field_score, "candidate")

        menu = self.references["menu"]
        menu_score = self._crop_score(image, menu, self._menu_rois)
        if menu_score >= self.menu_threshold:
            return LandmarkResult("menu", menu_score, "candidate")

        # Near-uniform very dark/bright frames are common during fades. Keep
        # these explicitly uncertain rather than treating them as a scene.
        brightness = float(image.mean())
        spread = float(image.std())
        if brightness < 18.0 or (brightness > 245.0 and spread < 8.0):
            return LandmarkResult("unknown", max(0.0, 1.0 - spread / 255.0), "uncertain_transition")
        best_score = max(title_score, field_score, menu_score)
        if best_score >= min(self.title_threshold, self.town_title_threshold,
                             self.field_threshold, self.menu_threshold) - 0.04:
            return LandmarkResult("unknown", best_score, "uncertain")
        return LandmarkResult("unknown", best_score, "no_match")


class CastleCorneliaDetector:
    """Backwards-compatible Castle-only raw matcher; one frame is a candidate."""

    def __init__(self, reference: str | Path | np.ndarray, *, threshold: float = 0.985):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        ref = _read_rgb(reference)
        if ref.shape[0] < TITLE_ROI[3] or ref.shape[1] < TITLE_ROI[2]:
            raise ValueError(f"reference must be at least 185x27, got {ref.shape}")
        x0, y0, x1, y1 = TITLE_ROI
        self.reference = ref[y0:y1, x0:x1]
        self.threshold = threshold

    def classify(self, frame: np.ndarray) -> LandmarkResult:
        image = _read_rgb(frame)
        x0, y0, x1, y1 = TITLE_ROI
        if image.shape[0] < y1 or image.shape[1] < x1:
            return LandmarkResult("unknown", 0.0, "uncertain")
        score = SceneAnchorClassifier._similarity(image[y0:y1, x0:x1], self.reference)
        if score >= self.threshold:
            return LandmarkResult("castle_cornelia_1f", score, "candidate")
        if score >= self.threshold - 0.025:
            return LandmarkResult("unknown", score, "uncertain")
        return LandmarkResult("unknown", score, "no_match")


@dataclass(frozen=True)
class SceneUpdate:
    candidate: LandmarkResult
    confirmed_scene: str | None
    transitioned: bool
    first_castle_entry: bool
    confirmation_count: int


class SceneConfirmationGate:
    """Two-consecutive-frame scene gate with one first-entry event per visit.

    Unknown/candidate mismatches clear the consecutive streak. The current
    confirmed scene remains stable through short fades. A Castle entry is
    emitted only when the preceding confirmed scene was the exterior field.
    Reset the gate at every environment reset; loading an interior checkpoint
    therefore cannot emit a false field-to-Castle entry event.
    """

    def __init__(self, consecutive_frames: int = 2):
        if consecutive_frames < 2:
            raise ValueError("consecutive_frames must be at least 2")
        self.consecutive_frames = int(consecutive_frames)
        self.reset()

    def reset(self) -> None:
        self._candidate_label: str | None = None
        self._streak = 0
        self.confirmed_scene: str | None = None

    def update(self, result: LandmarkResult) -> SceneUpdate:
        if result.status != "candidate" or result.label not in SCENE_LABELS:
            self._candidate_label = None
            self._streak = 0
            return SceneUpdate(result, self.confirmed_scene, False, False, 0)
        if result.label == self._candidate_label:
            self._streak += 1
        else:
            self._candidate_label = result.label
            self._streak = 1
        transitioned = False
        castle_entry = False
        if self._streak >= self.consecutive_frames and result.label != self.confirmed_scene:
            previous = self.confirmed_scene
            self.confirmed_scene = result.label
            transitioned = True
            castle_entry = previous == "cornelia_exterior_field" and result.label == "castle_cornelia_1f"
        return SceneUpdate(result, self.confirmed_scene, transitioned, castle_entry, self._streak)
