"""Conservative visual landmark checks for the GBA FF1 experiment.

The Castle Cornelia label is taken from the game's own static floor banner.
This intentionally identifies only the named 1F screen; it is not a general
scene classifier and assigns no semantics to RAM values or pixel motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


# The text banner is fixed at the top of 240x160 Stable-Retro frames. Keep a
# little padding around the glyphs so that an unrelated similar text fragment
# does not match. The reference is a visually reviewed gameplay frame.
TITLE_ROI = (55, 7, 185, 27)


@dataclass(frozen=True)
class LandmarkResult:
    label: str
    confidence: float
    status: str

    @property
    def reached(self) -> bool:
        return self.label == "castle_cornelia_1f" and self.status == "confirmed"


class CastleCorneliaDetector:
    """Match Castle Cornelia's in-game floor title, yielding explicit uncertainty.

    This compares a small text-bearing UI region with a reviewed reference;
    it does not use full-frame hashes, screen-change magnitude, or RAM.
    A non-match is reported as unknown, never as a different named scene.
    """

    def __init__(self, reference: str | Path | np.ndarray, *, threshold: float = 0.985):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if isinstance(reference, (str, Path)):
            with Image.open(reference) as image:
                ref = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            ref = np.asarray(reference, dtype=np.uint8)
        if ref.ndim != 3 or ref.shape[2] < 3 or ref.shape[0] < TITLE_ROI[3] or ref.shape[1] < TITLE_ROI[2]:
            raise ValueError(f"reference must be an RGB frame at least 185x27, got {ref.shape}")
        x0, y0, x1, y1 = TITLE_ROI
        self.reference = np.ascontiguousarray(ref[y0:y1, x0:x1, :3])
        self.threshold = threshold

    def classify(self, frame: np.ndarray) -> LandmarkResult:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"expected RGB frame, got {image.shape}")
        x0, y0, x1, y1 = TITLE_ROI
        if image.shape[0] < y1 or image.shape[1] < x1:
            return LandmarkResult("unknown", 0.0, "uncertain")
        candidate = image[y0:y1, x0:x1, :3]
        if candidate.shape != self.reference.shape:
            return LandmarkResult("unknown", 0.0, "uncertain")
        # Mean absolute RGB agreement tolerates tiny renderer noise, while the
        # high threshold requires the static title text and panel to agree.
        score = 1.0 - float(np.abs(candidate.astype(np.int16) - self.reference.astype(np.int16)).mean()) / 255.0
        if score >= self.threshold:
            return LandmarkResult("castle_cornelia_1f", score, "confirmed")
        if score >= self.threshold - 0.025:
            return LandmarkResult("unknown", score, "uncertain")
        return LandmarkResult("unknown", score, "no_match")
