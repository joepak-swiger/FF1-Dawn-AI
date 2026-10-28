import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ff1_dawn_ai.gba_landmark import SceneAnchorClassifier


ROOT = Path(__file__).resolve().parents[1]
CASTLE_EVIDENCE = ROOT / "docs/experiments/gba_castle_goal/20261028_052724_624774Z"
TITLE_SCREEN = ROOT / "docs/experiments/gba_scene_classifier/title_screen_negative_control.png"


def frame(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


class ReviewedSceneAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        refs = {
            "castle_cornelia_1f": ROOT / "docs/experiments/gba_castle_goal/reference_cornelia_1f.png",
            "cornelia_exterior_field": CASTLE_EVIDENCE / "overworld_after_castle_exit.png",
            "cornelia_town": CASTLE_EVIDENCE / "town_ram_collision.png",
            "menu": CASTLE_EVIDENCE / "party_menu_negative_control.png",
        }
        cls.classifier = SceneAnchorClassifier(
            refs,
            field_reference_variants=(ROOT / "docs/experiments/gba_scene_classifier/cornelia_exterior_checkpoint_anchor.png",),
        )

    def test_independently_reviewed_anchor_screens(self):
        samples = (
            (CASTLE_EVIDENCE / "castle_entry_confirmed.png", "castle_cornelia_1f"),
            (CASTLE_EVIDENCE / "town_ram_collision.png", "cornelia_town"),
            (CASTLE_EVIDENCE / "overworld_after_castle_exit.png", "cornelia_exterior_field"),
            (ROOT / "docs/experiments/gba_scene_classifier/cornelia_exterior_checkpoint_anchor.png",
             "cornelia_exterior_field"),
            (CASTLE_EVIDENCE / "party_menu_negative_control.png", "menu"),
        )
        for path, expected in samples:
            with self.subTest(path=path.name):
                self.assertEqual(self.classifier.classify(frame(path)).label, expected)

    def test_transition_menu_and_title_selection_are_not_scene_positives(self):
        negatives = (
            CASTLE_EVIDENCE / "castle_before_transition.png",
            TITLE_SCREEN,
        )
        for path in negatives:
            with self.subTest(path=path.name):
                self.assertEqual(self.classifier.classify(frame(path)).label, "unknown")

    def test_actual_dark_fade_stays_uncertain(self):
        fade = np.zeros((160, 240, 3), dtype=np.uint8)
        result = self.classifier.classify(fade)
        self.assertEqual(result.label, "unknown")
        self.assertEqual(result.status, "uncertain_transition")

    def test_exterior_anchor_tolerates_small_vertical_camera_scroll(self):
        original = frame(ROOT / "docs/experiments/gba_scene_classifier/cornelia_exterior_checkpoint_anchor.png")
        shifted = np.zeros_like(original)
        shifted[3:, :, :] = original[:-3, :, :]
        result = self.classifier.classify(shifted)
        self.assertEqual(result.label, "cornelia_exterior_field")
        self.assertTrue(result.matched)


if __name__ == "__main__":
    unittest.main()
