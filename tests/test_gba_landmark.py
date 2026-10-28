import unittest

import numpy as np
from PIL import Image

from ff1_dawn_ai.gba_landmark import CastleCorneliaDetector


class CastleLandmarkTests(unittest.TestCase):
    def setUp(self):
        # Deterministic synthetic frame with a unique, high-contrast title
        # patch. Tests validate scoring and uncertainty without the emulator.
        self.reference = np.zeros((160, 240, 3), dtype=np.uint8)
        self.reference[:, :] = (20, 35, 70)
        self.reference[7:27, 55:185] = (10, 25, 90)
        self.reference[11:22, 73:166] = (230, 220, 175)
        self.detector = CastleCorneliaDetector(self.reference)

    def test_exact_reviewed_banner_is_confirmed(self):
        result = self.detector.classify(self.reference.copy())
        self.assertTrue(result.reached)
        self.assertEqual(result.label, "castle_cornelia_1f")

    def test_animation_outside_banner_does_not_change_landmark(self):
        frame = self.reference.copy()
        frame[70:90, 100:130] = (255, 0, 0)
        self.assertTrue(self.detector.classify(frame).reached)

    def test_other_menu_title_is_unknown(self):
        frame = self.reference.copy()
        frame[11:22, 73:166] = (180, 200, 240)
        result = self.detector.classify(frame)
        self.assertFalse(result.reached)
        self.assertEqual(result.label, "unknown")

    def test_bad_frame_shape_is_uncertain(self):
        result = self.detector.classify(np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(result.status, "uncertain")

    def test_threshold_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            CastleCorneliaDetector(self.reference, threshold=0)


if __name__ == "__main__":
    unittest.main()
