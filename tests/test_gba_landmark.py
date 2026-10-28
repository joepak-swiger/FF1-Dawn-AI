import unittest

import numpy as np
from PIL import Image

from ff1_dawn_ai.gba_landmark import (
    CastleCorneliaDetector,
    LandmarkResult,
    SceneConfirmationGate,
)


class CastleLandmarkTests(unittest.TestCase):
    def setUp(self):
        # Deterministic synthetic frame with a unique, high-contrast title
        # patch. Tests validate scoring and uncertainty without the emulator.
        self.reference = np.zeros((160, 240, 3), dtype=np.uint8)
        self.reference[:, :] = (20, 35, 70)
        self.reference[7:27, 55:185] = (10, 25, 90)
        self.reference[11:22, 73:166] = (230, 220, 175)
        self.detector = CastleCorneliaDetector(self.reference)

    def test_single_matching_frame_is_only_a_candidate(self):
        result = self.detector.classify(self.reference.copy())
        self.assertTrue(result.matched)
        self.assertFalse(hasattr(result, "reached"))
        self.assertEqual(result.status, "candidate")
        self.assertEqual(result.label, "castle_cornelia_1f")

    def test_animation_outside_banner_does_not_change_landmark(self):
        frame = self.reference.copy()
        frame[70:90, 100:130] = (255, 0, 0)
        self.assertTrue(self.detector.classify(frame).matched)

    def test_other_menu_title_is_unknown(self):
        frame = self.reference.copy()
        frame[11:22, 73:166] = (180, 200, 240)
        result = self.detector.classify(frame)
        self.assertFalse(hasattr(result, "reached"))
        self.assertEqual(result.label, "unknown")

    def test_bad_frame_shape_is_uncertain(self):
        result = self.detector.classify(np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(result.status, "uncertain")

    def test_threshold_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            CastleCorneliaDetector(self.reference, threshold=0)

    def test_gate_confirms_two_consecutive_frames_and_emits_once(self):
        gate = SceneConfirmationGate()
        result = self.detector.classify(self.reference.copy())
        self.assertFalse(gate.update(result).first_castle_entry)
        update = gate.update(result)
        self.assertTrue(update.transitioned)
        self.assertEqual(update.confirmed_scene, "castle_cornelia_1f")
        self.assertFalse(update.first_castle_entry)  # no exterior was confirmed first
        self.assertFalse(gate.update(result).transitioned)

    def test_nonconsecutive_candidate_does_not_confirm(self):
        gate = SceneConfirmationGate()
        candidate = self.detector.classify(self.reference.copy())
        unknown = LandmarkResult("unknown", 0.0, "uncertain_transition")
        gate.update(candidate)
        gate.update(unknown)
        self.assertIsNone(gate.update(candidate).confirmed_scene)

    def test_candidate_must_follow_confirmed_exterior_for_entry_event(self):
        gate = SceneConfirmationGate()
        field = LandmarkResult("cornelia_exterior_field", 1.0, "candidate")
        castle = self.detector.classify(self.reference.copy())
        gate.update(field)
        self.assertEqual(gate.update(field).confirmed_scene, "cornelia_exterior_field")
        gate.update(castle)
        update = gate.update(castle)
        self.assertTrue(update.first_castle_entry)
        self.assertFalse(gate.update(castle).first_castle_entry)

    def test_gate_reset_prevents_loaded_interior_false_entry(self):
        gate = SceneConfirmationGate()
        result = self.detector.classify(self.reference.copy())
        gate.update(result)
        gate.update(result)
        gate.reset()
        gate.update(result)
        update = gate.update(result)
        self.assertTrue(update.transitioned)
        self.assertFalse(update.first_castle_entry)


if __name__ == "__main__":
    unittest.main()
