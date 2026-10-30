from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ff1_dawn_ai.gba_menu import GbaMenuController, GbaMenuSceneClassifier


ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "docs/experiments/gba_menu_controller/reference"


def classifier() -> GbaMenuSceneClassifier:
    return GbaMenuSceneClassifier({
        "main_menu": REFS / "main_menu_items_cursor.png",
        "main_menu_equipment_cursor": REFS / "main_menu_equipment_cursor.png",
        "empty_items": REFS / "empty_items_menu.png",
        "equipment_character_select": REFS / "equipment_character_select_sif.png",
        "sif_equipment": REFS / "sif_equipment_menu.png",
    })


class GbaMenuClassifierTests(unittest.TestCase):
    def test_reviewed_anchors_classify_and_transition_screen_stays_unknown(self):
        detector = classifier()
        for name, path in (
            ("main_menu", "main_menu_items_cursor.png"),
            ("main_menu_equipment_cursor", "main_menu_equipment_cursor.png"),
            ("empty_items", "empty_items_menu.png"),
            ("equipment_character_select", "equipment_character_select_sif.png"),
            ("sif_equipment", "sif_equipment_menu.png"),
        ):
            with Image.open(REFS / path) as image:
                scene, score = detector.classify(np.asarray(image.convert("RGB")))
            self.assertEqual(scene, name)
            self.assertAlmostEqual(score, 1.0)
        self.assertEqual(detector.classify(np.full((160, 240, 3), 255, dtype=np.uint8))[0], "unknown")

    def test_main_menu_detector_tolerates_small_cursor_or_animation_change(self):
        detector = classifier()
        with Image.open(REFS / "main_menu_items_cursor.png") as image:
            frame = np.asarray(image.convert("RGB")).copy()
        frame[2:12, 150:160] = 255 - frame[2:12, 150:160]
        self.assertEqual(detector.classify(frame)[0], "main_menu")


class GbaMenuControllerTests(unittest.TestCase):
    def test_inventory_and_equipment_paths_are_bounded_and_read_only(self):
        detector = classifier()
        templates = {}
        for name, path in (
            ("field", None),
            ("main_menu", "main_menu_items_cursor.png"),
            ("main_menu_equipment_cursor", "main_menu_equipment_cursor.png"),
            ("empty_items", "empty_items_menu.png"),
            ("equipment_character_select", "equipment_character_select_sif.png"),
            ("sif_equipment", "sif_equipment_menu.png"),
        ):
            if path:
                with Image.open(REFS / path) as image:
                    templates[name] = np.asarray(image.convert("RGB"))
        templates["field"] = np.zeros((160, 240, 3), dtype=np.uint8)

        class FakeGame:
            def __init__(self):
                self.scene = "field"
                self.downs = 0
                self.seen = []

            def step(self, action: str, frames: int):
                self.seen.append((action, frames))
                if action == "START":
                    self.scene = "main_menu"
                elif action == "DOWN" and self.scene == "main_menu":
                    self.downs += 1
                    self.scene = "main_menu_equipment_cursor"
                elif action == "A" and self.scene in {"main_menu", "main_menu_equipment_cursor"}:
                    if self.scene == "main_menu_equipment_cursor":
                        self.scene = "equipment_character_select"
                    else:
                        self.scene = "empty_items"
                elif action == "A" and self.scene == "equipment_character_select":
                    self.scene = "sif_equipment"
                return templates[self.scene]

        game = FakeGame()
        menu = GbaMenuController(game.step, detector)
        inventory = menu.inspect_inventory(templates["field"])
        self.assertEqual(inventory.status, "empty_inventory_confirmed")
        self.assertNotIn(("B", 1), game.seen)
        self.assertTrue(all(action in {"NOOP", "START", "A"} for action, _ in game.seen))

        game.scene, game.downs, game.seen = "field", 0, []
        menu = GbaMenuController(game.step, detector)
        gear = menu.inspect_sif_equipment(templates["field"])
        self.assertEqual(gear.status, "read_only_equipment_confirmed")
        self.assertIn(("DOWN", 1), game.seen)
        self.assertFalse(any(action in {"REMOVE", "USE", "BUY"} for action, _ in game.seen))
        self.assertLessEqual(gear.input_frames, 256)

    def test_unknown_checkpoint_menu_does_not_accept_as_success(self):
        detector = classifier()
        with Image.open(REFS / "main_menu_items_cursor.png") as image:
            main = np.asarray(image.convert("RGB"))
        controller = GbaMenuController(lambda action, frames: main, detector, max_wait_frames=4)
        with self.assertRaisesRegex(RuntimeError, "expected two consecutive empty_items"):
            controller.inspect_inventory(np.zeros((160, 240, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
