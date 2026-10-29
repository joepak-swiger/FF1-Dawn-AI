import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ff1_dawn_ai.gba_autoplay import (
    BattleCommandDetector,
    BattleConfirmationGate,
    BattleExpResultDetector,
    BattleGilResultDetector,
    BattleTargetDetector,
    CorneliaPostBattleMapDetector,
    ScriptedCastleExcursion,
    SeededRandomExplorer,
    conservative_battle_outcome,
)
from scripts.gba_autoplay import _overlay


class GbaAutoplayControllerTests(unittest.TestCase):
    def test_scripted_excursion_reverses_only_after_confirmed_scenes(self):
        controller = ScriptedCastleExcursion(max_approach_actions=3, max_return_actions=2)
        self.assertEqual([controller.next_action() for _ in range(3)], ["UP"] * 3)
        self.assertIsNone(controller.next_action())
        self.assertEqual(controller.stop_reason, "castle_entry_timeout")

        controller = ScriptedCastleExcursion()
        self.assertEqual(controller.next_action(), "UP")
        controller.observe("unknown")
        self.assertEqual(controller.next_action(), "UP")
        controller.observe("castle_cornelia_1f")
        self.assertEqual(controller.next_action(), "DOWN")
        controller.observe("unknown")
        self.assertEqual(controller.next_action(), "DOWN")
        controller.observe("cornelia_exterior_field")
        self.assertIsNone(controller.next_action())
        self.assertTrue(controller.castle_reached)
        self.assertTrue(controller.field_return_confirmed)

    def test_seeded_random_explorer_is_reproducible_and_movement_only(self):
        first = SeededRandomExplorer(42)
        second = SeededRandomExplorer(42)
        actions_a = [first.next_action() for _ in range(100)]
        actions_b = [second.next_action() for _ in range(100)]
        self.assertEqual(actions_a, actions_b)
        self.assertTrue(set(actions_a) <= {"UP", "DOWN", "LEFT", "RIGHT"})

    def test_unverified_screen_change_never_becomes_a_battle_outcome(self):
        self.assertEqual(conservative_battle_outcome(), "unknown")
        self.assertEqual(conservative_battle_outcome(independently_confirmed_result="screen_changed"), "unknown")
        self.assertEqual(conservative_battle_outcome(independently_confirmed_result="victory"), "victory")

    @staticmethod
    def _frame_with_reviewed_panel(path: Path) -> np.ndarray:
        crop = np.asarray(Image.open(path).convert("RGB"))
        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        frame[108:158, 0:100] = crop
        return frame

    def test_attack_cursor_panel_excludes_target_and_magic_controls(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / "docs/experiments/gba_autoplay/battle_ui_reference"
        detector = BattleCommandDetector(str(evidence / "attack_command_positive_left.png"), threshold=0.95)
        positives = (
            evidence / "attack_command_positive_left.png",
            evidence / "attack_command_positive_actor3_left.png",
        )
        for path in positives:
            frame = self._frame_with_reviewed_panel(path)
            self.assertGreaterEqual(detector.score(frame), 0.999, path.name)
            self.assertTrue(detector.is_candidate(frame), path.name)

        negatives = (
            evidence / "goblin_target_positive_left.png",
            evidence / "magic_menu_negative_left.png",
        )
        for path in negatives:
            frame = self._frame_with_reviewed_panel(path)
            self.assertFalse(detector.is_candidate(frame), path.name)

        gate = BattleConfirmationGate(detector, consecutive_frames=2)
        self.assertFalse(gate.update(self._frame_with_reviewed_panel(positives[0])))
        self.assertTrue(gate.update(self._frame_with_reviewed_panel(positives[1])))
        gate.reset()
        self.assertFalse(gate.update(self._frame_with_reviewed_panel(negatives[1])))

    def test_reviewed_goblin_target_panel_is_separate_from_command_panel(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / "docs/experiments/gba_autoplay/battle_ui_reference"
        detector = BattleTargetDetector(str(evidence / "goblin_target_positive_left.png"), threshold=0.95)
        positives = (evidence / "goblin_target_positive_left.png",
                     evidence / "goblin_target_positive_actor2_left.png")
        for path in positives:
            self.assertTrue(detector.is_candidate(self._frame_with_reviewed_panel(path)), path.name)
        negatives = (evidence / "attack_command_positive_left.png",
                     evidence / "magic_menu_negative_left.png")
        for path in negatives:
            self.assertFalse(detector.is_candidate(self._frame_with_reviewed_panel(path)), path.name)

    def test_specific_exp_award_has_independent_visual_negative_controls(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / "docs/experiments/gba_autoplay/battle_win_reference"
        detector = BattleExpResultDetector(str(evidence / "earned_6_exp_banner.png"))
        positive = np.asarray(Image.open(evidence / "earned_6_exp_frame.png").convert("RGB"))
        self.assertGreaterEqual(detector.score(positive), 0.999)
        self.assertTrue(detector.is_candidate(positive))
        annotated = np.asarray(_overlay(positive, policy="seeded random", phase="result",
                                        step=10, action="NOOP", scene="unknown", battle_count=1))
        self.assertTrue(detector.is_candidate(annotated), "video overlay must preserve the top EXP banner")

        negatives = (
            root / "docs/experiments/gba_castle_goal/reference_cornelia_1f.png",
            root / "docs/experiments/gba_castle_goal/20261028_052724_624774Z/party_menu_negative_control.png",
            root / "docs/experiments/gba_navigation_imitation/field_anchor_center_settled.png",
            root / "docs/experiments/gba_autoplay/20261029_023319_343248Z/battle_command_observation_01.png",
        )
        for path in negatives:
            frame = np.asarray(Image.open(path).convert("RGB"))
            self.assertFalse(detector.is_candidate(frame), f"false EXP match: {path.name}")

        gate = BattleConfirmationGate(detector, consecutive_frames=2)
        self.assertFalse(gate.update(positive))
        self.assertTrue(gate.update(positive))
        gate.reset()
        self.assertFalse(gate.update(np.asarray(Image.open(negatives[0]).convert("RGB"))))

    def test_exact_gil_message_and_narrow_postbattle_map_have_negative_controls(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / "docs/experiments/gba_autoplay/battle_win_reference"
        gil_detector = BattleGilResultDetector(str(evidence / "obtained_24_gil_banner.png"))
        gil_frames = (
            np.asarray(Image.open(evidence / "obtained_24_gil_frame.png").convert("RGB")),
        )
        self.assertTrue(gil_detector.is_candidate(gil_frames[0]))
        gil_gate = BattleConfirmationGate(gil_detector, consecutive_frames=2)
        self.assertFalse(gil_gate.update(gil_frames[0]))
        self.assertTrue(gil_gate.update(gil_frames[0]))

        map_detector = CorneliaPostBattleMapDetector(
            str(evidence / "cornelia_postbattle_static_anchor.png"), threshold=0.995
        )
        map_frames = (
            np.asarray(Image.open(evidence / "cornelia_postbattle_map_frame.png").convert("RGB")),
            np.asarray(Image.open(evidence / "cornelia_postbattle_map_frame_repeat.png").convert("RGB")),
        )
        self.assertTrue(all(map_detector.is_candidate(frame) for frame in map_frames))
        map_gate = BattleConfirmationGate(map_detector, consecutive_frames=2)
        self.assertFalse(map_gate.update(map_frames[0]))
        self.assertTrue(map_gate.update(map_frames[1]))

        negatives = (
            root / "docs/experiments/gba_autoplay/20261029_030049_618548Z/battle_command_confirmed.png",
            root / "docs/experiments/gba_castle_goal/reference_cornelia_1f.png",
            root / "docs/experiments/gba_castle_goal/20261028_052724_624774Z/party_menu_negative_control.png",
            root / "docs/experiments/gba_navigation_imitation/field_anchor_center_settled.png",
        )
        for path in negatives:
            frame = np.asarray(Image.open(path).convert("RGB"))
            self.assertFalse(map_detector.is_candidate(frame), path.name)


if __name__ == "__main__":
    unittest.main()
