import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ff1_dawn_ai.gba_autoplay import (
    BattleCommandDetector,
    BattleConfirmationGate,
    BattleExpResultDetector,
    BattleGilResultDetector,
    BattleLevelUpDetector,
    BattleTargetDetector,
    CorneliaPostBattleMapDetector,
    AutonomousSessionCoordinator,
    ScriptedCastleExcursion,
    SeededRandomExplorer,
    conservative_battle_outcome,
    compare_matched_movement_frames,
)
from scripts.gba_autoplay import _overlay
from scripts.gba_autoplay import sanitized_public_metrics


class GbaAutoplayControllerTests(unittest.TestCase):
    def test_public_metrics_keep_battle_level_evidence_and_omit_local_artifact_paths(self):
        raw = {
            "run_id": "run-1", "status": "bounded", "policy": "autonomous_session",
            "seed": 77, "checkpoint_sha256": "abc", "decisions": 12,
            "emulator_frames": 12, "diagnostic_emulator_frames": 2,
            "action_counts": {"UP": 3}, "confirmed_scenes": [],
            "castle_reached": False, "field_return_confirmed": False,
            "battle_command_screen_confirmed": True, "battles_entered": 1,
            "battles_finished": 1, "battle_victories": 1, "battle_escapes": 0,
            "battle_defeats": 0, "battle_unknown_outcomes": 0, "movement_resumed": 1,
            "movement_probes": [{"movement_signal_confirmed": True}],
            "battle_encounters": [{"battle_id": 1, "outcome": "victory", "exp_amount": 4}],
            "exp_awards": [4], "gil_awards": [18], "confirmed_level_ups": [],
            "battle_commands": 4, "battle_target_confirmed": True,
            "next_party_command_confirmed": False, "post_target_wait_frames": 2,
            "battle_A_presses": 8, "battle_command_panel_returned": False,
            "field_return_after_battle": False, "battle_resolution_wait_frames": 20,
            "battle_detection": {}, "battle_outcomes": {"victories": 1},
            "exp_reward_banner_confirmed": True,
            "specific_earned_6_exp_banner_confirmed": False,
            "specific_obtained_24_gil_banner_confirmed": False,
            "post_battle_map_view_confirmed": True, "unknown_scene_observations": 5,
            "elapsed_seconds": 1.2, "output_directory_local_only": "videos/private-run",
        }
        safe = sanitized_public_metrics(raw)
        self.assertEqual(safe["battle_encounters"], raw["battle_encounters"])
        self.assertEqual(safe["recognized_exp_amounts"], [4])
        self.assertEqual(safe["recognized_gil_amounts"], [18])
        self.assertEqual(safe["policy_is_learned"], False)
        self.assertEqual(safe["training_timesteps"], 0)
        self.assertNotIn("output_directory_local_only", safe)
        self.assertNotIn("videos/private-run", repr(safe))

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
        self.assertFalse(gate.confirmed, "a later mismatch must clear current confirmation")

    def test_reviewed_goblin_target_panel_is_separate_from_command_panel(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / "docs/experiments/gba_autoplay/battle_ui_reference"
        detector = BattleTargetDetector(str(evidence / "goblin_target_positive_left.png"),
            alternative_references=(str(evidence / "goblin_target_positive_four.png"),), threshold=0.95)
        positives = (evidence / "goblin_target_positive_left.png",
                     evidence / "goblin_target_positive_actor2_left.png",
                     evidence / "goblin_target_positive_four.png")
        for path in positives:
            self.assertTrue(detector.is_candidate(self._frame_with_reviewed_panel(path)), path.name)
        negatives = (evidence / "attack_command_positive_left.png",
                     evidence / "magic_menu_negative_left.png")
        for path in negatives:
            self.assertFalse(detector.is_candidate(self._frame_with_reviewed_panel(path)), path.name)

        # The four-Goblin screenshot was independently captured after B from
        # a reviewed Magic menu; it must be recognized by its exact reviewed
        # formation anchor, not by relaxing the matching threshold.
        four_frame = self._frame_with_reviewed_panel(evidence / "goblin_target_positive_four.png")
        self.assertGreaterEqual(detector.score(four_frame), 0.999)

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
            root / "docs/experiments/gba_autoplay/issue17_continuous/proof/battle_command_negative.png",
        )
        for path in negatives:
            frame = np.asarray(Image.open(path).convert("RGB"))
            self.assertFalse(detector.is_candidate(frame), f"false EXP match: {path.name}")

        gate = BattleConfirmationGate(detector, consecutive_frames=2)
        self.assertFalse(gate.update(positive))
        self.assertTrue(gate.update(positive))
        gate.reset()
        self.assertFalse(gate.update(np.asarray(Image.open(negatives[0]).convert("RGB"))))

    def test_reward_amount_templates_distinguish_real_6_vs_4_and_24_vs_18(self):
        root = Path(__file__).resolve().parents[1]
        refs = root / "docs/experiments/gba_autoplay/battle_win_reference"
        exp = BattleExpResultDetector(str(refs / "earned_6_exp_banner.png"),
            alternative_references={4: str(refs / "earned_4_exp_banner.png")})
        six = np.asarray(Image.open(refs / "earned_6_exp_frame.png").convert("RGB"))
        four = np.asarray(Image.open(refs / "earned_4_exp_frame.png").convert("RGB"))
        self.assertTrue(exp.is_candidate(six))
        self.assertTrue(exp.is_candidate(four))
        self.assertEqual(exp.recognized_amount(six), 6)
        self.assertEqual(exp.recognized_amount(four), 4)

        gil = BattleGilResultDetector(str(refs / "obtained_24_gil_banner.png"),
            alternative_references={18: str(refs / "obtained_18_gil_banner.png")})
        twenty_four = np.asarray(Image.open(refs / "obtained_24_gil_frame.png").convert("RGB"))
        eighteen = np.asarray(Image.open(refs / "obtained_18_gil_frame.png").convert("RGB"))
        self.assertTrue(gil.is_candidate(twenty_four))
        self.assertTrue(gil.is_candidate(eighteen))
        self.assertEqual(gil.recognized_amount(twenty_four), 24)
        self.assertEqual(gil.recognized_amount(eighteen), 18)

        unknown_amount = four.copy()
        unknown_amount[2:14, 122:132] = (0, 0, 0)
        self.assertTrue(exp.is_candidate(unknown_amount), "fixed reward phrase remains recognizable")
        self.assertIsNone(exp.recognized_amount(unknown_amount), "unreviewed digits remain unparsed")

    def test_battle_confirmation_is_a_current_streak_not_a_sticky_latch(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / "docs/experiments/gba_autoplay/battle_ui_reference"
        detector = BattleCommandDetector(str(evidence / "attack_command_positive_left.png"), threshold=0.95)
        positive = self._frame_with_reviewed_panel(evidence / "attack_command_positive_left.png")
        negative = self._frame_with_reviewed_panel(evidence / "magic_menu_negative_left.png")
        gate = BattleConfirmationGate(detector, consecutive_frames=2)
        self.assertFalse(gate.update(positive))
        self.assertTrue(gate.update(positive))
        self.assertTrue(gate.confirmed)
        self.assertFalse(gate.update(negative))
        self.assertFalse(gate.confirmed)
        self.assertFalse(gate.update(positive), "one stale/lookalike frame cannot reopen the gate")

    def test_continuous_encounter_accounting_is_idempotent_per_battle(self):
        session = AutonomousSessionCoordinator()
        first = session.battle_detected(step=12)
        self.assertEqual(first, 1)
        self.assertTrue(session.reward_confirmed("exp"))
        self.assertFalse(session.reward_confirmed("exp"))
        self.assertTrue(session.reward_confirmed("gil"))
        session.finish_battle("victory")
        session.finish_battle("victory")
        session.map_view_observed()
        session.movement_observed(confirmed=False)
        self.assertEqual(session.victories, 1)
        self.assertEqual(session.battles_finished, 1)
        self.assertEqual(session.map_views, 1)
        self.assertEqual(session.movement_resumed, 0)
        session.movement_observed(confirmed=True)
        second = session.new_encounter(step=30)
        self.assertEqual(second, 2)
        session.finish_battle("unknown")
        self.assertEqual(session.battles_entered, 2)
        self.assertEqual(session.battles_finished, 2)
        self.assertEqual(session.victories, 1)
        self.assertEqual(session.unknown_outcomes, 1)
        self.assertEqual([row["battle_id"] for row in session.encounters], [1, 2])

    def test_level_up_confirmation_is_specific_and_idempotent(self):
        session = AutonomousSessionCoordinator()
        session.battle_detected(step=1)
        self.assertTrue(session.level_up_confirmed("Sif"))
        self.assertFalse(session.level_up_confirmed("Sif"))
        self.assertEqual(session.encounters[0]["level_ups"], ["Sif"])

        root = Path(__file__).resolve().parents[1]
        evidence = root / "docs/experiments/gba_autoplay/battle_win_reference"
        detector = BattleLevelUpDetector(str(evidence / "sif_level_up_banner.png"))
        positive = np.asarray(Image.open(evidence / "sif_level_up_frame.png").convert("RGB"))
        negative = np.asarray(Image.open(evidence / "earned_4_exp_frame.png").convert("RGB"))
        self.assertTrue(detector.is_candidate(positive))
        self.assertGreaterEqual(detector.score(positive), 0.999)
        self.assertFalse(detector.is_candidate(negative))
        gate = BattleConfirmationGate(detector, consecutive_frames=2)
        self.assertFalse(gate.update(positive))
        self.assertTrue(gate.update(positive))

    def test_matched_movement_probe_requires_stable_repeats_and_opposite_controls(self):
        idle = np.zeros((24, 24, 3), dtype=np.uint8)
        forward = idle.copy()
        forward[:, :12] = 40
        opposite = idle.copy()
        opposite[:, 12:] = 40
        result = compare_matched_movement_frames(idle, idle, forward, forward, opposite, opposite)
        self.assertTrue(result["movement_signal_confirmed"])
        unstable_idle = idle.copy()
        unstable_idle[0, 0] = 255
        result = compare_matched_movement_frames(idle, unstable_idle, forward, forward,
                                                  opposite, opposite)
        self.assertFalse(result["movement_signal_confirmed"])
        result = compare_matched_movement_frames(idle, idle, forward, forward,
                                                  forward, forward)
        self.assertFalse(result["movement_signal_confirmed"])

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

        map_detector = CorneliaPostBattleMapDetector((
            str(evidence / "cornelia_postbattle_static_anchor.png"),
            str(evidence / "cornelia_postbattle_map_anchor_camera2.png"),
            str(evidence / "cornelia_postbattle_map_anchor_camera3.png"),
            str(evidence / "cornelia_postbattle_map_anchor_camera4.png"),
        ), threshold=0.995)
        map_frames = (
            np.asarray(Image.open(evidence / "cornelia_postbattle_map_frame.png").convert("RGB")),
            np.asarray(Image.open(evidence / "cornelia_postbattle_map_frame_repeat.png").convert("RGB")),
        )
        self.assertTrue(all(map_detector.is_candidate(frame) for frame in map_frames))
        second_view = np.asarray(Image.open(
            evidence / "cornelia_postbattle_map_frame_camera2.png").convert("RGB"))
        self.assertTrue(map_detector.is_candidate(second_view))
        # Candidate three was manually reviewed as the playable map view that
        # the earlier narrow anchor set missed. It is not a field coordinate;
        # runtime use still requires the two-frame gate and matched controls.
        candidate_view = np.asarray(Image.open(
            root / "docs/experiments/gba_autoplay/issue17_continuous/proof/seed1729_postbattle_camera_candidate_1.png"
        ).convert("RGB"))
        self.assertTrue(map_detector.is_candidate(candidate_view))
        second_candidate_view = np.asarray(Image.open(
            root / "docs/experiments/gba_autoplay/issue17_continuous/proof/seed1729_postbattle_camera_candidate_2.png"
        ).convert("RGB"))
        self.assertTrue(map_detector.is_candidate(second_candidate_view))
        map_gate = BattleConfirmationGate(map_detector, consecutive_frames=2)
        self.assertFalse(map_gate.update(map_frames[0]))
        self.assertTrue(map_gate.update(map_frames[1]))

        negatives = (
            root / "docs/experiments/gba_autoplay/issue17_continuous/proof/battle_command_negative.png",
            root / "docs/experiments/gba_castle_goal/reference_cornelia_1f.png",
            root / "docs/experiments/gba_castle_goal/20261028_052724_624774Z/party_menu_negative_control.png",
            root / "docs/experiments/gba_navigation_imitation/field_anchor_center_settled.png",
            root / "docs/experiments/gba_autoplay/issue17_continuous/proof/battle_command_negative.png",
        )
        for path in negatives:
            frame = np.asarray(Image.open(path).convert("RGB"))
            self.assertFalse(map_detector.is_candidate(frame), path.name)


if __name__ == "__main__":
    unittest.main()
