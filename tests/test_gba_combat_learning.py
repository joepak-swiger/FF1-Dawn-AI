"""Offline contract tests: synthetic images are tests, not FF1 training evidence."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ff1_dawn_ai.gba_combat_learning import (
    CombatDecisionRecorder, CombatLinearPolicy, feature, grouped_splits,
    load_dataset, train_from_directory, train_policy, ACTIONS,
)


class GBACombatLearningTests(unittest.TestCase):
    def test_record_train_save_reload_and_leakproof_split(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            recorder = CombatDecisionRecorder(base)
            for group in range(9):
                for action in ("A", "B", "A", "B"):
                    value = 30 + group * 11 + (action == "B") * 40
                    recorder.record(frame=np.full((160, 240, 3), value, dtype=np.uint8),
                                    stage="attack_command", action=action,
                                    group_id=f"encounter-parent-{group}", encounter_id=f"battle-{group}")
            X, y, rows = load_dataset(base)
            self.assertEqual((36, 1203), X.shape)
            splits = grouped_splits(rows)
            family_sets = [{rows[int(i)]["group_id"] for i in idx} for idx in splits.values()]
            self.assertTrue(family_sets[0].isdisjoint(family_sets[1]))
            self.assertTrue(family_sets[0].isdisjoint(family_sets[2]))
            self.assertTrue(family_sets[1].isdisjoint(family_sets[2]))
            self.assertEqual(set(range(36)), set(np.concatenate(list(splits.values())).tolist()))
            path = base / "test_policy.npz"
            metrics = train_from_directory(base, path)
            self.assertTrue(path.is_file())
            self.assertEqual(metrics["rl_timesteps"], 0)
            loaded = CombatLinearPolicy.load(path)
            predicted, confidence = loaded.predict(np.full((160, 240, 3), 50, np.uint8), "attack_command")
            self.assertIn(predicted, ("A", "B"))
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_no_fake_learning_from_only_A(self):
        with tempfile.TemporaryDirectory() as td:
            recorder = CombatDecisionRecorder(Path(td))
            for group in range(6):
                for sample in range(4):
                    recorder.record(frame=np.zeros((160, 240, 3), np.uint8), stage="target_confirm",
                                    action="A", group_id=f"group-{group}", encounter_id=f"battle-{group}")
            with self.assertRaisesRegex(ValueError, "only one"):
                train_from_directory(Path(td), Path(td) / "should-not-exist.npz")
            self.assertFalse((Path(td) / "should-not-exist.npz").exists())

    def test_corruption_and_unverified_labels_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            recorder = CombatDecisionRecorder(Path(td))
            with self.assertRaises(ValueError):
                recorder.record(frame=np.zeros((160, 240, 3), np.uint8), stage="attack_command",
                                action="A", group_id="x", encounter_id="x", source="learned")
            recorder.record(frame=np.zeros((160, 240, 3), np.uint8), stage="attack_command",
                            action="A", group_id="x", encounter_id="x")
            (Path(td) / "frames" / "decision_0000000.png").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_dataset(Path(td))
            with self.assertRaises(ValueError):
                grouped_splits([{"group_id": "x"} for _ in range(9)])

    def test_input_validation_and_low_sample_training(self):
        with self.assertRaises(ValueError):
            feature(np.zeros((160, 240), dtype=np.uint8), "attack_command")
        with self.assertRaises(ValueError):
            feature(np.zeros((160, 240, 3), dtype=np.uint8), "invalid")
        with self.assertRaisesRegex(ValueError, "insufficient"):
            train_policy(np.zeros((8, 1203), np.float32), np.array([5, 5, 5, 6, 5, 6, 5, 6]),
                         {"train": np.arange(4), "validation": np.arange(4, 6), "test": np.arange(6, 8)})


if __name__ == "__main__":
    unittest.main()
