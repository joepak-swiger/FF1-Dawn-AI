"""Offline contract tests: synthetic images are tests, not FF1 training evidence."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ff1_dawn_ai.gba_combat_learning import (
    CombatDecisionRecorder, CombatLinearPolicy, feature, grouped_splits,
    load_dataset, load_datasets, train_from_directory, train_from_directories,
    train_policy, ACTIONS,
)


class GBACombatLearningTests(unittest.TestCase):
    def test_record_train_save_reload_and_leakproof_split(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            recorder = CombatDecisionRecorder(base)
            for group in range(9):
                for sample, action in enumerate(("A", "B", "A", "B")):
                    value = 30 + group * 11 + (action == "B") * 40 + sample
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

    def test_multiple_capture_directories_keep_groups_and_deduplicate_exact_replays(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            directories = []
            for group in range(7):
                directory = root / f"capture-{group}"
                recorder = CombatDecisionRecorder(directory)
                for stage, action, value in (("attack_command", "A", 24 + group),
                                             ("target_confirm", "A", 64 + group),
                                             ("menu_other", "B", 120 + group)):
                    recorder.record(frame=np.full((160, 240, 3), value, dtype=np.uint8),
                                    stage=stage, action=action, group_id=f"parent-{group}",
                                    encounter_id=f"encounter-{group}")
                directories.append(directory)
            # Loading a replay of one capture twice must not duplicate samples.
            X, y, rows = load_datasets(directories + [directories[0]])
            self.assertEqual(21, len(rows))
            self.assertEqual(21, len(set((r["group_id"], r["frame_sha256"], r["stage"], r["action"]) for r in rows)))
            self.assertEqual(7, len({r["group_id"] for r in rows}))
            output = root / "combined_model.npz"
            metrics = train_from_directories(directories + [directories[0]], output, seed=12)
            self.assertTrue(output.is_file())
            self.assertEqual(7, metrics["unique_encounter_parent_groups"])
            self.assertEqual(21, metrics["deduplicated_samples"])
            for split_name in ("train", "validation", "test"):
                report = metrics[f"{split_name}_classification"]
                self.assertIn("confusion_matrix", report)
                self.assertIn("per_action", report)
                self.assertIn("per_stage", report)
                self.assertGreater(report["samples"], 0)
            self.assertIn("menu_other", metrics["test_classification"]["per_stage"])
            self.assertIn("B", metrics["test_classification"]["per_action"])

    def test_group_mapping_merges_replay_states_before_splitting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            directories = []
            for source_group in range(7):
                directory = root / f"capture-{source_group}"
                recorder = CombatDecisionRecorder(directory)
                parent = source_group // 2 if source_group < 4 else source_group
                for sample, (stage, action) in enumerate((("attack_command", "A"),
                                                          ("target_confirm", "A"),
                                                          ("menu_other", "B"))):
                    recorder.record(frame=np.full((160, 240, 3), 20 + source_group * 4 + sample, np.uint8),
                                    stage=stage, action=action,
                                    group_id=f"replay-{source_group}", encounter_id=f"enc-{source_group}")
                directories.append(directory)
            mapping = {f"replay-{i}": f"parent-{i // 2}" for i in range(4)}
            X, y, rows = load_datasets(directories, group_mapping=mapping)
            self.assertEqual(5, len({row["group_id"] for row in rows}))
            splits = grouped_splits(rows, seed=5)
            sets = [{rows[int(i)]["group_id"] for i in indices} for indices in splits.values()]
            self.assertTrue(sets[0].isdisjoint(sets[1]))
            self.assertTrue(sets[0].isdisjoint(sets[2]))
            self.assertTrue(sets[1].isdisjoint(sets[2]))

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
