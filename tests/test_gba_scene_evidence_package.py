import json
import tempfile
import unittest
from pathlib import Path

from scripts import package_gba_scene_learning_evidence as packager


class SceneLearningEvidencePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene_run = self.root / "videos/gba_scene_eval/scene-run"
        self.learning_run = self.root / "videos/gba_castle_learning/learn-run"
        self.scene_run.mkdir(parents=True)
        self.learning_run.mkdir(parents=True)
        validation = {"independent_entry_exit_episodes": 3,
                      "exterior_reset_candidates": 3, "castle_entry_confirmed": 3,
                      "exterior_exit_confirmed": 3, "cornelia_town_confirmed": 1,
                      "independent_random_episodes": 3, "menu_confirmed": True,
                      "independent_scene_confusion": {
                          key: {"expected": 1, "confirmed": 1, "missed": 0}
                          for key in ("cornelia_exterior_field", "cornelia_town",
                                      "castle_cornelia_1f", "menu")},
                      "unexpected_confirmed_scene_transitions": []}
        scene = {"run_id": "scene-run", "checkpoint_run_id": "ckpt",
                 "episode_level_validation": validation}
        (self.scene_run / "metrics.json").write_text(json.dumps(scene), encoding="utf-8")
        for filename in packager.SCENE_PROOFS:
            (self.scene_run / filename).write_bytes(b"proof")
        evaluation = {policy: {"successes": 0, "episodes": 3, "steps_to_goal": []}
                      for policy in ("SCRIPTED_UP", "SEEDED_RANDOM", "IDLE", "PPO_TRAINED")}
        learning = {"run_id": "learn-run", "checkpoint_run_id": "ckpt",
                    "checkpoint_sha256_unchanged": True,
                    "reward": {"value": 1.0}, "scene_gate": {"consecutive_frames": 2},
                    "training": {"timesteps_actual": 2048, "elapsed_seconds": 2.0,
                                 "episodes_with_goal_events": 0,
                                 "model_file_local_only": "private.zip"},
                    "evaluation": evaluation, "claim": "smoke", "elapsed_seconds": 5.0,
                    "episodes": [{"action_trace": "private"}]}
        (self.learning_run / "metrics.json").write_text(json.dumps(learning), encoding="utf-8")
        (self.learning_run / "policy_contact_sheet.png").write_bytes(b"sheet")

    def test_packages_only_whitelisted_evidence_and_aggregates(self):
        scene, learning = packager.package("scene-run", "learn-run", root=self.root)
        self.assertTrue((scene / "REPORT.md").is_file())
        self.assertEqual(len(list(scene.glob("*.png"))), len(packager.SCENE_PROOFS))
        self.assertTrue((learning / "policy_contact_sheet.png").is_file())
        metrics = json.loads((learning / "metrics.json").read_text(encoding="utf-8"))
        self.assertNotIn("episodes", metrics)
        self.assertNotIn("model_file_local_only", metrics["training"])
        self.assertFalse(list(learning.glob("*.zip")))

    def test_package_does_not_overwrite_completed_evidence(self):
        scene, _ = packager.package("scene-run", "learn-run", root=self.root)
        with self.assertRaises(FileExistsError):
            packager.package("scene-run", "learn-run", root=self.root)
        self.assertTrue((scene / "REPORT.md").is_file())

    def test_run_id_cannot_escape_local_capture_folder(self):
        with self.assertRaises(ValueError):
            packager.package("../outside", "learn-run", root=self.root)


if __name__ == "__main__":
    unittest.main()
