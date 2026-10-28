import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.package_gba_locomotion_evidence import package


class GbaLocomotionEvidenceTests(unittest.TestCase):
    def _run(self, root: Path) -> Path:
        source = root / "local-run"
        source.mkdir()
        movements = {}
        for horizon in ("1", "8"):
            movements[horizon] = {
                direction: [
                    {"scene_translation_candidate": {
                        "dx_pixels_candidate": (-8 if direction == "RIGHT" else 8 if direction == "LEFT" else 0),
                        "dy_pixels_candidate": (8 if direction == "UP" else -8 if direction == "DOWN" else 0),
                    }}
                    for _ in range(3)
                ]
                for direction in ("UP", "DOWN", "LEFT", "RIGHT")
            }
        reversals = {}
        for axis in ("vertical", "horizontal"):
            for horizon in (8, 16, 32):
                for repeat in range(1, 4):
                    reversals[f"{axis}_n{horizon}_r{repeat}"] = {
                        "return_scene_translation": {"dx_pixels_candidate": 0.01, "dy_pixels_candidate": 0.01},
                        "idle_scene_translation": {"dx_pixels_candidate": 0.0, "dy_pixels_candidate": 0.0},
                        "return_vs_start_changed_fraction": 0.06,
                        "idle_vs_start_changed_fraction": 0.05,
                    }
                reversals[f"{axis}_n{horizon}_ram_return_offsets"] = [13308]
        candidate = {
            "offset": 13308, "checkpoint_value": 132,
            "direction_values": {"UP": 142, "DOWN": 134, "LEFT": 138, "RIGHT": 138},
        }
        metrics = {
            "run_id": "run-id", "checkpoint_run_id": "checkpoint-id",
            "checkpoint_unchanged": True, "reset_observation_hashes_identical": True,
            "repeat_count": 3, "reset_count": 20, "action_frames_per_step": 1,
            "movement_horizons_emulator_frames": [1, 8],
            "reversal_tests": reversals,
            "ram_size_bytes": 527360,
            "ram_candidates_by_horizon": {
                "8": {"candidate_offsets": [candidate], "directional_candidate_counts": {
                    "UP": 5, "DOWN": 4, "LEFT": 3, "RIGHT": 3}},
                "1": {"candidate_offsets": [candidate], "directional_candidate_counts": {
                    "UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}},
            },
            "interpretation": {"rewards": "neutral; no training ran"},
        }
        metrics["directional_movement_metrics"] = {
            direction: {h: movements[h][direction] for h in movements}
            for direction in ("UP", "DOWN", "LEFT", "RIGHT")
        }
        (source / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        for name in ("movement_sheet.png", "reversal_sheet.png", "north_path_sheet.png"):
            Image.new("RGB", (64, 32), (25, 75, 100)).save(source / name)
        return source

    def test_packager_whitelists_findings_and_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._run(root)
            (source / "metrics.json").write_text(
                (source / "metrics.json").read_text().replace(
                    '"checkpoint_run_id": "checkpoint-id",',
                    '"checkpoint_run_id": "checkpoint-id", "checkpoint_sha256": "private", "absolute_checkpoint_path": "C:/private/state",',
                ), encoding="utf-8"
            )
            target = package(source, root / "published")
            metrics = json.loads((target / "metrics.json").read_text())
            self.assertNotIn("checkpoint_sha256", metrics)
            self.assertNotIn("absolute_checkpoint_path", metrics)
            self.assertFalse(metrics["absolute_coordinates_verified"])
            self.assertEqual(len(list(target.glob("*.png"))), 3)
            self.assertTrue((target / "REPORT.md").is_file())
            with self.assertRaises(FileExistsError):
                package(source, target)

    def test_packager_refuses_nonrepeatable_or_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._run(root)
            path = source / "metrics.json"
            metrics = json.loads(path.read_text())
            metrics["checkpoint_unchanged"] = False
            path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "incomplete locomotion"):
                package(source, root / "rejected")


if __name__ == "__main__":
    unittest.main()
