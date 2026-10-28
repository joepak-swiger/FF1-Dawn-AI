import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import package_gba_castle_evidence as packager


class CastleEvidencePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "videos" / "gba_castle_goal" / "synthetic-run"
        self.run_dir.mkdir(parents=True)
        self.old_root = packager.ROOT
        packager.ROOT = self.root
        self.addCleanup(setattr, packager, "ROOT", self.old_root)

        rows = [{"trial": "scripted_up_01", "step": step, "label": "unknown",
                 "status": "uncertain", "confidence": 0.98} for step in (119, 120, 121)]
        rows += [{"trial": "seeded_random_110", "step": 128, "label": "unknown",
                  "status": "no_match", "confidence": 0.7, "ram_0x33FC": 226, "ram_0x3404": 230}]
        (self.run_dir / "action_trace.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        frames = []
        for action in ("NOOP", "DOWN"):
            for frame in (8, 16, 24, 32, 40, 48, 56, 64):
                frames.append({"action": action, "frame": frame,
                               "scene_translation_candidate": {"dx_pixels_candidate": 0.0,
                                                                 "dy_pixels_candidate": 0.0},
                               "ram_0x33FC": 132, "ram_0x3404": 164})
        metrics = {
            "run_id": "synthetic-run", "checkpoint_run_id": "checkpoint-run",
            "checkpoint_unchanged": True, "reset_observations_identical": True,
            "reset_count": 25, "neutral_rewards_only": True, "training_started": False,
            "destination_detector": {"target": "Castle Cornelia 1F"},
            "policy_summary": {key: {"trials": 3, "successes": 0,
                                     "median_actions_to_confirmed_entry": None}
                               for key in ("SCRIPTED_UP", "SEEDED_RANDOM", "IDLE", "SIDE_DOWN")},
            "all_control_screens": {"frames_checked": 1, "castle_false_positive_frames": 0,
                                    "castle_false_positive_rate": 0.0},
            "start_menu_control": {"frames_checked": 1, "false_positive_frames": 0,
                                   "false_positive_rate": 0.0, "screenshots_retained_at_frames": [32]},
            "castle_exit_test": {"field_return_confirmed": False},
            "ram_candidate_label_differential": {"castle_confirmed_pairs_0x33fc_0x3404": [[226, 230]]},
            "reversal_8_frame_release_diagnostic": {"2": [{
                "immediate_return_scene_shift": {"dy_pixels_candidate": 15.0},
                "settled_return_scene_shift": {"dy_pixels_candidate": 0.0},
                "matched_idle_scene_shift": {"dy_pixels_candidate": 0.0}}]},
            "down_64_frame_diagnostic": {"per_frame_labels_and_ram": frames},
        }
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        blank = Image.fromarray(np.zeros((160, 240, 3), dtype=np.uint8))
        blank.save(self.run_dir / "castle_goal_contact_sheet.png")
        names = (
            "scripted_up_01_step_119.png", "scripted_up_01_step_120.png",
            "scripted_up_01_step_121.png", "start_menu_control_32.png",
            "seeded_random_110_step_128.png", "castle_exit_test_down_072.png",
            "castle_exit_test_down_080.png", "down64_down_64.png",
        )
        for name in names:
            blank.save(self.run_dir / name)

    def test_packages_only_reviewable_summary_and_selected_frames(self):
        output = packager.package(self.run_dir, reviewed_field_return=True)
        self.assertTrue((output / "REPORT.md").is_file())
        self.assertTrue((output / "town_ram_collision.png").is_file())
        self.assertFalse((output / "action_trace.jsonl").exists())
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertTrue(metrics["field_return_visual_review"]["reviewed"])

    def test_completed_package_is_not_overwritten(self):
        output = packager.package(self.run_dir)
        with self.assertRaises(FileExistsError):
            packager.package(self.run_dir)
        self.assertTrue((output / "REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
