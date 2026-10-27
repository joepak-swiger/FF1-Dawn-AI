import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ff1_dawn_ai.nes_scene_audit import parse_anchor, run_audit, visual_layout_family


class NesSceneAuditTests(unittest.TestCase):
    def test_anchor_parser_requires_explicit_session_step_and_label(self):
        self.assertEqual(parse_anchor("session_01:12=title"), {
            "session_id": "session_01", "step": 12, "label": "title"
        })
        with self.assertRaises(ValueError):
            parse_anchor("12=title")

    def test_layout_cues_separate_only_obvious_synthetic_layouts(self):
        self.assertEqual(visual_layout_family({
            "green_fraction": .3, "dark_fraction": .1,
            "right_party_panel_edge_brightness": 0.0,
        }), "overworld_map_layout_candidate")
        self.assertEqual(visual_layout_family({
            "green_fraction": .01, "dark_fraction": .3,
            "right_party_panel_edge_brightness": .2,
        }), "battle_party_panel_layout_candidate")
        self.assertEqual(visual_layout_family({
            "green_fraction": 0.0, "dark_fraction": .8,
            "right_party_panel_edge_brightness": 0.0,
        }), "dark_menu_or_transition_layout_candidate")

    def test_audit_accepts_csv_boolean_and_exports_no_machine_local_paths(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            capture = root / "capture"
            session = capture / "session_01"
            frames = session / "frames"
            frames.mkdir(parents=True)
            green = np.zeros((96, 128, 3), dtype=np.uint8)
            green[:, :, 1] = 180
            Image.fromarray(green).save(frames / "frame_00000_abcd.png")
            (session / "probe_events.csv").write_text(
                "step,battle_like\n0,False\n", encoding="utf-8"
            )
            (capture / "metrics.json").write_text(json.dumps({
                "run_id": "synthetic", "status": "complete",
                "episodes": [{"session_id": "session_01", "steps_recorded": 1, "visual_transitions": []}],
            }), encoding="utf-8")
            output = root / "audit"
            run_audit(capture, output, [{"session_id": "session_01", "step": 0, "label": "field"}])
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(metrics["uses_ram"])
            self.assertEqual(metrics["manual_visual_anchors"][0]["label"], "field")
            csv_text = (output / "frame_features.csv").read_text(encoding="utf-8")
            self.assertNotIn(str(root), csv_text)
            self.assertTrue((output / "evidence" / "transition_scenes_01.png").is_file())
            with self.assertRaises(FileExistsError):
                run_audit(capture, output)


if __name__ == "__main__":
    unittest.main()
