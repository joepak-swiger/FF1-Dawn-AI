import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ff1_dawn_ai.ram_analysis import (
    Sample,
    analyze_samples,
    discover_sessions,
    load_manual_csv,
    load_v09_npz,
    pixel_scene_features,
    write_analysis,
)


class RamAnalysisTests(unittest.TestCase):
    def test_visual_proxy_uses_only_rgb_and_preserves_historical_thresholds(self):
        field = np.zeros((224, 240, 3), dtype=np.uint8)
        field[:, :, 1] = 180
        battle = np.zeros((224, 240, 3), dtype=np.uint8)
        battle[180:220, 20:40] = 255
        self.assertFalse(pixel_scene_features(field)["battle_like"])
        self.assertTrue(pixel_scene_features(battle)["battle_like"])

    def test_manual_loader_excludes_label_snapshots_and_keeps_epochs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            path = Path(temp) / "manual_ram_events.csv"
            rows = [
                {"step": 0, "sample_type": "label", "label": "field_overworld", "label_epoch": "1", "ram_0014": "23"},
                {"step": 1, "sample_type": "action", "label": "field_overworld", "label_epoch": "1", "ram_0014": "23"},
                {"step": 2, "sample_type": "action", "label": "battle", "label_epoch": "2", "ram_0014": "0"},
                {"step": 3, "sample_type": "action", "label": "unlabeled", "label_epoch": "", "ram_0014": "0"},
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            samples = load_manual_csv(path, "session-a")
            self.assertEqual([s.step for s in samples], [1, 2, 3])
            self.assertEqual(samples[0].label_epoch, 1)
            result = analyze_samples(samples, [0x14])
            candidate = result["candidates"][0]
            self.assertEqual(candidate["field"]["mode"], 23)
            self.assertEqual(candidate["battle"]["mode"], 0)
            self.assertEqual(candidate["ambiguous_or_non_binary_samples"], 1)

    def test_v09_npz_loader_uses_pixel_labels_not_ram_values(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            rams = np.zeros((4, 32), dtype=np.uint8)
            rams[:, 0x14] = [23, 23, 0, 0]
            np.savez_compressed(root / "ram_samples.npz", rams=rams, battle_like=[False, False, True, True])
            rows = [{"step": str(i), "battle_like": str(i >= 2).lower()} for i in range(4)]
            with (root / "probe_events.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            samples = load_v09_npz(root / "ram_samples.npz", "v09")
            self.assertEqual(samples[0].label_source, "visual_heuristic_v0_9")
            self.assertEqual(samples[-1].label, "battle")
            candidate = analyze_samples(samples, [0x14])["candidates"][0]
            self.assertEqual(candidate["field"]["mode"], 23)
            self.assertEqual(candidate["battle"]["mode"], 0)
            self.assertEqual(candidate["confusion_if_battle_mode_is_battle"]["false_positive"], 0)

    def test_recursive_discovery_keeps_repeated_episode_names_independent(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            for run_id in ("run-a", "run-b"):
                session = root / run_id / "session_01"
                session.mkdir(parents=True)
                rams = np.zeros((2, 32), dtype=np.uint8)
                rams[:, 0x14] = [23, 0]
                np.savez_compressed(session / "ram_samples.npz", rams=rams, battle_like=[False, True])
                with (session / "probe_events.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["step"])
                    writer.writeheader()
                    writer.writerows([{"step": 0}, {"step": 1}])
            samples = discover_sessions([root], addresses=[0x14])
            result = analyze_samples(samples, [0x14])
            self.assertEqual(result["session_count"], 2)
            self.assertEqual(len(result["candidates"][0]["sessions"]), 2)

    def test_analysis_writes_compact_artifacts_and_refuses_overwrite(self):
        samples = [
            Sample("session", 0, "field_overworld", "manual_v1", "action", {0x14: 23}),
            Sample("session", 1, "battle", "manual_v1", "action", {0x14: 0}),
        ]
        result = analyze_samples(samples, [0x14])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            out = Path(temp) / "analysis"
            write_analysis(result, out)
            self.assertTrue((out / "run_report.txt").exists())
            self.assertTrue((out / "candidate_summary.csv").exists())
            self.assertTrue((out / "transitions.csv").exists())
            with self.assertRaises(FileExistsError):
                write_analysis(result, out)

    def test_multiple_sessions_keep_edges_separate_and_count_conflicts(self):
        samples = [
            Sample("s1", 0, "field_overworld", "manual_v1", "action", {0x14: 23}, 1),
            Sample("s1", 1, "field_overworld", "manual_v1", "action", {0x14: 0}, 1),
            Sample("s1", 2, "battle", "manual_v1", "action", {0x14: 0}, 2),
            Sample("s2", 0, "field_overworld", "manual_v1", "action", {0x14: 23}, 1),
            Sample("s2", 1, "battle", "manual_v1", "action", {0x14: 0}, 2),
        ]
        result = analyze_samples(samples, [0x14])
        candidate = result["candidates"][0]
        self.assertEqual(result["session_count"], 2)
        self.assertTrue(candidate["same_distinct_modes_across_sessions"])
        self.assertEqual(candidate["confusion_if_battle_mode_is_battle"]["false_positive"], 1)
        self.assertAlmostEqual(candidate["confusion_if_battle_mode_is_battle"]["false_positive_rate"], 1 / 3, delta=1e-5)
        self.assertAlmostEqual(candidate["balanced_accuracy"], 0.833333, delta=1e-5)
        s1_edges = [e for e in result["transitions"] if e["session_id"] == "s1"]
        self.assertEqual([e["event"] for e in s1_edges], ["enter_battle_candidate"])
        self.assertEqual(s1_edges[0]["label_status"], "label_conflict")

    def test_real_local_captures_when_available(self):
        project_root = Path(__file__).resolve().parents[1]
        v09 = project_root / "videos" / "nes_ram_probe_v0_9"
        v1 = project_root / "videos" / "nes_ram_verify_v1_0"
        if not (v09 / "ram_samples.npz").exists() or not (v1 / "manual_ram_events.csv").exists():
            self.skipTest("Ignored local v0.9/v1 capture artifacts are not available")
        samples = discover_sessions([v09, v1], addresses=[0x14, 0xD2])
        result = analyze_samples(samples, [0x14, 0xD2])
        candidate = next(item for item in result["candidates"] if item["address_dec"] == 0x14)
        by_source = candidate["confusion_by_label_source"]
        self.assertIn("visual_heuristic_v0_9", by_source)
        self.assertIn("manual_v1", by_source)
        self.assertEqual(by_source["visual_heuristic_v0_9"]["false_positive"], 1)
        self.assertGreater(by_source["manual_v1"]["false_positive"], 0)
        self.assertGreater(candidate["ambiguous_or_non_binary_samples"], 0)
        self.assertFalse(candidate["same_distinct_modes_across_sessions"])


if __name__ == "__main__":
    unittest.main()
