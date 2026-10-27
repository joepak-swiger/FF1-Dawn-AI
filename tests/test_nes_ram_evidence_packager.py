import csv
import unittest
import uuid
from pathlib import Path

from PIL import Image
import numpy as np

from scripts.package_nes_ram_evidence import (
    candidate_addresses,
    label_error_timing,
    make_contact_sheet,
    transition_counts,
)


class NesEvidencePackagerTests(unittest.TestCase):
    def test_candidate_selection_keeps_top_candidates_and_watch_list(self):
        rows = [{"address_hex": f"0x{i:04X}"} for i in range(16, 32)]
        selected = candidate_addresses(rows, limit=2)
        self.assertEqual(selected[:2], ["0x0010", "0x0011"])
        self.assertTrue({"0x0014", "0x00D2", "0x0270"}.issubset(selected))

    def test_transition_summary_only_contains_selected_candidates(self):
        path = Path(__file__).resolve().parent / f".fixture_{uuid.uuid4().hex}.csv"
        try:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["session_id", "address_hex", "event", "label_status"])
                writer.writeheader()
                writer.writerows([
                    {"session_id": "run/session_01", "address_hex": "0x0014", "event": "enter_battle_candidate", "label_status": "agrees"},
                    {"session_id": "run/session_01", "address_hex": "0x0014", "event": "exit_battle_candidate", "label_status": "label_conflict"},
                    {"session_id": "run/session_01", "address_hex": "0x00D2", "event": "value_change", "label_status": "agrees"},
                ])
            result = transition_counts(path, {"0x0014"})
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(result["0x0014"]["entry_edges"], 1)
        self.assertEqual(result["0x0014"]["exit_edges"], 1)
        self.assertEqual(result["0x0014"]["edges_conflicting_with_pixel_proxy"], 1)
        self.assertNotIn("0x00D2", result)

    def test_contact_sheet_is_resized_and_does_not_return_source_paths(self):
        source = Path(__file__).resolve().parent / f".fixture_{uuid.uuid4().hex}.png"
        output = Path(__file__).resolve().parent / f".fixture_{uuid.uuid4().hex}.png"
        try:
            Image.new("RGB", (240, 224), "green").save(source)
            manifest = make_contact_sheet([
                {
                    "session_id": "session_01",
                    "transition_label": "first battle-like entry",
                    "transition_step": 93,
                    "saved_frame_step": 120,
                    "source_path": source,
                }
            ], output)
            with Image.open(output) as sheet:
                self.assertEqual(sheet.size, (216, 87))
            self.assertNotIn("source_path", manifest[0])
        finally:
            source.unlink(missing_ok=True)
            output.unlink(missing_ok=True)

    def test_disagreement_timing_separates_transition_edges_from_stable_samples(self):
        rams = np.zeros((8, 32), dtype=np.uint8)
        labels = np.array([False, False, True, True, True, True, False, False])
        rams[:, 20] = np.array([23, 23, 0, 0, 0, 23, 23, 23], dtype=np.uint8)
        result = label_error_timing(rams, labels, 20, 0, [2, 6], tolerance=1)
        self.assertEqual(result["disagreement_samples"], 1)
        self.assertEqual(result["within_transition_tolerance_samples"], 1)
        self.assertEqual(result["away_from_transition_samples"], 0)


if __name__ == "__main__":
    unittest.main()
