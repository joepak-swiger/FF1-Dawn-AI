import unittest

import numpy as np

from ff1_dawn_ai.gba_position import analyze_ram_position_candidates, reversible_offsets


class GbaPositionAnalysisTests(unittest.TestCase):
    def test_direction_candidates_require_repeatability_and_matched_noop(self):
        baseline = [np.array([10, 20, 30, 40], dtype=np.uint8) for _ in range(3)]
        controls = {
            "NOOP": [np.array([10, 20, 31 + i, 40], dtype=np.uint8) for i in range(3)],
            "UP": [np.array([11, 20, 31 + i, 40 + i], dtype=np.uint8) for i in range(3)],
            "DOWN": [np.array([9, 20, 31 + i, 40 + i], dtype=np.uint8) for i in range(3)],
            "LEFT": [np.array([10, 21, 31 + i, 40 + i], dtype=np.uint8) for i in range(3)],
            "RIGHT": [np.array([10, 19, 31 + i, 40 + i], dtype=np.uint8) for i in range(3)],
        }
        result = analyze_ram_position_candidates(baseline, controls)
        candidates = {item["offset"]: item for item in result["candidate_offsets"]}
        self.assertIn(0, candidates)
        self.assertIn(1, candidates)
        self.assertNotIn(2, candidates)  # also changes during no-op
        self.assertNotIn(3, candidates)  # unstable under direction repeats
        self.assertEqual(candidates[0]["classification"], "unverified_direction_correlated_byte")

    def test_reversal_requires_matched_idle_and_exact_repeatable_values(self):
        start = [np.array([1, 2, 3], dtype=np.uint8) for _ in range(3)]
        returned = [np.array([1, 2, 4], dtype=np.uint8) for _ in range(3)]
        idle = [np.array([1, 2, 3], dtype=np.uint8) for _ in range(3)]
        outbound = [np.array([9, 8, 4], dtype=np.uint8) for _ in range(3)]
        self.assertEqual(reversible_offsets(start, outbound, returned, idle), [0, 1])

    def test_candidate_analysis_rejects_mismatched_repeat_counts(self):
        base = [np.zeros(4, dtype=np.uint8) for _ in range(3)]
        controls = {name: [row.copy() for row in base] for name in ("NOOP", "UP", "DOWN", "LEFT", "RIGHT")}
        controls["UP"].pop()
        with self.assertRaisesRegex(ValueError, "same repeats"):
            analyze_ram_position_candidates(base, controls)


if __name__ == "__main__":
    unittest.main()
