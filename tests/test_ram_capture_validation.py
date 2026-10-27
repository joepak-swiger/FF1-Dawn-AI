import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_ram_capture import validate_run


def make_run(root: Path, session_count: int = 3) -> Path:
    run_dir = root / "capture"
    run_dir.mkdir()
    sessions = []
    for index in range(1, session_count + 1):
        session_id = f"session_{index:02d}"
        session_dir = run_dir / session_id
        (session_dir / "frames").mkdir(parents=True)
        (session_dir / "frames" / "frame_00000.png").write_bytes(b"proof")
        np.savez_compressed(
            session_dir / "ram_samples.npz",
            rams=np.zeros((2, 16), dtype=np.uint8),
            battle_like=np.array([False, True]),
        )
        with (session_dir / "probe_events.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["step"])
            writer.writeheader()
            writer.writerows([{"step": 0}, {"step": 1}])
        (session_dir / "metrics.json").write_text(json.dumps({"status": "complete", "steps_recorded": 2}))
        sessions.append({"session_id": session_id})
    (run_dir / "metrics.json").write_text(json.dumps({"status": "complete", "episodes": sessions}))
    return run_dir


class RamCaptureValidationTests(unittest.TestCase):
    def test_accepts_complete_capture_and_returns_independent_sessions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            run_dir = make_run(Path(temp))
            self.assertEqual(validate_run(run_dir), ["session_01", "session_02", "session_03"])

    def test_rejects_failed_or_partial_run_before_analysis(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            run_dir = make_run(Path(temp))
            metrics_path = run_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["status"] = "failed"
            metrics_path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "status is 'failed'"):
                validate_run(run_dir)

    def test_rejects_missing_session_data_and_sample_count_mismatch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            run_dir = make_run(Path(temp))
            (run_dir / "session_02" / "ram_samples.npz").unlink()
            with self.assertRaisesRegex(ValueError, "missing metrics, RAM samples"):
                validate_run(run_dir)

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            run_dir = make_run(Path(temp))
            metrics_path = run_dir / "session_01" / "metrics.json"
            metrics_path.write_text(json.dumps({"status": "complete", "steps_recorded": 9}))
            with self.assertRaisesRegex(ValueError, "sample count does not match"):
                validate_run(run_dir)

    def test_validates_real_local_three_session_capture_when_present(self):
        project = Path(__file__).resolve().parents[1]
        captures = project / "videos" / "nes_ram_auto_capture"
        if not captures.is_dir():
            self.skipTest("Ignored automated capture output is not present")
        candidates = [path for path in captures.iterdir() if path.is_dir() and (path / "metrics.json").is_file()]
        if not candidates:
            self.skipTest("No complete local automated capture run is available")
        run = max(candidates, key=lambda path: path.stat().st_mtime)
        self.assertEqual(len(validate_run(run)), 3)


if __name__ == "__main__":
    unittest.main()
