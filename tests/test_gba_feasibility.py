import struct
import json
import sys
import unittest
import uuid
import zlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from ff1_dawn_ai.gba_feasibility import inspect_file, png_chunks
from scripts.gba_feasibility_probe import (
    GAME,
    _initial_milestones,
    _save_trace_frame,
    _summarize_party_trace,
    runtime_smoke,
)


def make_chunk(name: bytes, payload: bytes) -> bytes:
    body = name + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


class GbaStateFormatTests(unittest.TestCase):
    def test_synthetic_runtime_reports_separate_milestones_and_partial_trace(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            project = Path(temp)
            integration = project / "custom_integrations" / GAME
            integration.mkdir(parents=True)
            (integration / "rom.gba").write_bytes(b"synthetic fixture")
            output = project / "run"
            output.mkdir()

            class FakeEnv:
                buttons = ["A", "B", "START", "SELECT", "DOWN"]

                def __init__(self, fail_after=None):
                    self.action_space = SimpleNamespace(shape=(len(self.buttons),))
                    self.counter = 0
                    self.fail_after = fail_after
                    self.em = SimpleNamespace(get_state=self.get_state, set_state=self.set_state)

                def observation(self):
                    return np.full((160, 240, 3), self.counter % 256, dtype=np.uint8)

                def reset(self):
                    self.counter = 0
                    return self.observation(), {}

                def step(self, _action):
                    self.counter += 1
                    if self.fail_after is not None and self.counter >= self.fail_after:
                        raise RuntimeError("synthetic input failure")
                    return self.observation(), 0.0, False, False, {}

                def get_state(self):
                    return self.counter.to_bytes(8, "little")

                def set_state(self, state):
                    self.counter = int.from_bytes(state, "little")

                def close(self):
                    pass

            fake_env = FakeEnv()
            fake_retro = SimpleNamespace(
                data=SimpleNamespace(
                    Integrations=SimpleNamespace(ALL=0, add_custom_path=lambda _path: None)
                ),
                State=SimpleNamespace(NONE=0),
                make=lambda **_kwargs: fake_env,
            )
            with patch("scripts.gba_feasibility_probe.PROJECT_ROOT", project), patch.dict(sys.modules, {"stable_retro": fake_retro}):
                report = runtime_smoke(output, automate_party_setup=True)

            milestones = report["milestones"]
            self.assertEqual(report["status"], "passed")
            self.assertEqual(milestones["rom_integration_boots"]["status"], "partially_verified")
            self.assertEqual(milestones["input_controls_function"]["status"], "partially_verified")
            self.assertEqual(milestones["party_creation_works"]["status"], "not_verified")
            self.assertEqual(milestones["playable_scene_reached"]["status"], "not_verified")
            self.assertEqual(milestones["native_state_serialized"]["status"], "passed")
            self.assertEqual(milestones["state_restored_after_reset"]["status"], "passed")
            self.assertEqual(milestones["gameplay_resumes_correctly"]["status"], "not_verified")
            self.assertEqual(report["party_setup_trace_screenshots"], 95)
            self.assertTrue((output / "party_setup_trace.csv").is_file())
            self.assertEqual(len(report["party_setup_contact_sheets"]), 2)
            self.assertTrue((output / "runtime_progress.json").is_file())

            failed_env = FakeEnv(fail_after=370)
            fake_retro.make = lambda **_kwargs: failed_env
            failed_output = project / "failed_run"
            failed_output.mkdir()
            with patch("scripts.gba_feasibility_probe.PROJECT_ROOT", project), patch.dict(sys.modules, {"stable_retro": fake_retro}):
                with self.assertRaisesRegex(RuntimeError, "synthetic input failure"):
                    runtime_smoke(failed_output, automate_party_setup=True)
            progress = json.loads((failed_output / "runtime_progress.json").read_text(encoding="utf-8"))
            self.assertIn("synthetic input failure", progress["failure"])
            self.assertTrue((failed_output / "party_setup_trace.csv").is_file())
            self.assertTrue((failed_output / "party_setup_trace_summary.json").is_file())

    def test_party_trace_saves_input_frame_and_changed_pixel_fraction(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            output = Path(temp)
            before = np.zeros((8, 8, 3), dtype=np.uint8)
            after = before.copy()
            after[0, 0] = 255
            rows = []
            _save_trace_frame(output, rows, 0, "A", "released", after, before)
            self.assertEqual(rows[0]["changed_pixel_fraction"], "0.015625")
            self.assertTrue((output / rows[0]["screenshot"]).is_file())
            _save_trace_frame(output, rows, 1, "DOWN", "released", after, after)
            self.assertEqual(rows[0]["screen_id"], rows[1]["screen_id"])
            trace = (output / "party_setup_trace.csv").read_text(encoding="utf-8")
            self.assertIn("A,released,0.015625", trace)
            summary = _summarize_party_trace(output, rows)
            self.assertEqual(summary["ordered_action_screens"], 2)
            self.assertEqual(len(summary["consecutive_exact_duplicate_screens"]), 1)
            self.assertTrue((output / summary["contact_sheets"][0]).is_file())
            self.assertFalse(summary["screen_semantics_classified"])

    def test_runtime_milestones_begin_unverified(self):
        milestones = _initial_milestones()
        self.assertEqual(milestones["rom_integration_boots"]["status"], "not_run")
        self.assertEqual(milestones["party_creation_works"]["status"], "not_verified")
        self.assertEqual(milestones["playable_scene_reached"]["status"], "not_verified")
        self.assertEqual(milestones["gameplay_resumes_correctly"]["status"], "not_verified")

    def test_reads_png_chunk_inventory_without_decoding_payloads(self):
        sample = (
            b"\x89PNG\r\n\x1a\n"
            + make_chunk(b"IHDR", b"header")
            + make_chunk(b"gbAs", b"state payload")
            + make_chunk(b"IEND", b"")
        )
        self.assertEqual(png_chunks(sample), [("IHDR", 6), ("gbAs", 13), ("IEND", 0)])

    def test_rejects_bad_crc_and_trailing_data(self):
        valid = b"\x89PNG\r\n\x1a\n" + make_chunk(b"IEND", b"")
        corrupted = valid[:-1] + bytes([valid[-1] ^ 1])
        with self.assertRaisesRegex(ValueError, "bad CRC"):
            png_chunks(corrupted)
        with self.assertRaisesRegex(ValueError, "after PNG IEND"):
            png_chunks(valid + b"extra")

    def test_inspect_file_reports_size_and_chunk_metadata(self):
        sample = b"\x89PNG\r\n\x1a\n" + make_chunk(b"gbAs", b"x") + make_chunk(b"IEND", b"")
        path = Path(__file__).resolve().parent / f".fixture_{uuid.uuid4().hex}.ss0"
        try:
            path.write_bytes(sample)
            result = inspect_file(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertTrue(result["is_png_container"])
        self.assertEqual(result["size_bytes"], len(sample))
        self.assertEqual([c["type"] for c in result["png_chunks"]], ["gbAs", "IEND"])


if __name__ == "__main__":
    unittest.main()
