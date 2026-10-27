import struct
import unittest
import uuid
import zlib
import tempfile
from pathlib import Path

import numpy as np
from ff1_dawn_ai.gba_feasibility import inspect_file, png_chunks
from scripts.gba_feasibility_probe import _save_trace_frame


def make_chunk(name: bytes, payload: bytes) -> bytes:
    body = name + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


class GbaStateFormatTests(unittest.TestCase):
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
            trace = (output / "party_setup_trace.csv").read_text(encoding="utf-8")
            self.assertIn("A,released,0.015625", trace)

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
