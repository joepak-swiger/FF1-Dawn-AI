import struct
import unittest
import uuid
import zlib
from pathlib import Path

from ff1_dawn_ai.gba_feasibility import inspect_file, png_chunks


def make_chunk(name: bytes, payload: bytes) -> bytes:
    body = name + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


class GbaStateFormatTests(unittest.TestCase):
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
