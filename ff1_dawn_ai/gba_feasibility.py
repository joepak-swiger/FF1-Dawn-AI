"""Read-only format inspection helpers for local GBA state artifacts."""

from __future__ import annotations

import struct
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_chunks(data: bytes) -> list[tuple[str, int]]:
    """Return PNG chunk names and payload lengths, validating framing and CRCs."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    chunks: list[tuple[str, int]] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG chunk header or CRC")
        length = struct.unpack_from(">I", data, offset)[0]
        name_bytes = data[offset + 4 : offset + 8]
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("PNG chunk type is not ASCII") from exc
        end = offset + 12 + length
        if end > len(data):
            raise ValueError(f"truncated PNG chunk {name}")
        expected_crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        actual_crc = zlib_crc32(name_bytes + data[offset + 8 : offset + 8 + length])
        if expected_crc != actual_crc:
            raise ValueError(f"PNG chunk {name} has a bad CRC")
        chunks.append((name, length))
        offset = end
        if name == "IEND":
            if offset != len(data):
                raise ValueError("unexpected bytes after PNG IEND")
            return chunks
    raise ValueError("PNG is missing IEND")


def zlib_crc32(data: bytes) -> int:
    # Kept as a tiny wrapper so this module depends only on the standard library.
    import zlib

    return zlib.crc32(data) & 0xFFFFFFFF


def inspect_file(path: Path) -> dict:
    """Summarize a state/save file without interpreting or exporting its data."""
    path = Path(path)
    data = path.read_bytes()
    result = {
        "name": path.name,
        "size_bytes": len(data),
        "prefix_hex": data[:8].hex(),
        "is_png_container": data.startswith(PNG_SIGNATURE),
    }
    if result["is_png_container"]:
        result["png_chunks"] = [
            {"type": name, "payload_bytes": size} for name, size in png_chunks(data)
        ]
    return result
