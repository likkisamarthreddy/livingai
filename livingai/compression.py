"""Checkpoint compression.

Agent state blobs (long conversation histories, retrieved documents) can be
large. Checkpoints are compressed before storage to hit the 60-80% size
reduction target from the Technical Execution Plan.

The plan names *zstd*. To keep the core **zero-dependency**, the default
:class:`ZlibCompressor` uses the standard library. A zstd-backed compressor is a
drop-in replacement implementing the same :class:`Compressor` protocol — the
engine only depends on the protocol, never a concrete codec.

A one-byte codec header is prepended to every payload so blobs are
self-describing and future codecs can coexist with old data.
"""

from __future__ import annotations

import zlib
from typing import Protocol, runtime_checkable


__all__ = ["Compressor", "ZlibCompressor", "NoopCompressor"]


# Codec identifiers stored as the first byte of every compressed blob.
_CODEC_NONE = 0x00
_CODEC_ZLIB = 0x01


@runtime_checkable
class Compressor(Protocol):
    """Reversible byte transform used for checkpoint blobs."""

    def compress(self, data: bytes) -> bytes: ...

    def decompress(self, blob: bytes) -> bytes: ...


class NoopCompressor:
    """Pass-through codec, useful for benchmarking and already-small blobs."""

    def compress(self, data: bytes) -> bytes:
        return bytes([_CODEC_NONE]) + data

    def decompress(self, blob: bytes) -> bytes:
        return _strip_and_dispatch(blob)


class ZlibCompressor:
    """Standard-library zlib compressor (default).

    Args:
        level: zlib compression level 0-9. Level 6 is a good size/speed balance
            and is well within the 50 ms checkpoint budget for typical blobs.
    """

    def __init__(self, level: int = 6) -> None:
        if not 0 <= level <= 9:
            raise ValueError("zlib level must be between 0 and 9")
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return bytes([_CODEC_ZLIB]) + zlib.compress(data, self.level)

    def decompress(self, blob: bytes) -> bytes:
        return _strip_and_dispatch(blob)


def _strip_and_dispatch(blob: bytes) -> bytes:
    """Decode a self-describing blob regardless of which codec wrote it."""
    if not blob:
        return b""
    codec, payload = blob[0], blob[1:]
    if codec == _CODEC_NONE:
        return payload
    if codec == _CODEC_ZLIB:
        return zlib.decompress(payload)
    raise ValueError(f"Unknown checkpoint codec byte: {codec:#04x}")
