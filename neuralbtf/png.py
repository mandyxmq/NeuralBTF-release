"""Minimal PNG writer (8-bit RGB), for the qualitative comparison images.

The reference evaluation script used Mitsuba's ``Bitmap`` to write these; the
standard library can do it in a few lines, so the release keeps its three
dependencies.  EXR (see :mod:`neuralbtf.exr`) is still the format for anything
you intend to measure -- PNG is display-referred and 8 bits deep.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """The sRGB transfer function, applied to values in ``[0, 1]``."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def to_display(image: np.ndarray, exposure_stops: float = 0.0) -> np.ndarray:
    """Linear radiance -> 8-bit sRGB, clipping highlights."""
    scaled = np.clip(np.asarray(image, dtype=np.float32), 0.0, None) * (2.0**exposure_stops)
    return np.round(linear_to_srgb(scaled) * 255.0).astype(np.uint8)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def write_png(path: str, image: np.ndarray, exposure_stops: float = 0.0,
              srgb: bool = True) -> None:
    """Write ``(H, W, 3)`` data to ``path``.

    With ``srgb`` (the default) the input is linear radiance and is exposed and
    sRGB-encoded; otherwise it is taken to be display values in ``[0, 1]``.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got shape {image.shape}")

    if srgb:
        rgb = to_display(image, exposure_stops)
    else:
        rgb = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

    height, width, _ = rgb.shape
    # Each scanline is prefixed with its filter type; 0 means "no filtering".
    raw = np.concatenate([np.zeros((height, 1), np.uint8), rgb.reshape(height, -1)], axis=1)

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        fh.write(_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)))
        fh.write(_chunk(b"IEND", b""))
