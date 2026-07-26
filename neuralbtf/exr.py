"""Minimal OpenEXR reader/writer (uncompressed, 32-bit float, RGB).

The reference code used Mitsuba's ``Bitmap`` for this one job.  Images are written
here with the standard library instead, so training needs no renderer installed.
Files are plain scanline EXRs and open in Mitsuba, OpenCV, tev, Blender, etc.
"""

from __future__ import annotations

import struct

import numpy as np

_MAGIC = 20000630
_VERSION = 2
_PIXEL_TYPE_FLOAT = 2
_NO_COMPRESSION = 0
_CHANNELS = ("B", "G", "R")  # EXR requires channels in alphabetical order


def _attr(name: str, type_name: str, payload: bytes) -> bytes:
    return (name.encode() + b"\0" + type_name.encode() + b"\0"
            + struct.pack("<i", len(payload)) + payload)


def _header(width: int, height: int) -> bytes:
    channels = b"".join(
        c.encode() + b"\0" + struct.pack("<iBxxxii", _PIXEL_TYPE_FLOAT, 0, 1, 1)
        for c in _CHANNELS
    ) + b"\0"
    box = struct.pack("<iiii", 0, 0, width - 1, height - 1)

    return b"".join([
        struct.pack("<ii", _MAGIC, _VERSION),
        _attr("channels", "chlist", channels),
        _attr("compression", "compression", bytes([_NO_COMPRESSION])),
        _attr("dataWindow", "box2i", box),
        _attr("displayWindow", "box2i", box),
        _attr("lineOrder", "lineOrder", bytes([0])),          # increasing y
        _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attr("screenWindowWidth", "float", struct.pack("<f", 1.0)),
        b"\0",
    ])


def write_exr(path: str, image: np.ndarray) -> None:
    """Write ``(H, W, 3)`` RGB (or ``(H, W)`` greyscale) float data to ``path``."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got shape {image.shape}")

    height, width, _ = image.shape
    header = _header(width, height)

    # One uncompressed scanline per chunk: [int32 y][int32 nbytes][B...][G...][R...]
    line_bytes = width * 4 * len(_CHANNELS)
    offset = len(header) + 8 * height
    offsets = [offset + y * (8 + line_bytes) for y in range(height)]

    # (H, W, RGB) -> (H, BGR, W) so each scanline is channel-planar in EXR order.
    planar = np.ascontiguousarray(image[:, :, ::-1].transpose(0, 2, 1))

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack(f"<{height}Q", *offsets))
        for y in range(height):
            fh.write(struct.pack("<ii", y, line_bytes))
            fh.write(planar[y].tobytes())


def read_exr(path: str) -> np.ndarray:
    """Read an uncompressed 32-bit float EXR written by :func:`write_exr`.

    Only the subset of the format produced above is supported; use a full EXR
    library for arbitrary files.
    """
    with open(path, "rb") as fh:
        data = fh.read()

    magic, version = struct.unpack_from("<ii", data, 0)
    if magic != _MAGIC:
        raise ValueError(f"{path}: not an OpenEXR file")
    if version & 0x200:
        raise ValueError(f"{path}: tiled EXR files are not supported")

    pos = 8
    attrs: dict[str, bytes] = {}
    while data[pos] != 0:
        name_end = data.index(b"\0", pos)
        name = data[pos:name_end].decode()
        type_end = data.index(b"\0", name_end + 1)
        size = struct.unpack_from("<i", data, type_end + 1)[0]
        payload_start = type_end + 5
        attrs[name] = data[payload_start:payload_start + size]
        pos = payload_start + size
    pos += 1

    if attrs["compression"][0] != _NO_COMPRESSION:
        raise ValueError(f"{path}: only uncompressed EXR files are supported")

    x_min, y_min, x_max, y_max = struct.unpack("<iiii", attrs["dataWindow"])
    width, height = x_max - x_min + 1, y_max - y_min + 1

    names, cursor = [], 0
    chlist = attrs["channels"]
    while cursor < len(chlist) - 1:
        end = chlist.index(b"\0", cursor)
        names.append(chlist[cursor:end].decode())
        pixel_type = struct.unpack_from("<i", chlist, end + 1)[0]
        if pixel_type != _PIXEL_TYPE_FLOAT:
            raise ValueError(f"{path}: only 32-bit float channels are supported")
        cursor = end + 1 + 16

    pos += 8 * height  # skip the offset table; scanlines follow in order
    out = np.empty((height, len(names), width), dtype=np.float32)
    for y in range(height):
        pos += 8  # per-scanline [y][nbytes]
        count = width * len(names)
        out[y] = np.frombuffer(data, np.float32, count, pos).reshape(len(names), width)
        pos += count * 4

    rgb = [out[:, names.index(c), :] for c in ("R", "G", "B") if c in names]
    return np.stack(rgb, axis=-1)
