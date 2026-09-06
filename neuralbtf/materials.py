"""The 17 captured materials, and the crop each one is trained and scored on.

The capture is divided into a grid of square tiles of ``crop_size`` pixels, indexed
row-major with four tiles per row, and one tile per material is used.  The tile was
picked by hand: in focus, and free of capture artefacts.  ``training/train_all.py``
and ``training/evaluate_all.py`` share this table so training and evaluation always
look at the same pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import Crop

TILES_PER_ROW = 4

TRAIN_TEMPLATE = "real_{name}_full_factor1.hdf5"    # dense direction sweep
TEST_TEMPLATE = "real_{name}_random_factor1.hdf5"   # held-out directions


@dataclass(frozen=True)
class Material:
    name: str
    crop_size: int   # width and height of the crop, in pixels
    crop_index: int  # index into the crop_size grid of tiles, row-major, 4 per row


MATERIALS = [
    Material("red_leather_08", 350, 6),
    Material("sari_05", 512, 6),
    Material("olive_green_velvet_01", 512, 6),
    Material("light_grey_satin_02", 350, 6),
    Material("green_curtain_01", 512, 6),
    Material("dark_blue_cloth_01", 512, 6),
    Material("goldflower_blue_cloth_01", 512, 6),
    Material("circles_01", 512, 6),
    Material("brown_velvet_tiles_01", 512, 6),
    Material("pink_hand_towel_02", 450, 6),
    Material("yellow_vase_pattern_01", 512, 3),
    Material("red_gold_cloudy_temple_01", 512, 13),
    Material("curvy_browns_01", 512, 7),
    Material("trees_on_yellow_01", 512, 6),
    Material("gold_flowers_on_stripes_01", 500, 6),
    Material("vanilla_flowers_on_chevrons_01", 512, 6),
    Material("silky_smooth_green_01", 400, 6),
]

NAMES = [m.name for m in MATERIALS]


def get(material: str | int) -> Material:
    """Look a material up by index or by name."""
    if isinstance(material, int):
        if not 0 <= material < len(MATERIALS):
            raise IndexError(f"material index {material} out of range [0, {len(MATERIALS) - 1}]")
        return MATERIALS[material]
    for m in MATERIALS:
        if m.name == material:
            return m
    raise KeyError(f"unknown material {material!r}; known: {', '.join(NAMES)}")


def tile_crop(crop_size: int, crop_index: int) -> Crop:
    """The ``crop_index``-th tile of ``crop_size`` pixels, row-major, 4 per row."""
    xstart = (crop_index % TILES_PER_ROW) * crop_size
    ystart = (crop_index // TILES_PER_ROW) * crop_size
    return Crop(xstart, ystart, crop_size, crop_size)


def crop_for(material: str | int, crop_size: int = 0, crop_index: int | None = None) -> Crop:
    """The crop of ``material``, with optional overrides for size and tile."""
    m = get(material)
    return tile_crop(crop_size or m.crop_size,
                     m.crop_index if crop_index is None else crop_index)


def full_capture_settings(width: int, height: int | None = None) -> dict:
    """Crop and texture resolutions for fitting a *whole* capture ("2k").

    The pyramid is pinned to the capture's own size -- finest level at the pixel
    grid, coarsest a quarter of it, and the neural offset's feature texture at a
    sixteenth -- so a 2048² capture trains 512 -> 2048 and a 1400² one 350 -> 1400.
    The tiles in :data:`MATERIALS` are a quarter of their capture, which is why the
    coarsest level here is the resolution the tiled runs fit at.
    """
    height = width if height is None else height
    base = max(width, height)
    if base % 4:
        raise ValueError(f"capture size {width}x{height} does not divide into a pyramid; "
                         "the finest level must be a multiple of 4")
    return {
        "crop": Crop(0, 0, width, height),
        "texture_res": base,
        "min_res": base // 4,
        "off_texture_res": base // 16,  # rounds down, as the reference driver's tile//4 did
    }
