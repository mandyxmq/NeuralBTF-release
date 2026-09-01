"""Weighted latent-space graph cuts and Wang tile assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .config import WangSearchConfig
from .latent import LatentLayout


VERTICAL_EDGES = ("v0", "v1")
HORIZONTAL_EDGES = ("h0", "h1")
DIAMOND_KEYS = VERTICAL_EDGES + HORIZONTAL_EDGES
CORNER_KEYS = ("nw", "ne", "sw", "se")


@dataclass(frozen=True)
class TileEdges:
    north: str
    east: str
    south: str
    west: str

    def as_dict(self) -> dict:
        return {
            "north": self.north,
            "east": self.east,
            "south": self.south,
            "west": self.west,
        }


@dataclass
class WangTile:
    index: int
    edges: TileEdges
    latent: np.ndarray
    seam: np.ndarray


def random_crop(
    latent: np.ndarray,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    source_h, source_w, _ = latent.shape
    if height > source_h or width > source_w:
        raise ValueError(
            f"source {source_w}x{source_h} is smaller than crop {width}x{height}"
        )
    y0 = int(rng.integers(0, source_h - height + 1))
    x0 = int(rng.integers(0, source_w - width + 1))
    return np.asarray(
        latent[y0 : y0 + height, x0 : x0 + width],
        dtype=np.float32,
    ).copy()


def monotone_seam(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lowest-cost right/down path from the upper-left to lower-right corner."""
    if cost.ndim != 2 or not cost.size:
        raise ValueError("seam cost must be a non-empty 2D array")
    height, width = cost.shape
    accumulated = np.full((height, width), np.inf, dtype=np.float64)
    from_top = np.zeros((height, width), dtype=bool)
    accumulated[0, 0] = float(cost[0, 0])

    for diagonal in range(1, height + width - 1):
        y_min = max(0, diagonal - (width - 1))
        y_max = min(height - 1, diagonal)
        y = np.arange(y_min, y_max + 1)
        x = diagonal - y
        top = np.full(y.shape, np.inf, dtype=np.float64)
        left = np.full(y.shape, np.inf, dtype=np.float64)
        has_top = y > 0
        has_left = x > 0
        top[has_top] = accumulated[y[has_top] - 1, x[has_top]]
        left[has_left] = accumulated[y[has_left], x[has_left] - 1]
        choose_top = top < left
        accumulated[y, x] = cost[y, x] + np.minimum(top, left)
        from_top[y, x] = choose_top

    path = []
    y = height - 1
    x = width - 1
    while True:
        path.append((y, x))
        if y == 0 and x == 0:
            break
        if x == 0 or (y > 0 and from_top[y, x]):
            y -= 1
        else:
            x -= 1
    path.reverse()
    path_array = np.asarray(path, dtype=np.int32)

    side = np.zeros((height, width), dtype=np.float32)
    seam_x = {}
    for row, column in path:
        seam_x[row] = column
    last_column = 0
    for row in range(height):
        if row in seam_x:
            last_column = seam_x[row]
        side[row, : last_column + 1] = 1.0
    return path_array, side


def _quadrant(array: np.ndarray, name: str) -> np.ndarray:
    center_y = array.shape[0] // 2
    center_x = array.shape[1] // 2
    slices = {
        "ul": (slice(0, center_y), slice(0, center_x)),
        "ur": (slice(0, center_y), slice(center_x, array.shape[1])),
        "ll": (slice(center_y, array.shape[0]), slice(0, center_x)),
        "lr": (
            slice(center_y, array.shape[0]),
            slice(center_x, array.shape[1]),
        ),
    }
    if name not in slices:
        raise ValueError(f"unknown quadrant {name}")
    y, x = slices[name]
    return array[y, x].copy()


def _channel_cost(
    first: np.ndarray,
    second: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    if stop <= start:
        return np.zeros(first.shape[:2], dtype=np.float32)
    difference = first[..., start:stop] - second[..., start:stop]
    return np.mean(np.square(difference), axis=2, dtype=np.float32)


def _pair_maps(
    first: np.ndarray,
    second: np.ndarray,
    layout: LatentLayout,
) -> Tuple[np.ndarray, np.ndarray]:
    appearance = _channel_cost(
        first, second, 0, layout.appearance_channels
    )
    offset = _channel_cost(
        first,
        second,
        layout.appearance_channels,
        layout.total_channels,
    )
    return appearance, offset


def _corner_pairs(
    vertical: np.ndarray,
    horizontal: np.ndarray,
):
    return (
        (_quadrant(vertical, "ur"), _quadrant(horizontal, "ll"), False),
        (_quadrant(vertical, "ul"), _quadrant(horizontal, "lr"), True),
        (_quadrant(vertical, "lr"), _quadrant(horizontal, "ul"), True),
        (_quadrant(vertical, "ll"), _quadrant(horizontal, "ur"), False),
    )


def pair_loss(
    vertical: np.ndarray,
    horizontal: np.ndarray,
    layout: LatentLayout,
    weights: WangSearchConfig,
) -> np.ndarray:
    totals = np.zeros(4, dtype=np.float64)
    for first, second, flip in _corner_pairs(vertical, horizontal):
        appearance, offset = _pair_maps(first, second, layout)
        totals[0] += float(appearance.mean()) * weights.pixel_appearance
        totals[1] += float(offset.mean()) * weights.pixel_offset
        seam_cost = (
            appearance * weights.seam_appearance
            + offset * weights.seam_offset
        )
        if flip:
            seam_cost = np.flipud(seam_cost)
            appearance = np.flipud(appearance)
            offset = np.flipud(offset)
        path, _ = monotone_seam(seam_cost)
        totals[2] += (
            float(appearance[path[:, 0], path[:, 1]].mean())
            * weights.seam_appearance
        )
        totals[3] += (
            float(offset[path[:, 0], path[:, 1]].mean())
            * weights.seam_offset
        )
    return totals


def optimize_diamonds(
    latent: np.ndarray,
    *,
    height: int,
    width: int,
    layout: LatentLayout,
    config: WangSearchConfig,
    seed: int,
    progress: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    rng = np.random.default_rng(int(seed))
    current = {
        key: random_crop(latent, height, width, rng)
        for key in DIAMOND_KEYS
    }
    cache = {}
    totals = np.zeros(4, dtype=np.float64)
    for vertical in VERTICAL_EDGES:
        for horizontal in HORIZONTAL_EDGES:
            value = pair_loss(
                current[vertical], current[horizontal], layout, config
            )
            cache[(vertical, horizontal)] = value
            totals += value

    best = {key: value.copy() for key, value in current.items()}
    best_loss = float(totals.sum() / 16.0)
    history = []
    interval = max(1, config.iterations // 10)

    for iteration in range(config.iterations + 1):
        current_loss = float(totals.sum() / 16.0)
        components = totals / 16.0
        history.append(
            {
                "iteration": iteration,
                "current_loss": current_loss,
                "best_loss": best_loss,
                "pixel_appearance": float(components[0]),
                "pixel_offset": float(components[1]),
                "seam_appearance": float(components[2]),
                "seam_offset": float(components[3]),
            }
        )
        if progress and (
            iteration % interval == 0 or iteration == config.iterations
        ):
            print(
                f"    search {iteration}/{config.iterations} | "
                f"current {current_loss:.6f} | best {best_loss:.6f}"
            )
        if iteration == config.iterations:
            break

        key = DIAMOND_KEYS[int(rng.integers(0, len(DIAMOND_KEYS)))]
        proposal = random_crop(latent, height, width, rng)
        candidate = totals.copy()
        changed = {}
        partners = (
            HORIZONTAL_EDGES if key in VERTICAL_EDGES else VERTICAL_EDGES
        )
        for partner in partners:
            pair_key = (
                (key, partner) if key in VERTICAL_EDGES else (partner, key)
            )
            vertical_patch = (
                proposal if key in VERTICAL_EDGES else current[partner]
            )
            horizontal_patch = (
                current[partner] if key in VERTICAL_EDGES else proposal
            )
            value = pair_loss(
                vertical_patch, horizontal_patch, layout, config
            )
            candidate += value - cache[pair_key]
            changed[pair_key] = value

        candidate_loss = float(candidate.sum() / 16.0)
        if candidate_loss < current_loss:
            current[key] = proposal
            totals = candidate
            cache.update(changed)
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best = {
                    name: value.copy() for name, value in current.items()
                }
    return best, history


def _stitch(
    first: np.ndarray,
    second: np.ndarray,
    layout: LatentLayout,
    weights: WangSearchConfig,
    *,
    flip_vertical: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    appearance, offset = _pair_maps(first, second, layout)
    cost = (
        appearance * weights.seam_appearance
        + offset * weights.seam_offset
    )
    working_cost = np.flipud(cost) if flip_vertical else cost
    path, side = monotone_seam(working_cost)
    if flip_vertical:
        side = np.flipud(side)
        path[:, 0] = cost.shape[0] - 1 - path[:, 0]
    stitched = side[..., None] * first + (1.0 - side[..., None]) * second
    seam = np.zeros(cost.shape, dtype=bool)
    seam[path[:, 0], path[:, 1]] = True
    return stitched.astype(np.float32, copy=False), seam


def build_corner_lookup(
    diamonds: Dict[str, np.ndarray],
    layout: LatentLayout,
    weights: WangSearchConfig,
) -> Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]]:
    lookup = {}
    for vertical in VERTICAL_EDGES:
        for horizontal in HORIZONTAL_EDGES:
            v = diamonds[vertical]
            h = diamonds[horizontal]
            lookup[(vertical, horizontal, "nw")] = _stitch(
                _quadrant(v, "ur"),
                _quadrant(h, "ll"),
                layout,
                weights,
                flip_vertical=False,
            )
            lookup[(vertical, horizontal, "ne")] = _stitch(
                _quadrant(h, "lr"),
                _quadrant(v, "ul"),
                layout,
                weights,
                flip_vertical=True,
            )
            lookup[(vertical, horizontal, "sw")] = _stitch(
                _quadrant(v, "lr"),
                _quadrant(h, "ul"),
                layout,
                weights,
                flip_vertical=True,
            )
            lookup[(vertical, horizontal, "se")] = _stitch(
                _quadrant(h, "ur"),
                _quadrant(v, "ll"),
                layout,
                weights,
                flip_vertical=False,
            )
    return lookup


def _assemble_tile(
    edges: TileEdges,
    lookup: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]],
    *,
    height: int,
    width: int,
    channels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    center_y = height // 2
    center_x = width // 2
    latent = np.empty((height, width, channels), dtype=np.float32)
    seam = np.zeros((height, width), dtype=bool)
    entries = (
        (
            (slice(0, center_y), slice(0, center_x)),
            (edges.west, edges.north, "nw"),
        ),
        (
            (slice(0, center_y), slice(center_x, width)),
            (edges.east, edges.north, "ne"),
        ),
        (
            (slice(center_y, height), slice(0, center_x)),
            (edges.west, edges.south, "sw"),
        ),
        (
            (slice(center_y, height), slice(center_x, width)),
            (edges.east, edges.south, "se"),
        ),
    )
    for (ys, xs), key in entries:
        patch, patch_seam = lookup[key]
        latent[ys, xs] = patch
        seam[ys, xs] = patch_seam
    return latent, seam


def build_tile_set(
    lookup: Dict[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray]],
    *,
    height: int,
    width: int,
    channels: int,
) -> List[WangTile]:
    result = []
    index = 0
    for north in HORIZONTAL_EDGES:
        for east in VERTICAL_EDGES:
            for south in HORIZONTAL_EDGES:
                for west in VERTICAL_EDGES:
                    edges = TileEdges(north, east, south, west)
                    latent, seam = _assemble_tile(
                        edges,
                        lookup,
                        height=height,
                        width=width,
                        channels=channels,
                    )
                    result.append(WangTile(index, edges, latent, seam))
                    index += 1
    return result


def matching_tile_indices(
    edges: Iterable[TileEdges],
    *,
    north: Optional[str] = None,
    west: Optional[str] = None,
    south: Optional[str] = None,
    east: Optional[str] = None,
) -> List[int]:
    return [
        index
        for index, value in enumerate(edges)
        if (north is None or value.north == north)
        and (west is None or value.west == west)
        and (south is None or value.south == south)
        and (east is None or value.east == east)
    ]


def generate_tile_grid(
    edges: List[TileEdges],
    *,
    rows: int,
    columns: int,
    seed: int,
    periodic: bool,
) -> List[List[int]]:
    """Generate a matching Wang grid, optionally enforcing toroidal boundaries."""
    rng = np.random.default_rng(int(seed))
    grid = [[-1 for _ in range(columns)] for _ in range(rows)]

    def solve(position: int) -> bool:
        if position == rows * columns:
            return True
        row, column = divmod(position, columns)
        north = (
            edges[grid[row - 1][column]].south if row > 0 else None
        )
        west = (
            edges[grid[row][column - 1]].east if column > 0 else None
        )
        south = (
            edges[grid[0][column]].north
            if periodic and row == rows - 1
            else None
        )
        east = (
            edges[grid[row][0]].west
            if periodic and column == columns - 1
            else None
        )
        candidates = matching_tile_indices(
            edges,
            north=north,
            west=west,
            south=south,
            east=east,
        )
        rng.shuffle(candidates)
        for candidate in candidates:
            grid[row][column] = candidate
            if solve(position + 1):
                return True
        grid[row][column] = -1
        return False

    if not solve(0):
        raise RuntimeError("could not construct a matching Wang tile grid")
    return grid


def assemble_tile_grid(
    tiles: List[np.ndarray],
    grid: List[List[int]],
) -> np.ndarray:
    if not tiles or not grid or not grid[0]:
        raise ValueError("tiles and grid cannot be empty")
    height, width, channels = tiles[0].shape
    output = np.empty(
        (len(grid) * height, len(grid[0]) * width, channels),
        dtype=np.float32,
    )
    for row, indices in enumerate(grid):
        for column, index in enumerate(indices):
            output[
                row * height : (row + 1) * height,
                column * width : (column + 1) * width,
            ] = tiles[index]
    return output


def write_tile_grid(
    tile_paths: Dict[int, Path],
    grid: List[List[int]],
    output_path: Path,
) -> Tuple[int, int, int]:
    """Assemble a potentially large tile grid directly into a NumPy memmap."""
    if not tile_paths or not grid or not grid[0]:
        raise ValueError("tile paths and grid cannot be empty")
    first = np.load(next(iter(tile_paths.values())), mmap_mode="r")
    if first.ndim != 3:
        raise ValueError(f"expected HWC tile arrays, got {first.shape}")
    tile_h, tile_w, channels = first.shape
    shape = (len(grid) * tile_h, len(grid[0]) * tile_w, channels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=shape
    )
    for row, indices in enumerate(grid):
        if len(indices) != len(grid[0]):
            raise ValueError("tile grid rows must have equal length")
        for column, index in enumerate(indices):
            if index not in tile_paths:
                raise ValueError(f"tile grid references missing tile {index}")
            tile = np.load(tile_paths[index], mmap_mode="r")
            if tile.shape != (tile_h, tile_w, channels):
                raise ValueError(
                    f"tile {index} shape {tile.shape} does not match "
                    f"{(tile_h, tile_w, channels)}"
                )
            output[
                row * tile_h : (row + 1) * tile_h,
                column * tile_w : (column + 1) * tile_w,
            ] = tile
    output.flush()
    del output
    return shape
