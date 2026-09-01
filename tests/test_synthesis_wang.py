#!/usr/bin/env python3
"""CPU checks for periodicity, graph cuts, and Wang compatibility."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralbtf.synthesis.config import WangSearchConfig  # noqa: E402
from neuralbtf.synthesis.latent import LatentLayout  # noqa: E402
from neuralbtf.synthesis.periodicity import analyze_periodicity  # noqa: E402
from neuralbtf.synthesis.wang import (assemble_tile_grid,  # noqa: E402
                                      build_corner_lookup, build_tile_set,
                                      generate_tile_grid, monotone_seam,
                                      optimize_diamonds)


def check(name, function) -> None:
    function()
    print(f"ok    {name}")


def graph_cut_is_monotone() -> None:
    cost = np.ones((7, 9), dtype=np.float32)
    cost[0, :] = 0.1
    cost[:, -1] = 0.1
    path, side = monotone_seam(cost)
    assert tuple(path[0]) == (0, 0)
    assert tuple(path[-1]) == (6, 8)
    steps = np.diff(path, axis=0)
    assert np.all((steps == (1, 0)).all(axis=1) | (steps == (0, 1)).all(axis=1))
    assert side.shape == cost.shape
    assert set(np.unique(side)).issubset({0.0, 1.0})


def complete_tile_set_and_periodic_grid() -> None:
    rng = np.random.default_rng(2)
    latent = rng.normal(size=(24, 28, 6)).astype(np.float32)
    layout = LatentLayout(
        appearance_channels=4,
        offset_channels=2,
        channels_per_level=2,
        appearance_resolutions=(8, 16),
        offset_mode="2d",
    )
    search = WangSearchConfig(iterations=2)
    diamonds, history = optimize_diamonds(
        latent,
        height=8,
        width=12,
        layout=layout,
        config=search,
        seed=3,
        progress=False,
    )
    assert len(diamonds) == 4
    assert len(history) == 3
    lookup = build_corner_lookup(diamonds, layout, search)
    tiles = build_tile_set(lookup, height=8, width=12, channels=6)
    assert len(tiles) == 16
    assert all(tile.latent.shape == (8, 12, 6) for tile in tiles)
    assert all(tile.seam.shape == (8, 12) for tile in tiles)

    edges = [tile.edges for tile in tiles]
    grid = generate_tile_grid(
        edges, rows=3, columns=4, seed=5, periodic=True
    )
    for row in range(3):
        for column in range(4):
            current = edges[grid[row][column]]
            right = edges[grid[row][(column + 1) % 4]]
            below = edges[grid[(row + 1) % 3][column]]
            assert current.east == right.west
            assert current.south == below.north
    assembled = assemble_tile_grid([tile.latent for tile in tiles], grid)
    assert assembled.shape == (24, 48, 6)


def periodic_signal_is_detected() -> None:
    pattern = np.tile(
        np.array([0, 0, 1, 1, 2, 2, 1, 1], dtype=np.float32),
        8,
    )
    image = pattern[:, None] + pattern[None, :]
    latent = np.repeat(image[..., None], 3, axis=2)
    analysis = analyze_periodicity(
        latent,
        prominence_ratio=0.05,
        minimum_peak_distance=4,
    )
    assert analysis.period_x in (8, 16)
    assert analysis.period_y in (8, 16)


if __name__ == "__main__":
    checks = [
        ("graph_cut_is_monotone", graph_cut_is_monotone),
        ("complete_tile_set_and_periodic_grid", complete_tile_set_and_periodic_grid),
        ("periodic_signal_is_detected", periodic_signal_is_detected),
    ]
    for check_name, check_function in checks:
        check(check_name, check_function)
    print(f"\n{len(checks)}/{len(checks)} checks passed")
