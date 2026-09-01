#!/usr/bin/env python3
"""CPU checks for RePaint masks, conditioning, and synthesis artifacts."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuralbtf.synthesis.artifacts import load_tile_set
from neuralbtf.synthesis.repaint import build_cross_mask, repaint_latent
from neuralbtf.synthesis.wang import write_tile_grid


class _Prediction:
    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class _ProbeModel(torch.nn.Module):
    def __init__(self, expected_log_zoom: float) -> None:
        super().__init__()
        self.expected_log_zoom = expected_log_zoom
        self.calls = 0

    def forward(self, model_input: torch.Tensor, timestep) -> _Prediction:
        zoom = model_input[:, :1]
        assert torch.allclose(
            zoom, torch.full_like(zoom, self.expected_log_zoom)
        )
        self.calls += 1
        return _Prediction(torch.zeros_like(model_input[:, 1:]))


def check(name, function) -> None:
    function()
    print(f"ok    {name}")


def cross_mask_tapers_at_wang_boundaries() -> None:
    seam = np.zeros((32, 40), dtype=bool)
    seam[16, :] = True
    seam[:, 20] = True
    mask = build_cross_mask(
        seam, core_width=7, taper_power=0.5, flat_ratio=0.8
    )[..., 0]
    assert mask.shape == seam.shape
    assert mask.dtype == np.float32
    assert mask[16, 20] == 1.0
    assert mask[:, 20].sum() == 32
    assert mask[:, 0].sum() < mask[:, 20].sum()


def repaint_is_deterministic_and_preserves_context() -> None:
    rng = np.random.default_rng(4)
    context = rng.normal(size=(8, 8, 3)).astype(np.float32)
    mask = np.zeros((8, 8, 1), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    scheduler = DDIMScheduler(
        num_train_timesteps=20,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="v_prediction",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    model = _ProbeModel(expected_log_zoom=1.0)
    arguments = dict(
        zoom=2.0,
        steps=4,
        jump=2,
        repeats=2,
        seed=9,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        strength=1.0,
        feather=0,
        progress=False,
    )
    first = repaint_latent(model, scheduler, context, mask, **arguments)
    second = repaint_latent(model, scheduler, context, mask, **arguments)
    assert model.calls > 0
    assert np.array_equal(first, second)
    assert np.array_equal(first[mask[..., 0] == 0], context[mask[..., 0] == 0])
    assert np.isfinite(first).all()


def tile_manifest_and_memmap_grid_round_trip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tiles = root / "tiles"
        seams = root / "seams"
        tiles.mkdir()
        seams.mkdir()
        records = []
        paths = {}
        for index in range(3):
            latent = np.full((4, 6, 2), float(index), dtype=np.float32)
            latent_path = tiles / f"tile_{index:02d}.npy"
            seam_path = seams / f"seam_{index:02d}.npy"
            np.save(latent_path, latent)
            np.save(seam_path, np.zeros((4, 6), dtype=bool))
            paths[index] = latent_path
            records.append(
                {
                    "index": index,
                    "edges": {
                        "north": "h0",
                        "east": "v0",
                        "south": "h0",
                        "west": "v0",
                    },
                    "latent": f"tiles/{latent_path.name}",
                    "seam": f"seams/{seam_path.name}",
                }
            )
        with (root / "tile_set.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "complete": False,
                    "tile_height": 4,
                    "tile_width": 6,
                    "channels": 2,
                    "tiles": records,
                },
                handle,
            )
        artifact = load_tile_set(root)
        assert not artifact.complete
        assert len(artifact.entries) == 3
        output_path = root / "grid.npy"
        shape = write_tile_grid(paths, [[0, 1, 2], [2, 1, 0]], output_path)
        output = np.load(output_path, mmap_mode="r")
        assert shape == (8, 18, 2)
        assert output.shape == shape
        assert np.all(output[:4, :6] == 0.0)
        assert np.all(output[4:, :6] == 2.0)


if __name__ == "__main__":
    checks = (
        ("cross_mask_tapers_at_wang_boundaries", cross_mask_tapers_at_wang_boundaries),
        (
            "repaint_is_deterministic_and_preserves_context",
            repaint_is_deterministic_and_preserves_context,
        ),
        ("tile_manifest_and_memmap_grid_round_trip", tile_manifest_and_memmap_grid_round_trip),
    )
    for label, function in checks:
        check(label, function)
    print(f"\n{len(checks)}/{len(checks)} checks passed")
