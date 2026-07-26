"""Far-field BTF data stored as HDF5.

Expected datasets (see ``README.md``):

    ground_light       (N, 2)        light direction xy, one per image
    ground_camera_dir  (N, 2)        view direction xy, one per image
    ground_color       (N, H, W, 3)  radiance, linear float32

Only the requested crop window and the requested number of directions are read
from disk, so a multi-GB capture never has to fit in RAM.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

LIGHT_KEY = "ground_light"
VIEW_KEY = "ground_camera_dir"
COLOR_KEY = "ground_color"


@dataclass(frozen=True)
class Crop:
    """Pixel window into the captured images."""

    xstart: int = 0
    ystart: int = 0
    width: int = 512
    height: int = 512

    def validate(self, img_h: int, img_w: int) -> None:
        if self.xstart < 0 or self.ystart < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError(f"invalid crop {self}")
        if self.xstart + self.width > img_w or self.ystart + self.height > img_h:
            raise ValueError(
                f"crop x[{self.xstart}:{self.xstart + self.width}] "
                f"y[{self.ystart}:{self.ystart + self.height}] does not fit in a "
                f"{img_w}x{img_h} capture"
            )


@dataclass
class BTFSlice:
    """A crop of a BTF capture held in host memory."""

    light: np.ndarray   # (N, 2) float32
    view: np.ndarray    # (N, 2) float32
    color: np.ndarray   # (N, H, W, 3) float32

    @property
    def n_dirs(self) -> int:
        return self.color.shape[0]

    @property
    def height(self) -> int:
        return self.color.shape[1]

    @property
    def width(self) -> int:
        return self.color.shape[2]

    @property
    def dirs(self) -> np.ndarray:
        """(N, 4) light and view directions concatenated."""
        return np.concatenate((self.light, self.view), axis=-1)

    def subset(self, start: int, stop: int) -> "BTFSlice":
        """A view of directions ``[start:stop]``; the pixel data is not copied."""
        return BTFSlice(self.light[start:stop], self.view[start:stop], self.color[start:stop])


def _check_far_field(f: h5py.File, path: str) -> None:
    missing = [k for k in (LIGHT_KEY, VIEW_KEY, COLOR_KEY) if k not in f]
    if missing:
        raise KeyError(f"{path}: missing dataset(s) {missing}; found {list(f.keys())}")
    if f[VIEW_KEY].ndim != 2:
        raise ValueError(
            f"{path}: '{VIEW_KEY}' has shape {f[VIEW_KEY].shape}, expected (N, 2). "
            "This looks like a near-field capture (per-pixel view directions); "
            "this release trains on far-field data only."
        )


def capture_shape(path: str) -> tuple[int, int, int]:
    """``(n_dirs, height, width)`` of a capture, without reading any pixels."""
    with h5py.File(path, "r") as f:
        _check_far_field(f, path)
        n, height, width, _ = f[COLOR_KEY].shape
    return n, height, width


def read_btf(path: str, crop: Crop | None = None, max_dirs: int | None = None,
             scale: float = 1.0) -> BTFSlice:
    """Read ``max_dirs`` directions of ``path``, cropped to ``crop``."""
    with h5py.File(path, "r") as f:
        _check_far_field(f, path)

        color_dset = f[COLOR_KEY]
        n_available, img_h, img_w, _ = color_dset.shape
        n = n_available if max_dirs is None else min(max_dirs, n_available)

        crop = crop or Crop(0, 0, img_w, img_h)
        crop.validate(img_h, img_w)
        y0, y1 = crop.ystart, crop.ystart + crop.height
        x0, x1 = crop.xstart, crop.xstart + crop.width

        print(f"[data] {path}: {n}/{n_available} directions, "
              f"crop x[{x0}:{x1}] y[{y0}:{y1}] of {img_w}x{img_h}")

        light = np.asarray(f[LIGHT_KEY][:n], dtype=np.float32)
        view = np.asarray(f[VIEW_KEY][:n], dtype=np.float32)
        color = np.asarray(color_dset[:n, y0:y1, x0:x1, :], dtype=np.float32)

    if scale != 1.0:
        color *= scale
    return BTFSlice(light=light, view=view, color=color)


def split_held_out(data: BTFSlice, val_split: float = 0.5) -> tuple[BTFSlice, BTFSlice]:
    """Cut a held-out capture into its validation and test halves.

    Training watches only the validation part; the test part is scored once, after
    training.  ``val_split`` is the fraction that becomes validation -- the default
    0.5 reproduces the ``nval = numdir // 2`` convention of the reference evaluation
    scripts, and 1.0 leaves no test set (validate on everything).
    """
    n_val = int(data.n_dirs * val_split)
    if n_val == 0:
        raise ValueError(f"val_split {val_split} leaves no validation directions "
                         f"out of {data.n_dirs}")
    return data.subset(0, n_val), data.subset(n_val, data.n_dirs)


class BTFImageDataset(Dataset):
    """One item is one captured image plus its (light, view) direction pair."""

    def __init__(self, data: BTFSlice):
        self.dirs = torch.from_numpy(np.ascontiguousarray(data.dirs)).float()
        self.color = torch.from_numpy(data.color)

    def __len__(self) -> int:
        return self.dirs.shape[0]

    def __getitem__(self, idx: int):
        return self.dirs[idx], self.color[idx], idx


def stratified_uv(xnum: int, ynum: int, device: torch.device,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """One jittered sample per cell of an ``ynum x xnum`` grid over the unit square.

    Returns ``(ynum * xnum, 2)`` in row-major order, i.e. sample ``i * xnum + j``
    lies in the cell of pixel ``(row i, column j)``.
    """
    dx, dy = 1.0 / xnum, 1.0 / ynum
    x = torch.linspace(0.0, 1.0 - dx, xnum, device=device)
    y = torch.linspace(0.0, 1.0 - dy, ynum, device=device)
    base = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1).reshape(-1, 2)
    jitter = torch.rand(base.shape, device=device, generator=generator)
    return base + jitter * torch.tensor([dx, dy], device=device)


def pixel_center_uv(width: int, height: int, device: torch.device) -> torch.Tensor:
    """``(height * width, 2)`` uv coordinates of pixel centres, row-major."""
    x = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=device)
    y = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=device)
    return torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1).reshape(-1, 2)


def make_model_input(uv: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Pair every uv sample with every direction pair.

    ``uv``: ``(P, 2)``, ``dirs``: ``(B, 4)`` -> ``(B * P, 6)``, image-major.
    """
    n_uv = uv.shape[0]
    return torch.cat([uv.repeat(dirs.shape[0], 1), dirs.repeat_interleave(n_uv, dim=0)], dim=-1)
