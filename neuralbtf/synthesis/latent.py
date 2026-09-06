"""Extraction of the spatial feature fields from a trained neural BTF."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LatentLayout:
    """Channel layout of an extracted `(H, W, C)` latent stack."""

    appearance_channels: int
    offset_channels: int
    channels_per_level: int
    appearance_resolutions: Tuple[int, ...]
    offset_mode: str

    @property
    def total_channels(self) -> int:
        return self.appearance_channels + self.offset_channels

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "LatentLayout":
        return cls(
            appearance_channels=int(values["appearance_channels"]),
            offset_channels=int(values["offset_channels"]),
            channels_per_level=int(values["channels_per_level"]),
            appearance_resolutions=tuple(int(x) for x in values["appearance_resolutions"]),
            offset_mode=str(values["offset_mode"]),
        )


def _resample_texture(texture: torch.Tensor, height: int, width: int) -> np.ndarray:
    sampled = F.interpolate(texture.float(), size=(height, width), mode="bilinear",
                            align_corners=False)
    return sampled[0].permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def extract_latent_stack(
    model: torch.nn.Module,
    height: Optional[int] = None,
    width: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, LatentLayout]:
    """Rasterise all spatial model features onto one regular pixel grid.

    Appearance levels are concatenated from coarse to fine, followed by the
    offset feature texture when the model has one. Each source texture is
    resampled independently before concatenation, preserving the model's
    align-corners-false sampling convention.
    """
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model = model.to(device).eval()

    cfg = model.cfg
    height = int(height or cfg.texture_res)
    width = int(width or cfg.texture_res)
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid extraction size {width}x{height}")

    channels_per_level = int(model.textures[0].shape[1])
    level_resolutions = tuple(int(tex.shape[-1]) for tex in model.textures)
    appearance_channels = channels_per_level * len(model.textures)
    has_offset = bool(model.use_neural_offset)
    offset_channels = int(cfg.off_texture_channels) if has_offset else 0
    total_channels = appearance_channels + offset_channels

    latent = np.empty((height, width, total_channels), dtype=np.float32)
    start = 0
    for texture in model.textures:
        stop = start + channels_per_level
        latent[:, :, start:stop] = _resample_texture(texture, height, width)
        start = stop
    if has_offset:
        latent[:, :, start:] = _resample_texture(model.offset_texture, height, width)

    layout = LatentLayout(
        appearance_channels=appearance_channels,
        offset_channels=offset_channels,
        channels_per_level=channels_per_level,
        appearance_resolutions=level_resolutions,
        offset_mode=cfg.offset_mode if has_offset else "none",
    )
    return latent, layout
