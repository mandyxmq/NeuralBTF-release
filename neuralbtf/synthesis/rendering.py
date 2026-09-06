"""Rendering helpers for checkpoints and externally synthesised latent stacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from ..data import make_model_input, pixel_center_uv
from .latent import LatentLayout


@dataclass(frozen=True)
class LatentRender:
    """Decoded RGB and display-encoded neural-offset prediction.

    The offset image stores the UV-scale-adjusted, ungated prediction in RG,
    mapped from [-1, 1] to [0, 1]. The RGB warp separately applies the learned
    offset gate.
    """

    rgb: np.ndarray
    offset: np.ndarray


def _validate_direction(xy: Sequence[float], name: str) -> Tuple[float, float]:
    if len(xy) != 2:
        raise ValueError(f"{name} must contain x and y")
    x, y = float(xy[0]), float(xy[1])
    if x * x + y * y > 1.0 + 1e-6:
        raise ValueError(f"{name} lies outside the projected unit hemisphere: {xy}")
    return x, y


def _direction_tensor(xy: Sequence[float], device: torch.device) -> torch.Tensor:
    x, y = _validate_direction(xy, "direction")
    result = torch.tensor([[x, y]], device=device, dtype=torch.float32)
    z = torch.sqrt(torch.clamp(1.0 - (result * result).sum(-1, keepdim=True), min=0.0))
    return torch.cat([result, z], dim=-1)


def _pixel_uv(indices: torch.Tensor, height: int, width: int) -> torch.Tensor:
    u = (torch.remainder(indices, width).float() + 0.5) / width
    v = (torch.div(indices, width, rounding_mode="floor").float() + 0.5) / height
    return torch.stack([u, v], dim=-1)


def _sample_periodic(features: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample an HWC tensor with a toroidal boundary."""
    height, width, channels = features.shape
    x = uv[:, 0] * width - 0.5
    y = uv[:, 1] * height - 0.5
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    fx = (x - x0.float()).unsqueeze(-1)
    fy = (y - y0.float()).unsqueeze(-1)
    x0 = torch.remainder(x0, width)
    y0 = torch.remainder(y0, height)
    x1 = torch.remainder(x0 + 1, width)
    y1 = torch.remainder(y0 + 1, height)

    flat = features.reshape(height * width, channels)
    p00 = flat[y0 * width + x0]
    p10 = flat[y0 * width + x1]
    p01 = flat[y1 * width + x0]
    p11 = flat[y1 * width + x1]
    top = p00 + (p10 - p00) * fx
    bottom = p01 + (p11 - p01) * fx
    return top + (bottom - top) * fy


@torch.no_grad()
def render_model(
    model: torch.nn.Module,
    height: int,
    width: int,
    light_xy: Sequence[float],
    view_xy: Sequence[float],
    device: Optional[torch.device] = None,
    chunk_size: int = 262144,
) -> np.ndarray:
    """Render the model's internal textures at output pixel centres."""
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model = model.to(device).eval()
    light = _validate_direction(light_xy, "light_xy")
    view = _validate_direction(view_xy, "view_xy")
    uv = pixel_center_uv(width, height, device)
    direction = torch.tensor(
        [light[0], light[1], view[0], view[1]],
        device=device,
        dtype=torch.float32,
    ).view(1, 4)
    output = np.empty((height * width, 3), dtype=np.float32)

    for start in range(0, height * width, chunk_size):
        stop = min(start + chunk_size, height * width)
        model_input = make_model_input(uv[start:stop], direction)
        prediction = torch.nan_to_num(
            model(model_input).float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        output[start:stop] = prediction.cpu().numpy()
    return output.reshape(height, width, 3)


@torch.no_grad()
def render_latent_stack(
    model: torch.nn.Module,
    latent: np.ndarray,
    layout: LatentLayout,
    light_xy: Sequence[float],
    view_xy: Sequence[float],
    device: Optional[torch.device] = None,
    output_height: Optional[int] = None,
    output_width: Optional[int] = None,
    uv_scale: Sequence[float] = (1.0, 1.0),
    chunk_size: int = 262144,
) -> LatentRender:
    """Decode an external latent stack with the BTF's offset and RGB networks.

    `uv_scale=(scale_x, scale_y)` is the synthesized domain size relative to the
    captured BTF. The learned offset is divided by this scale so its displacement
    remains constant in source-material units, including for rectangular outputs.
    """
    if latent.ndim != 3 or latent.shape[2] != layout.total_channels:
        raise ValueError(
            f"latent shape {latent.shape} does not match {layout.total_channels} channels"
        )
    if len(uv_scale) != 2 or float(uv_scale[0]) <= 0.0 or float(uv_scale[1]) <= 0.0:
        raise ValueError("uv_scale must contain two positive values")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model = model.to(device).eval()
    latent_tensor = torch.as_tensor(
        np.asarray(latent, dtype=np.float32), device=device
    ).contiguous()
    appearance = latent_tensor[:, :, :layout.appearance_channels]
    offset_features = latent_tensor[:, :, layout.appearance_channels:]

    height = int(output_height or latent.shape[0])
    width = int(output_width or latent.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid render size {width}x{height}")

    light_3d = _direction_tensor(light_xy, device)
    view_3d = _direction_tensor(view_xy, device)
    encoded = model.dir_encoder(torch.cat([light_3d, view_3d], dim=-1))
    scale = torch.tensor(
        [float(uv_scale[0]), float(uv_scale[1])],
        device=device,
        dtype=torch.float32,
    ).view(1, 2)

    rgb = np.empty((height * width, 3), dtype=np.float32)
    offset_image = np.full((height * width, 3), 0.5, dtype=np.float32)
    for start in range(0, height * width, chunk_size):
        stop = min(start + chunk_size, height * width)
        indices = torch.arange(start, stop, device=device)
        uv = _pixel_uv(indices, height, width)
        count = stop - start

        if layout.offset_channels:
            off_features = _sample_periodic(offset_features, uv)
            view_batch = view_3d.expand(count, -1)
            predicted = torch.tanh(model.offset_mlp(torch.cat([off_features, view_batch], -1)))
            if layout.offset_mode == "1d":
                projected_view = view_batch[:, :2] / torch.clamp(
                    view_batch[:, 2:3], min=0.6
                )
                predicted = projected_view * predicted
            elif layout.offset_mode != "2d":
                raise ValueError(f"unsupported offset mode {layout.offset_mode!r}")
            scaled_offset = predicted / scale
            applied_offset = model.offset_gate * scaled_offset
            shifted_uv = torch.remainder(uv + applied_offset, 1.0)
            offset_viz = scaled_offset * 0.5 + 0.5
            offset_image[start:stop, :2] = offset_viz.cpu().numpy()
        else:
            shifted_uv = uv

        spatial = _sample_periodic(appearance, shifted_uv)
        directional = encoded.expand(count, -1)
        rgb[start:stop] = model.mlp(torch.cat([spatial, directional], -1)).float().cpu().numpy()

    return LatentRender(
        rgb=rgb.reshape(height, width, 3),
        offset=offset_image.reshape(height, width, 3),
    )
