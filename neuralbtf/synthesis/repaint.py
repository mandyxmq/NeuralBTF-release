"""Memory-conscious two-stage RePaint for latent Wang-tile seams."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from scipy.ndimage import distance_transform_edt

from .diffusion import cuda_memory


def build_cross_mask(
    seam: np.ndarray,
    *,
    core_width: int,
    taper_power: float,
    flat_ratio: float,
) -> np.ndarray:
    """Expand graph-cut seams into a cross-shaped edit mask.

    The band narrows toward tile boundaries so RePaint does not alter the Wang
    edge signatures that make neighboring tiles compatible.
    """
    seam = np.asarray(seam)
    if seam.ndim == 3:
        seam = seam[..., 0]
    if seam.ndim != 2:
        raise ValueError(f"expected a 2D seam mask, got {seam.shape}")
    seam = seam > 0.5
    if not np.any(seam):
        return np.zeros((*seam.shape, 1), dtype=np.float32)

    distance_to_seam = distance_transform_edt(~seam)
    height, width = seam.shape
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    distance_to_border = np.minimum(
        np.minimum(y, height - 1 - y),
        np.minimum(x, width - 1 - x),
    )
    flat_radius = float(distance_to_border.max()) * float(flat_ratio)
    normalized = np.clip(
        distance_to_border / max(flat_radius, 1e-6), 0.0, 1.0
    )
    half_width = float(core_width) * np.power(normalized, float(taper_power))
    return (distance_to_seam <= half_width).astype(np.float32)[..., None]


def resize_hwc(
    array: np.ndarray,
    *,
    height: int,
    width: int,
    nearest: bool = False,
) -> np.ndarray:
    """Resize an HWC array on CPU while preserving float32 storage."""
    if height <= 0 or width <= 0:
        raise ValueError("resize dimensions must be positive")
    source = np.asarray(array, dtype=np.float32)
    if source.ndim == 2:
        source = source[..., None]
    tensor = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0)
    if nearest:
        resized = F.interpolate(tensor, size=(height, width), mode="nearest")
    else:
        resized = F.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=height < source.shape[0] or width < source.shape[1],
        )
    return (
        resized[0]
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
        .astype(np.float32, copy=False)
    )


def downsample_for_zoom(
    array: np.ndarray,
    zoom: float,
    *,
    multiple: int = 8,
    nearest: bool = False,
) -> np.ndarray:
    if zoom <= 0.0:
        raise ValueError("zoom must be positive")
    height, width = array.shape[:2]
    target_h = max(multiple, int(np.floor(height / zoom)) // multiple * multiple)
    target_w = max(multiple, int(np.floor(width / zoom)) // multiple * multiple)
    target_h = min(height, target_h)
    target_w = min(width, target_w)
    return resize_hwc(
        array, height=target_h, width=target_w, nearest=nearest
    )


def _random_like(
    reference: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def _alpha_at(
    scheduler: DDIMScheduler,
    timestep: int,
    device: torch.device,
) -> torch.Tensor:
    if timestep < 0:
        value = scheduler.final_alpha_cumprod
    else:
        value = scheduler.alphas_cumprod[timestep]
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _seeded_noise(
    shape: Tuple[int, ...],
    *,
    seed: int,
    key: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(
        int(seed) + 104729 * (int(key) + 1)
    )
    return torch.randn(shape, device=device, dtype=dtype, generator=generator)


def _q_sample(
    clean: torch.Tensor,
    alpha: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    return alpha.sqrt() * clean + (1.0 - alpha).clamp_min(0.0).sqrt() * noise


def _epsilon_from_prediction(
    prediction: torch.Tensor,
    current: torch.Tensor,
    alpha: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "epsilon":
        return prediction
    if prediction_type == "v_prediction":
        return alpha.sqrt() * prediction + (1.0 - alpha).sqrt() * current
    raise ValueError(f"unsupported prediction type {prediction_type!r}")


def _reverse_step(
    current: torch.Tensor,
    epsilon: torch.Tensor,
    alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Stochastic DDIM/DDPM-form transition to an explicitly chosen state."""
    predicted_clean = (
        current - (1.0 - alpha).sqrt() * epsilon
    ) / alpha.sqrt().clamp_min(1e-12)
    variance = (
        (1.0 - target_alpha)
        / (1.0 - alpha).clamp_min(1e-12)
        * (1.0 - alpha / target_alpha.clamp_min(1e-12))
    ).clamp_min(0.0)
    direction = (1.0 - target_alpha - variance).clamp_min(0.0).sqrt()
    noise = _random_like(current, generator)
    return (
        target_alpha.sqrt() * predicted_clean
        + direction * epsilon
        + variance.sqrt() * noise
    )


def _forward_step(
    current: torch.Tensor,
    source_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Re-noise from a cleaner state to an earlier RePaint state."""
    ratio = (target_alpha / source_alpha.clamp_min(1e-12)).clamp(0.0, 1.0)
    return ratio.sqrt() * current + (1.0 - ratio).sqrt() * _random_like(
        current, generator
    )


def _block_boundaries(start: int, transitions: int, jump: int) -> list[int]:
    boundaries = [int(start)]
    while boundaries[-1] < transitions:
        boundaries.append(min(transitions, boundaries[-1] + int(jump)))
    return boundaries


def _feather_alpha(mask: np.ndarray, width: int) -> np.ndarray:
    binary = np.asarray(mask[..., 0] > 0.5)
    if width <= 0:
        return binary.astype(np.float32)[..., None]
    inside_distance = distance_transform_edt(binary)
    alpha = np.clip(inside_distance / float(width), 0.0, 1.0)
    return alpha.astype(np.float32)[..., None]


@torch.inference_mode()
def repaint_latent(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    context: np.ndarray,
    mask: np.ndarray,
    *,
    zoom: float,
    steps: int,
    jump: int,
    repeats: int,
    seed: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    initial: Optional[np.ndarray] = None,
    strength: float = 1.0,
    feather: int = 0,
    progress: bool = True,
) -> np.ndarray:
    """Repaint one normalized latent while preserving pixels outside ``mask``.

    Noise rolling is deliberately absent: Wang compatibility fixes the domain
    boundary, and rolling would move the edit mask away from its graph-cut seam.
    """
    context_np = np.asarray(context, dtype=np.float32)
    if context_np.ndim != 3:
        raise ValueError(f"expected an HWC latent, got {context_np.shape}")
    height, width, channels = context_np.shape
    if mask.shape[:2] != (height, width):
        raise ValueError("mask and latent dimensions do not match")
    if initial is not None and np.asarray(initial).shape != context_np.shape:
        raise ValueError("initial latent and context dimensions do not match")
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must lie in (0, 1]")

    scheduler.set_timesteps(int(steps), device=device)
    timesteps = [int(value) for value in scheduler.timesteps]
    state_timesteps = timesteps + [-1]
    start = int(round(int(steps) * (1.0 - float(strength))))
    start = max(0, min(start, int(steps) - 1))
    boundaries = _block_boundaries(start, int(steps), int(jump))

    state_dtype = compute_dtype if device.type == "cuda" else torch.float32
    clean = (
        torch.from_numpy(context_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=state_dtype)
    )
    edit = torch.from_numpy((mask[..., :1] > 0.5).astype(np.float32))
    edit = edit.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=state_dtype)
    known = 1.0 - edit

    start_timestep = state_timesteps[start]
    start_alpha = _alpha_at(scheduler, start_timestep, device)
    context_noise = _seeded_noise(
        tuple(clean.shape),
        seed=seed,
        key=start_timestep + 2,
        device=device,
        dtype=state_dtype,
    )
    known_state = _q_sample(clean, start_alpha, context_noise)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    if initial is None:
        edit_state = _random_like(clean, generator)
    else:
        guess = (
            torch.from_numpy(np.asarray(initial, dtype=np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=state_dtype)
        )
        edit_state = _q_sample(guess, start_alpha, _random_like(guess, generator))
    current = known * known_state + edit * edit_state

    zoom_channel = torch.full(
        (1, 1, height, width),
        float(np.log2(zoom)),
        device=device,
        dtype=state_dtype,
    )
    total_blocks = len(boundaries) - 1
    model.eval()

    for block_index, (block_start, block_stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:]), start=1
    ):
        for repeat_index in range(int(repeats)):
            for state_index in range(block_start, block_stop):
                timestep = state_timesteps[state_index]
                next_timestep = state_timesteps[state_index + 1]
                alpha = _alpha_at(scheduler, timestep, device)
                next_alpha = _alpha_at(scheduler, next_timestep, device)
                autocast = (
                    torch.autocast("cuda", dtype=compute_dtype)
                    if device.type == "cuda" and compute_dtype == torch.float16
                    else nullcontext()
                )
                with autocast:
                    prediction = model(
                        torch.cat([zoom_channel, current], dim=1), timestep
                    ).sample
                current_float = current.float()
                epsilon = _epsilon_from_prediction(
                    prediction.float(),
                    current_float,
                    alpha,
                    scheduler.config.prediction_type,
                )
                step_generator = torch.Generator(device=device).manual_seed(
                    int(seed)
                    + 1_000_003 * block_index
                    + 10_007 * repeat_index
                    + timestep
                )
                current = _reverse_step(
                    current_float,
                    epsilon,
                    alpha,
                    next_alpha,
                    step_generator,
                ).to(dtype=state_dtype)
                current = torch.nan_to_num(current, nan=0.0, posinf=1e4, neginf=-1e4)

            target_timestep = state_timesteps[block_stop]
            target_alpha = _alpha_at(scheduler, target_timestep, device)
            known_noise = _seeded_noise(
                tuple(clean.shape),
                seed=seed,
                key=target_timestep + 2,
                device=device,
                dtype=state_dtype,
            )
            known_state = _q_sample(clean, target_alpha, known_noise)
            current = known * known_state + edit * current

            if repeat_index + 1 < int(repeats):
                source_alpha = target_alpha
                earlier_timestep = state_timesteps[block_start]
                earlier_alpha = _alpha_at(scheduler, earlier_timestep, device)
                jump_generator = torch.Generator(device=device).manual_seed(
                    int(seed) + 2_000_003 * block_index + repeat_index
                )
                current = _forward_step(
                    current.float(), source_alpha, earlier_alpha, jump_generator
                ).to(dtype=state_dtype)

        if progress:
            print(
                f"    repaint block {block_index}/{total_blocks} | "
                f"states {block_start}->{block_stop} | "
                f"t={target_timestep} | VRAM {cuda_memory(device)}"
            )

    result = (
        current[0]
        .permute(1, 2, 0)
        .float()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    result = np.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
    alpha = _feather_alpha(mask, feather)
    return (alpha * result + (1.0 - alpha) * context_np).astype(
        np.float32, copy=False
    )
