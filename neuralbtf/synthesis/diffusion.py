"""Zoom-conditioned diffusion over a single neural-BTF latent stack."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel
from torch.utils.data import Dataset

from .config import (
    DiffusionConfig,
    DiffusionModelConfig,
    DiffusionScheduleConfig,
)


def compute_latent_statistics(
    latent: np.ndarray,
    *,
    row_chunk: int = 64,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel moments without allocating another full latent stack."""
    if latent.ndim != 3:
        raise ValueError(f"expected an HWC latent, got {latent.shape}")
    channels = latent.shape[2]
    total = np.zeros(channels, dtype=np.float64)
    total_square = np.zeros(channels, dtype=np.float64)
    count = 0
    for start in range(0, latent.shape[0], row_chunk):
        chunk = np.asarray(latent[start : start + row_chunk], dtype=np.float32)
        flat = chunk.reshape(-1, channels).astype(np.float64, copy=False)
        total += flat.sum(axis=0)
        total_square += np.square(flat).sum(axis=0)
        count += flat.shape[0]
    mean = total / max(count, 1)
    variance = np.maximum(total_square / max(count, 1) - np.square(mean), 0.0)
    return mean.astype(np.float32), (np.sqrt(variance) + epsilon).astype(np.float32)


class LatentCropDataset(Dataset):
    """Deterministic, non-wrapped zoom crops from one material latent."""

    def __init__(
        self,
        latent: np.ndarray,
        *,
        crop_size: int,
        samples: int,
        zoom_range: Tuple[float, float],
        extremes_probability: float,
        seed: int,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        if latent.ndim != 3:
            raise ValueError(f"expected an HWC latent, got {latent.shape}")
        self.height, self.width, self.channels = latent.shape
        self.crop_size = int(crop_size)
        self.samples = int(samples)
        self.zoom_range = (float(zoom_range[0]), float(zoom_range[1]))
        self.extremes_probability = float(extremes_probability)
        self.seed = int(seed)
        self.epoch = 0

        maximum_read = int(round(self.crop_size * self.zoom_range[1]))
        if maximum_read > min(self.height, self.width):
            raise ValueError(
                f"crop_size * max_zoom is {maximum_read}, but the latent is only "
                f"{self.width}x{self.height}; reduce crop_size or zoom_range"
            )
        if mean is None or std is None:
            mean, std = compute_latent_statistics(latent)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (self.channels,) or self.std.shape != (self.channels,):
            raise ValueError("normalization statistics do not match latent channels")

        normalized = (
            np.asarray(latent, dtype=np.float32) - self.mean.reshape(1, 1, -1)
        ) / self.std.reshape(1, 1, -1)
        self.tensor = torch.from_numpy(normalized).permute(2, 0, 1).contiguous()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples

    def _rng(self, index: int) -> np.random.Generator:
        sequence = np.random.SeedSequence([self.seed, self.epoch, int(index)])
        return np.random.default_rng(sequence)

    def __getitem__(self, index: int):
        rng = self._rng(index)
        z_min, z_max = self.zoom_range
        draw = float(rng.random())
        if draw < self.extremes_probability:
            zoom = z_min
        elif draw > 1.0 - self.extremes_probability:
            zoom = z_max
        else:
            zoom = float(rng.uniform(z_min, z_max))

        read_size = int(round(self.crop_size * zoom))
        y0 = int(rng.integers(0, self.height - read_size + 1))
        x0 = int(rng.integers(0, self.width - read_size + 1))
        patch = self.tensor[:, y0 : y0 + read_size, x0 : x0 + read_size]
        if read_size != self.crop_size:
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
                antialias=read_size > self.crop_size,
            )[0]
        return patch, torch.tensor(zoom, dtype=torch.float32)


def _make_circular(module: nn.Module) -> nn.Module:
    """Replace convolution padding with periodic padding, preserving parameters."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            replacement = nn.Conv2d(
                child.in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode="circular",
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                if child.bias is not None:
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
        else:
            _make_circular(child)
    return module


def build_unet(
    latent_channels: int,
    model_config: DiffusionModelConfig,
    device: torch.device,
) -> nn.Module:
    down_blocks = []
    up_blocks = []
    count = len(model_config.channel_multipliers)
    for index in range(count):
        last = index == count - 1
        use_attention = model_config.attention and (
            last
            if model_config.attention_last_only
            else index >= count - 2
        )
        down_blocks.append(
            "AttnDownBlock2D" if use_attention else "DownBlock2D"
        )
        up_blocks.append(
            "AttnUpBlock2D" if use_attention else "UpBlock2D"
        )

    model = UNet2DModel(
        sample_size=None,
        in_channels=int(latent_channels) + 1,
        out_channels=int(latent_channels),
        layers_per_block=model_config.layers_per_block,
        block_out_channels=model_config.block_channels,
        down_block_types=tuple(down_blocks),
        up_block_types=tuple(up_blocks),
        norm_num_groups=model_config.norm_groups,
        attention_head_dim=(
            model_config.attention_head_dim if model_config.attention else None
        ),
    )
    return _make_circular(model).to(device)


def _schedule_kwargs(config: DiffusionScheduleConfig) -> Dict[str, Any]:
    return {
        "num_train_timesteps": config.train_timesteps,
        "beta_schedule": config.beta_schedule,
        "prediction_type": config.prediction_type,
        "clip_sample": config.clip_sample,
    }


def build_training_scheduler(config: DiffusionScheduleConfig) -> DDPMScheduler:
    """DDPM defines the forward Markov noising process used for training."""
    return DDPMScheduler(**_schedule_kwargs(config))


def build_sampling_scheduler(config: DiffusionScheduleConfig) -> DDIMScheduler:
    """DDIM traverses the trained DDPM marginals during synthesis."""
    return DDIMScheduler(
        **_schedule_kwargs(config),
        set_alpha_to_one=config.set_alpha_to_one,
        steps_offset=config.steps_offset,
    )


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                self.shadow[name].copy_(value.detach())

    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        state.update(self.shadow)
        model.load_state_dict(state, strict=True)

    def cpu_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in self.shadow.items()
        }


def _prediction_target(
    scheduler: DDPMScheduler,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if scheduler.config.prediction_type == "epsilon":
        return noise
    if scheduler.config.prediction_type == "v_prediction":
        return scheduler.get_velocity(clean, noise, timesteps)
    raise ValueError(f"unsupported prediction type {scheduler.config.prediction_type}")


def _min_snr_weights(
    scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    alpha = scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
    snr = alpha / torch.clamp_min(1.0 - alpha, 1e-8)
    clipped = torch.minimum(snr, torch.full_like(snr, float(gamma)))
    if scheduler.config.prediction_type == "v_prediction":
        return clipped / (snr + 1.0)
    return clipped / torch.clamp_min(snr, 1e-8)


def _zoom_channel(
    zoom: torch.Tensor,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.log2(zoom.float()).view(-1, 1, 1, 1)
    return value.expand(-1, 1, height, width).to(dtype=dtype)


def cuda_memory_stats(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda":
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def cuda_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "-"
    stats = cuda_memory_stats(device)
    mib = 1024**2
    allocated = stats["allocated_bytes"] / mib
    reserved = stats["reserved_bytes"] / mib
    peak_allocated = stats["peak_allocated_bytes"] / mib
    peak_reserved = stats["peak_reserved_bytes"] / mib
    return (
        f"allocated {allocated:.0f}/{peak_allocated:.0f}MB | "
        f"reserved {reserved:.0f}/{peak_reserved:.0f}MB (current/peak)"
    )


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def train_epoch(
    dataloader,
    dataset: LatentCropDataset,
    model: nn.Module,
    scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    config: DiffusionConfig,
    device: torch.device,
    *,
    epoch: int,
    global_step: int,
) -> Tuple[float, list, int, bool]:
    model.train()
    dataset.set_epoch(epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    losses = []
    start_time = time.perf_counter()
    maximum_steps = config.train.max_steps
    stopped = False

    for batch_index, (clean, zoom) in enumerate(dataloader, start=1):
        if maximum_steps is not None and global_step >= maximum_steps:
            stopped = True
            break
        clean = clean.to(device, non_blocking=True)
        zoom = zoom.to(device, non_blocking=True)
        batch, _, height, width = clean.shape
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (batch,),
            device=device,
        )
        noise = torch.randn_like(clean)
        noised = scheduler.add_noise(clean, noise, timesteps)
        target = _prediction_target(scheduler, clean, noise, timesteps)
        model_input = torch.cat(
            [_zoom_channel(zoom, height, width, clean.dtype), noised],
            dim=1,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.float16,
        ):
            prediction = model(model_input, timesteps).sample
            weights = _min_snr_weights(
                scheduler, timesteps, config.train.min_snr_gamma
            ).view(batch, 1, 1, 1)
            loss = torch.mean(weights * torch.square(prediction - target))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.train.gradient_clip
        )
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        global_step += 1
        losses.append(float(loss.item()))
        progress_due = (
            batch_index == 1
            or batch_index % config.train.print_every == 0
            or batch_index == len(dataloader)
        )
        if progress_due:
            elapsed = time.perf_counter() - start_time
            rate = elapsed / batch_index
            eta = rate * (len(dataloader) - batch_index)
            average = float(np.mean(losses))
            print(
                f"    step {batch_index}/{len(dataloader)} | "
                f"global {global_step} | loss {average:.6f} | "
                f"elapsed {format_duration(elapsed)} | "
                f"eta {format_duration(eta)} | VRAM {cuda_memory(device)}",
                flush=True,
            )

    return float(np.mean(losses)) if losses else math.nan, losses, global_step, stopped


@torch.no_grad()
def evaluate(
    dataloader,
    model: nn.Module,
    scheduler: DDPMScheduler,
    config: DiffusionConfig,
    device: torch.device,
    *,
    seed: int,
) -> float:
    model.eval()
    losses = []
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for clean, zoom in dataloader:
        clean = clean.to(device, non_blocking=True)
        zoom = zoom.to(device, non_blocking=True)
        batch, _, height, width = clean.shape
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (batch,),
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            clean.shape,
            device=device,
            dtype=clean.dtype,
            generator=generator,
        )
        noised = scheduler.add_noise(clean, noise, timesteps)
        target = _prediction_target(scheduler, clean, noise, timesteps)
        model_input = torch.cat(
            [_zoom_channel(zoom, height, width, clean.dtype), noised],
            dim=1,
        )
        prediction = model(model_input, timesteps).sample
        weights = _min_snr_weights(
            scheduler, timesteps, config.train.min_snr_gamma
        ).view(batch, 1, 1, 1)
        losses.append(float(torch.mean(weights * (prediction - target) ** 2)))
    return float(np.mean(losses)) if losses else math.nan


def _roll_batch(
    tensor: torch.Tensor,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    return torch.roll(tensor, shifts=(shift_y, shift_x), dims=(-2, -1))


@torch.no_grad()
def sample(
    model: nn.Module,
    schedule_config: DiffusionScheduleConfig,
    *,
    latent_channels: int,
    height: int,
    width: int,
    zoom: float,
    steps: int,
    seed: int,
    eta: float,
    roll_during_sampling: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    progress: bool = True,
) -> torch.Tensor:
    model.eval()
    scheduler = build_sampling_scheduler(schedule_config)
    scheduler.set_timesteps(int(steps), device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    current = torch.randn(
        (1, latent_channels, int(height), int(width)),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    zoom_values = torch.full((1,), float(zoom), device=device)
    zoom_channel = _zoom_channel(
        zoom_values, int(height), int(width), current.dtype
    )
    tick = max(1, len(scheduler.timesteps) // 5)

    for index, timestep in enumerate(scheduler.timesteps, start=1):
        model_input = torch.cat([zoom_channel, current], dim=1)
        shift_y = 0
        shift_x = 0
        if roll_during_sampling:
            shift_y = int(
                torch.randint(0, height, (1,), device=device, generator=generator)
            )
            shift_x = int(
                torch.randint(0, width, (1,), device=device, generator=generator)
            )
            model_input = _roll_batch(model_input, shift_y, shift_x)

        prediction = model(model_input, timestep).sample
        if roll_during_sampling:
            prediction = _roll_batch(prediction, -shift_y, -shift_x)
        current = scheduler.step(
            prediction,
            timestep,
            current,
            eta=float(eta),
            generator=generator,
        ).prev_sample

        if progress and (index % tick == 0 or index == len(scheduler.timesteps)):
            print(
                f"    sample {index}/{len(scheduler.timesteps)} | "
                f"t={int(timestep)} | VRAM {cuda_memory(device)}"
            )
    return current


def save_checkpoint(
    path: Path,
    ema: ExponentialMovingAverage,
    *,
    latent_channels: int,
    config: DiffusionConfig,
    epoch: int,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model": ema.cpu_state_dict(),
            "latent_channels": int(latent_channels),
            "model_config": asdict(config.model),
            "schedule_config": asdict(config.schedule),
            "epoch": int(epoch),
            "global_step": int(global_step),
        },
        path,
    )


def _model_config_from_dict(values: Dict[str, Any]) -> DiffusionModelConfig:
    copied = dict(values)
    copied["channel_multipliers"] = tuple(copied["channel_multipliers"])
    config = DiffusionModelConfig(**copied)
    config.validate()
    return config


def _schedule_config_from_dict(values: Dict[str, Any]) -> DiffusionScheduleConfig:
    config = DiffusionScheduleConfig(**values)
    config.validate()
    return config


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, DiffusionScheduleConfig, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("schema_version") != 1:
        raise ValueError(f"unsupported diffusion checkpoint schema in {path}")
    model_config = _model_config_from_dict(checkpoint["model_config"])
    schedule_config = _schedule_config_from_dict(checkpoint["schedule_config"])
    model = build_unet(int(checkpoint["latent_channels"]), model_config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    return model, schedule_config, checkpoint
