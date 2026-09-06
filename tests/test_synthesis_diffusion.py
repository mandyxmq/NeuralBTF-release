#!/usr/bin/env python3
"""CPU checks for the optional latent-diffusion synthesis extra."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralbtf.synthesis.config import (DiffusionConfig, DiffusionDataConfig,  # noqa: E402
                                        DiffusionModelConfig,
                                        DiffusionPreviewConfig,
                                        DiffusionScheduleConfig,
                                        DiffusionTrainConfig)
from neuralbtf.synthesis.diffusion import (ExponentialMovingAverage,  # noqa: E402
                                           LatentCropDataset,
                                           build_sampling_scheduler,
                                           build_training_scheduler,
                                           build_unet, cuda_memory,
                                           cuda_memory_stats, load_checkpoint,
                                           save_checkpoint)


def check(name, function) -> None:
    function()
    print(f"ok    {name}")


def crops_are_reproducible_and_non_wrapped() -> None:
    y, x = np.meshgrid(
        np.arange(32, dtype=np.float32),
        np.arange(40, dtype=np.float32),
        indexing="ij",
    )
    latent = np.stack([x, y, x + y], axis=-1)
    first = LatentCropDataset(
        latent,
        crop_size=8,
        samples=4,
        zoom_range=(1.0, 2.0),
        extremes_probability=0.25,
        seed=7,
    )
    second = LatentCropDataset(
        latent,
        crop_size=8,
        samples=4,
        zoom_range=(1.0, 2.0),
        extremes_probability=0.25,
        seed=7,
        mean=first.mean,
        std=first.std,
    )
    crop_a, zoom_a = first[2]
    crop_b, zoom_b = second[2]
    assert torch.equal(crop_a, crop_b)
    assert torch.equal(zoom_a, zoom_b)
    assert crop_a.shape == (3, 8, 8)
    assert torch.isfinite(crop_a).all()

    first.set_epoch(1)
    crop_c, _ = first[2]
    assert not torch.equal(crop_a, crop_c)


def cpu_memory_reporting_is_safe() -> None:
    device = torch.device("cpu")
    assert cuda_memory(device) == "-"
    assert cuda_memory_stats(device) == {
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
    }


def schedulers_and_checkpoint_roundtrip() -> None:
    model_config = DiffusionModelConfig(
        base_channels=16,
        channel_multipliers=(1, 2),
        layers_per_block=1,
        norm_groups=8,
        attention=False,
    )
    schedule_config = DiffusionScheduleConfig(train_timesteps=20)
    config = DiffusionConfig(
        data=DiffusionDataConfig(
            crop_size=8,
            samples_per_epoch=4,
            validation_samples=0,
            batch_size=2,
            zoom_range=(1.0, 2.0),
            workers=0,
        ),
        schedule=schedule_config,
        model=model_config,
        train=DiffusionTrainConfig(epochs=1, print_every=1),
        preview=DiffusionPreviewConfig(enabled=False),
    )
    model = build_unet(3, model_config, torch.device("cpu"))
    output = model(torch.randn(2, 4, 16, 16), torch.tensor([1, 2])).sample
    assert output.shape == (2, 3, 16, 16)
    assert build_training_scheduler(schedule_config).__class__.__name__ == "DDPMScheduler"
    assert build_sampling_scheduler(schedule_config).__class__.__name__ == "DDIMScheduler"

    ema = ExponentialMovingAverage(model, decay=0.9)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model_final.pt"
        save_checkpoint(
            path,
            ema,
            latent_channels=3,
            config=config,
            epoch=1,
            global_step=2,
        )
        loaded, loaded_schedule, checkpoint = load_checkpoint(
            path, torch.device("cpu")
        )
        assert checkpoint["global_step"] == 2
        assert loaded_schedule.train_timesteps == 20
        loaded_output = loaded(
            torch.randn(1, 4, 16, 16), torch.tensor([1])
        ).sample
        assert loaded_output.shape == (1, 3, 16, 16)


if __name__ == "__main__":
    checks = [
        ("crops_are_reproducible_and_non_wrapped", crops_are_reproducible_and_non_wrapped),
        ("cpu_memory_reporting_is_safe", cpu_memory_reporting_is_safe),
        ("schedulers_and_checkpoint_roundtrip", schedulers_and_checkpoint_roundtrip),
    ]
    for check_name, check_function in checks:
        check(check_name, check_function)
    print(f"\n{len(checks)}/{len(checks)} checks passed")
