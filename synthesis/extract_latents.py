#!/usr/bin/env python3
"""Extract and validate the spatial latent stack of a trained neural BTF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from neuralbtf import load_model, write_exr, write_png  # noqa: E402
from neuralbtf.synthesis import (extract_latent_stack, render_latent_stack,  # noqa: E402
                                 render_model)
from neuralbtf.synthesis.artifacts import write_json  # noqa: E402
from neuralbtf.synthesis.config import (default_synthesis_root,  # noqa: E402
                                        load_synthesis_config,
                                        resolve_config_path, resolve_device,
                                        synchronize_repaint_zooms,
                                        synthesis_stage_paths)


DEFAULT_CONFIG = HERE / "configs" / "default.json"


def _optional_positive(value: Optional[int], fallback: int) -> int:
    resolved = fallback if value is None else int(value)
    if resolved <= 0:
        raise ValueError("image dimensions must be positive")
    return resolved


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="bootstrap JSON synthesis configuration")
    parser.add_argument("--checkpoint", required=True,
                        help="trained BTF checkpoint")
    parser.add_argument("--output-dir", default="",
                        help="synthesis output root; overrides the inferred path")
    parser.add_argument("--device", default="", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument("--height", type=int, default=None, help="latent height override")
    parser.add_argument("--width", type=int, default=None, help="latent width override")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace known files in an existing latent stage")
    args = parser.parse_args(argv)

    config_path = str(Path(args.config).resolve()) if args.config else None
    config = load_synthesis_config(config_path)
    config = synchronize_repaint_zooms(config)
    checkpoint = resolve_config_path(args.checkpoint, None)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    output_value = args.output_dir or config.output_dir
    output_root = (
        resolve_config_path(output_value, None if args.output_dir else config_path)
        if output_value else default_synthesis_root(checkpoint)
    )
    paths = synthesis_stage_paths(output_root, "latent")
    stage_dir = paths.artifacts
    preview_dir = paths.visualization
    project_path = output_root / "project.json"
    known_outputs = (
        stage_dir / "latent.npy",
        stage_dir / "metadata.json",
        stage_dir / "config.resolved.json",
        project_path,
    )
    if not args.overwrite and any(path.exists() for path in known_outputs):
        raise FileExistsError(
            f"latent outputs already exist in {stage_dir}; pass --overwrite to replace them"
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    if config.extract.preview.enabled:
        preview_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device or config.device
    device = resolve_device(device_name)
    model, model_config = load_model(str(checkpoint), device=device)
    height = _optional_positive(
        args.height if args.height is not None else config.extract.height,
        model_config.texture_res,
    )
    width = _optional_positive(
        args.width if args.width is not None else config.extract.width,
        model_config.texture_res,
    )
    preview_config = config.extract.preview
    light_xy = preview_config.light_xy
    view_xy = preview_config.view_xy

    print("--- Latent extraction ---")
    print(f"checkpoint: {checkpoint}")
    print(f"output:     {stage_dir}")
    print(f"resolution: {width}x{height}")
    print(f"device:     {device}")
    print(f"diffusion zoom: {config.diffusion.data.zoom_range}")
    print(
        "repaint zooms: "
        f"coarse={config.repaint.process.coarse_zoom:g}, "
        f"fine={config.repaint.process.fine_zoom:g} (synchronized)"
    )

    latent, layout = extract_latent_stack(model, height=height, width=width, device=device)
    latent_path = stage_dir / "latent.npy"
    np.save(latent_path, latent)
    print(f"[saved] {latent_path}  shape={latent.shape}")

    print(f"[preview] light={light_xy} view={view_xy}")
    decoded = render_latent_stack(
        model,
        latent,
        layout,
        light_xy=light_xy,
        view_xy=view_xy,
        device=device,
        chunk_size=preview_config.chunk_size,
    )
    direct = render_model(
        model,
        height,
        width,
        light_xy=light_xy,
        view_xy=view_xy,
        device=device,
        chunk_size=preview_config.chunk_size,
    )

    preview = {
        "enabled": preview_config.enabled,
        "light_xy": list(light_xy),
        "view_xy": list(view_xy),
    }
    if preview_config.enabled:
        latent_rgb_path = preview_dir / "extracted_latent_rgb.png"
        model_rgb_path = preview_dir / "checkpoint_rgb.png"
        offset_path = preview_dir / "extracted_latent_offset.png"
        write_png(str(latent_rgb_path), decoded.rgb)
        write_png(str(model_rgb_path), direct)
        write_png(str(offset_path), decoded.offset, srgb=False)
        if preview_config.save_exr:
            write_exr(str(preview_dir / "extracted_latent_rgb.exr"), decoded.rgb)
            write_exr(str(preview_dir / "checkpoint_rgb.exr"), direct)
        preview.update({
            "latent_rgb": str(latent_rgb_path.relative_to(output_root)),
            "checkpoint_rgb": str(model_rgb_path.relative_to(output_root)),
            "offset": str(offset_path.relative_to(output_root)),
        })

    absolute_difference = np.abs(decoded.rgb - direct)
    mae = float(absolute_difference.mean())
    maximum = float(absolute_difference.max())
    print(f"[validation] MAE={mae:.8f}, max={maximum:.8f}")

    validation = {
        "mean_absolute_error": mae,
        "maximum_absolute_error": maximum,
    }

    resolved = config.as_dict()
    resolved.update({
        "output_dir": str(output_root),
        "device": str(device),
    })
    resolved["extract"]["height"] = height
    resolved["extract"]["width"] = width
    metadata = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "model_config": model_config.as_dict(),
        "output_root": str(output_root),
        "project": str(project_path.relative_to(output_root)),
        "height": height,
        "width": width,
        "channels": layout.total_channels,
        "layout": layout.as_dict(),
        "latent": str(latent_path.relative_to(output_root)),
        "preview": preview,
        "validation": validation,
    }
    write_json(stage_dir / "config.resolved.json", resolved)
    write_json(project_path, resolved)
    write_json(stage_dir / "metadata.json", metadata)
    print(f"[saved] {stage_dir / 'metadata.json'}")
    print(f"[project] {project_path}")
    print("[done]")


if __name__ == "__main__":
    main()
