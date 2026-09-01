#!/usr/bin/env python3
"""Inpaint raw Wang-tile seams with two-stage latent RePaint."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from neuralbtf import write_exr, write_png  # noqa: E402
from neuralbtf.png import to_display  # noqa: E402
from neuralbtf.synthesis import render_latent_stack  # noqa: E402
from neuralbtf.synthesis.artifacts import (load_latent_artifact,  # noqa: E402
                                           load_source_model, load_tile_set,
                                           prepare_stage, preview_directions,
                                           read_json, source_uv_scale,
                                           write_json)
from neuralbtf.synthesis.config import (load_synthesis_config,  # noqa: E402
                                        resolve_device, resolve_output_root,
                                        synthesis_stage_paths)
from neuralbtf.synthesis.diffusion import (build_sampling_scheduler,  # noqa: E402
                                           cuda_memory_stats, format_duration,
                                           load_checkpoint)
from neuralbtf.synthesis.repaint import (build_cross_mask,  # noqa: E402
                                         downsample_for_zoom,
                                         repaint_latent, resize_hwc)


EDGE_COLORS = {
    "v0": (255, 0, 0),
    "v1": (0, 255, 0),
    "h0": (0, 0, 255),
    "h1": (255, 255, 0),
}


def _fit_size(height: int, width: int, maximum: int) -> Tuple[int, int]:
    scale = min(float(maximum) / height, float(maximum) / width, 1.0)
    return max(8, int(round(height * scale))), max(8, int(round(width * scale)))


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _edge_frame(image: np.ndarray, edges: dict, relative_width: float = 0.015) -> Image.Image:
    base = Image.fromarray(image)
    bar = max(3, int(round(min(base.size) * relative_width)))
    framed = Image.new("RGB", (base.width + 2 * bar, base.height + 2 * bar), "white")
    framed.paste(base, (bar, bar))
    draw = ImageDraw.Draw(framed)
    draw.rectangle((0, 0, framed.width, bar - 1), fill=EDGE_COLORS[edges["north"]])
    draw.rectangle(
        (0, framed.height - bar, framed.width, framed.height),
        fill=EDGE_COLORS[edges["south"]],
    )
    draw.rectangle((0, 0, bar - 1, framed.height), fill=EDGE_COLORS[edges["west"]])
    draw.rectangle(
        (framed.width - bar, 0, framed.width, framed.height),
        fill=EDGE_COLORS[edges["east"]],
    )
    return framed


def _save_grid(images: list[Image.Image], path: Path) -> None:
    if len(images) != 16:
        raise ValueError("a complete Wang tile visualization requires 16 images")
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    canvas = Image.new("RGB", (4 * cell_w, 4 * cell_h), "white")
    for index, image in enumerate(images):
        row, column = divmod(index, 4)
        canvas.paste(
            image,
            (
                column * cell_w + (cell_w - image.width) // 2,
                row * cell_h + (cell_h - image.height) // 2,
            ),
        )
    canvas.save(path)


def _precision(configured: str, device: torch.device, tile_size: int) -> torch.dtype:
    if configured == "float16":
        if device.type != "cuda":
            raise ValueError("float16 RePaint requires CUDA")
        return torch.float16
    if configured == "float32":
        return torch.float32
    return torch.float16 if device.type == "cuda" and tile_size >= 1024 else torch.float32


def _normalization(diffusion_dir: Path, channels: int) -> Tuple[np.ndarray, np.ndarray]:
    values = read_json(diffusion_dir / "normalization.json")
    mean = np.asarray(values["mean"], dtype=np.float32)
    std = np.asarray(values["std"], dtype=np.float32)
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError("diffusion normalization does not match tile channels")
    if np.any(std <= 0.0):
        raise ValueError("diffusion normalization contains a non-positive standard deviation")
    return mean.reshape(1, 1, -1), std.reshape(1, 1, -1)


def _finite_latent(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        array, nan=0.0, posinf=1e4, neginf=-1e4
    ).astype(np.float32, copy=False)


def _check_zoom_training_endpoints(
    diffusion_dir: Path,
    coarse: float,
    fine: float,
) -> None:
    resolved_path = diffusion_dir / "config.resolved.json"
    if not resolved_path.is_file():
        return
    values = read_json(resolved_path)
    trained = values["diffusion"]["data"]["zoom_range"]
    low, high = float(trained[0]), float(trained[1])
    expected = (
        ("coarse_zoom", coarse, high, "maximum"),
        ("fine_zoom", fine, low, "minimum"),
    )
    for name, value, endpoint, endpoint_name in expected:
        if not np.isclose(value, endpoint, rtol=0.0, atol=1e-6):
            raise ValueError(
                f"repaint.process.{name}={value:g} must match the trained "
                f"{endpoint_name} zoom {endpoint:g}; update project.json"
            )


def _render_preview(
    model,
    latent: np.ndarray,
    *,
    artifact,
    full_height: int,
    full_width: int,
    maximum: int,
    device: torch.device,
):
    height, width = _fit_size(full_height, full_width, maximum)
    light_xy, view_xy = preview_directions(artifact)
    return render_latent_stack(
        model,
        latent,
        artifact.layout,
        light_xy=light_xy,
        view_xy=view_xy,
        output_height=height,
        output_width=width,
        uv_scale=source_uv_scale(
            artifact,
            domain_height=full_height,
            domain_width=full_width,
        ),
        device=device,
    )


def _save_preview_stage(
    directory: Path,
    name: str,
    rendered,
    *,
    save_offset: bool,
    save_exr: bool,
) -> None:
    write_png(str(directory / f"{name}_rgb.png"), rendered.rgb)
    if save_offset:
        write_png(str(directory / f"{name}_offset.png"), rendered.offset, srgb=False)
    if save_exr:
        write_exr(str(directory / f"{name}_rgb.exr"), rendered.rgb)


def _clear_stage_previews(directory: Path) -> None:
    patterns = (
        "tile_*_cross_mask.png",
        "tile_*_raw_*",
        "tile_*_coarse_*",
        "tile_*_fine_*",
    )
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="project.json generated by extract_latents.py",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="synthesis root; overrides project config output_dir",
    )
    parser.add_argument("--device", default="", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument(
        "--first-tile-only",
        action="store_true",
        help="process only repaint.process.preview_tile for a quick check",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace known files in an existing RePaint stage",
    )
    args = parser.parse_args(argv)

    config_path = str(Path(args.config).resolve())
    config = load_synthesis_config(config_path)
    output_root = resolve_output_root(
        config,
        config_path=config_path,
        override=args.output_dir,
    )
    paths = synthesis_stage_paths(output_root, "repaint")
    manifest_path = paths.artifacts / "tile_set.json"
    metadata_path = paths.artifacts / "metadata.json"
    prepare_stage(paths.artifacts, (manifest_path, metadata_path), overwrite=args.overwrite)

    artifact = load_latent_artifact(output_root)
    raw_set = load_tile_set(output_root / "wang_tiles")
    if raw_set.channels != artifact.layout.total_channels:
        raise ValueError("Wang tile channels do not match the extracted latent")
    process_config = config.repaint.process
    _check_zoom_training_endpoints(
        output_root / "diffusion",
        process_config.coarse_zoom,
        process_config.fine_zoom,
    )
    entries = list(raw_set.entries)
    if args.first_tile_only:
        wanted = process_config.preview_tile
        entries = [entry for entry in entries if entry.index == wanted]
        if not entries:
            raise ValueError(f"raw Wang tile set does not contain tile {wanted}")

    device = resolve_device(args.device or config.device)
    compute_dtype = _precision(
        process_config.precision,
        device,
        max(raw_set.tile_height, raw_set.tile_width),
    )
    diffusion_dir = output_root / "diffusion"
    checkpoint_path = diffusion_dir / "model_final.pt"
    mean, std = _normalization(diffusion_dir, raw_set.channels)
    model, schedule_config, checkpoint = load_checkpoint(
        checkpoint_path, device, dtype=compute_dtype
    )
    scheduler = build_sampling_scheduler(schedule_config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("--- Wang seam RePaint ---")
    print(f"raw tiles:     {raw_set.root}")
    print(f"diffusion:     {checkpoint_path}")
    print(f"output:        {paths.artifacts}")
    print(f"tile shape:    {raw_set.tile_width}x{raw_set.tile_height}")
    print(f"coarse/fine:   {process_config.coarse_zoom:g}/{process_config.fine_zoom:g}")
    print(f"precision:     {str(compute_dtype).removeprefix('torch.')}")
    print(f"device:        {device}")

    tiles_dir = paths.artifacts / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings = []
    output_entries = []
    preview_payload = None

    for counter, entry in enumerate(entries, start=1):
        if entry.seam_path is None:
            raise ValueError(f"raw tile {entry.index} has no graph-cut seam")
        tile_started = time.perf_counter()
        raw = np.asarray(np.load(entry.latent_path), dtype=np.float32)
        seam = np.load(entry.seam_path)
        mask = build_cross_mask(
            seam,
            core_width=config.repaint.mask.core_width,
            taper_power=config.repaint.mask.taper_power,
            flat_ratio=config.repaint.mask.flat_ratio,
        )
        normalized = (raw - mean) / std
        coarse_context = downsample_for_zoom(
            normalized, process_config.coarse_zoom, multiple=8
        )
        coarse_mask = downsample_for_zoom(
            mask, process_config.coarse_zoom, multiple=8, nearest=True
        )
        coarse_feather = max(
            0, int(round(process_config.feather / process_config.coarse_zoom))
        )
        print(
            f"[tile {entry.index:02d}] coarse "
            f"{coarse_context.shape[1]}x{coarse_context.shape[0]}"
        )
        coarse_normalized = repaint_latent(
            model,
            scheduler,
            coarse_context,
            coarse_mask,
            zoom=process_config.coarse_zoom,
            steps=process_config.steps,
            jump=process_config.jump,
            repeats=process_config.repeats,
            seed=config.repaint.seed + entry.index * 1009,
            device=device,
            compute_dtype=compute_dtype,
            strength=1.0,
            feather=coarse_feather,
        )
        initial = resize_hwc(
            coarse_normalized,
            height=raw_set.tile_height,
            width=raw_set.tile_width,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[tile {entry.index:02d}] fine {raw_set.tile_width}x{raw_set.tile_height}")
        final_normalized = repaint_latent(
            model,
            scheduler,
            normalized,
            mask,
            initial=initial,
            strength=process_config.fine_strength,
            zoom=process_config.fine_zoom,
            steps=process_config.steps,
            jump=process_config.jump,
            repeats=process_config.repeats,
            seed=config.repaint.seed + entry.index * 1009 + 1,
            device=device,
            compute_dtype=compute_dtype,
            feather=process_config.feather,
        )
        final = _finite_latent(final_normalized * std + mean)
        output_path = tiles_dir / f"tile_{entry.index:02d}.npy"
        np.save(output_path, final)
        output_entries.append(
            {
                "index": entry.index,
                "edges": entry.edges,
                "latent": _relative(output_path, paths.artifacts),
                "seam": _relative(entry.seam_path, paths.artifacts),
            }
        )

        if entry.index == process_config.preview_tile:
            preview_payload = {
                "entry": entry,
                "raw": raw.copy(),
                "mask": mask.copy(),
                "coarse": _finite_latent(coarse_normalized * std + mean),
                "fine": final.copy(),
            }
        seconds = time.perf_counter() - tile_started
        timings.append({"tile": entry.index, "seconds": seconds})
        print(
            f"[tile {entry.index:02d}] {counter}/{len(entries)} complete | "
            f"{format_duration(seconds)}"
        )
        del raw, seam, mask, normalized, coarse_context, coarse_mask
        del coarse_normalized, initial, final_normalized, final
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    memory_stats = cuda_memory_stats(device)
    del model, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    complete = raw_set.complete and len(output_entries) == 16
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "complete": complete,
            "source_manifest": str(raw_set.manifest_path),
            "tile_height": raw_set.tile_height,
            "tile_width": raw_set.tile_width,
            "channels": raw_set.channels,
            "tiles": output_entries,
        },
    )

    if config.repaint.visualization.enabled and preview_payload is not None:
        paths.visualization.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            _clear_stage_previews(paths.visualization)
        source_model = load_source_model(artifact, device)
        visual = config.repaint.visualization
        preview_tile = int(preview_payload["entry"].index)
        preview_stem = f"tile_{preview_tile:02d}"
        mask_image = np.round(
            np.clip(preview_payload["mask"][..., 0], 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        Image.fromarray(mask_image).save(
            paths.visualization / f"{preview_stem}_cross_mask.png"
        )
        for name in ("raw", "coarse", "fine"):
            rendered = _render_preview(
                source_model,
                preview_payload[name],
                artifact=artifact,
                full_height=raw_set.tile_height,
                full_width=raw_set.tile_width,
                maximum=visual.preview_size,
                device=device,
            )
            _save_preview_stage(
                paths.visualization,
                f"{preview_stem}_{name}",
                rendered,
                save_offset=visual.save_offset,
                save_exr=visual.save_exr,
            )

        if complete and not args.first_tile_only:
            grid_rgb = []
            grid_offset = []
            individual = paths.visualization / "tiles"
            individual.mkdir(parents=True, exist_ok=True)
            entry_by_index = {entry.index: entry for entry in raw_set.entries}
            for record in sorted(output_entries, key=lambda value: value["index"]):
                tile = np.load(paths.artifacts / record["latent"])
                rendered = _render_preview(
                    source_model,
                    tile,
                    artifact=artifact,
                    full_height=raw_set.tile_height,
                    full_width=raw_set.tile_width,
                    maximum=min(visual.preview_size, 512),
                    device=device,
                )
                stem = f"tile_{record['index']:02d}"
                write_png(str(individual / f"{stem}_rgb.png"), rendered.rgb)
                grid_rgb.append(
                    _edge_frame(to_display(rendered.rgb), entry_by_index[record["index"]].edges)
                )
                if visual.save_offset:
                    write_png(
                        str(individual / f"{stem}_offset.png"),
                        rendered.offset,
                        srgb=False,
                    )
                    offset = np.round(np.clip(rendered.offset, 0.0, 1.0) * 255.0).astype(np.uint8)
                    grid_offset.append(
                        _edge_frame(offset, entry_by_index[record["index"]].edges)
                    )
            _save_grid(grid_rgb, paths.visualization / "repainted_tiles_rgb.png")
            if grid_offset:
                _save_grid(grid_offset, paths.visualization / "repainted_tiles_offset.png")
        del source_model

    elapsed = time.perf_counter() - started
    resolved = config.as_dict()
    resolved["output_dir"] = str(output_root)
    write_json(paths.artifacts / "config.resolved.json", resolved)
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "source_tile_set": str(raw_set.manifest_path),
            "diffusion_checkpoint": str(checkpoint_path),
            "diffusion_epoch": int(checkpoint["epoch"]),
            "diffusion_global_step": int(checkpoint["global_step"]),
            "processed_tiles": [entry["index"] for entry in output_entries],
            "preview_tile": (
                int(preview_payload["entry"].index)
                if preview_payload is not None
                else None
            ),
            "complete": complete,
            "precision": str(compute_dtype).removeprefix("torch."),
            "coarse_zoom": process_config.coarse_zoom,
            "fine_zoom": process_config.fine_zoom,
            "fine_strength": process_config.fine_strength,
            "timings": timings,
            "elapsed_seconds": elapsed,
            "peak_allocated_vram_bytes": memory_stats["peak_allocated_bytes"],
            "peak_reserved_vram_bytes": memory_stats["peak_reserved_bytes"],
            "noise_rolling": False,
        },
    )
    print(f"[saved] {manifest_path}")
    print(f"[done] {len(entries)} tile(s), {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
