#!/usr/bin/env python3
"""Construct the 16 raw latent Wang tiles with weighted graph cuts."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from neuralbtf import write_exr, write_png  # noqa: E402
from neuralbtf.png import to_display  # noqa: E402
from neuralbtf.synthesis import render_latent_stack  # noqa: E402
from neuralbtf.synthesis.artifacts import (load_latent_artifact,  # noqa: E402
                                           load_source_model,
                                           prepare_stage,
                                           preview_directions,
                                           source_uv_scale,
                                           write_json)
from neuralbtf.synthesis.config import (load_synthesis_config,  # noqa: E402
                                        resolve_device, resolve_output_root,
                                        synthesis_stage_paths)
from neuralbtf.synthesis.periodicity import (analyze_periodicity,  # noqa: E402
                                              select_tile_shape)
from neuralbtf.synthesis.wang import (build_corner_lookup,  # noqa: E402
                                      build_tile_set, optimize_diamonds)


EDGE_COLORS = {
    "v0": (255, 0, 0),
    "v1": (0, 255, 0),
    "h0": (0, 0, 255),
    "h1": (255, 255, 0),
}


def _fit_size(height: int, width: int, maximum: int) -> Tuple[int, int]:
    scale = min(float(maximum) / height, float(maximum) / width, 1.0)
    result_h = max(8, int(round(height * scale)))
    result_w = max(8, int(round(width * scale)))
    return result_h, result_w


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32)).view(1, 1, *mask.shape)
    resized = F.interpolate(tensor, size=(height, width), mode="nearest")
    return resized[0, 0].numpy() > 0.5


def _edge_frame(image: np.ndarray, edges, relative_width: float = 0.015) -> Image.Image:
    base = Image.fromarray(image)
    width = max(3, int(round(min(base.size) * relative_width)))
    framed = Image.new(
        "RGB",
        (base.width + 2 * width, base.height + 2 * width),
        (255, 255, 255),
    )
    framed.paste(base, (width, width))
    draw = ImageDraw.Draw(framed)
    draw.rectangle((0, 0, framed.width, width - 1), fill=EDGE_COLORS[edges.north])
    draw.rectangle(
        (0, framed.height - width, framed.width, framed.height),
        fill=EDGE_COLORS[edges.south],
    )
    draw.rectangle((0, 0, width - 1, framed.height), fill=EDGE_COLORS[edges.west])
    draw.rectangle(
        (framed.width - width, 0, framed.width, framed.height),
        fill=EDGE_COLORS[edges.east],
    )
    return framed


def _save_grid(images, path: Path) -> None:
    if len(images) != 16:
        raise ValueError("the complete Wang set must contain 16 images")
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    canvas = Image.new("RGB", (4 * cell_w, 4 * cell_h), (255, 255, 255))
    for index, image in enumerate(images):
        row, column = divmod(index, 4)
        x = column * cell_w + (cell_w - image.width) // 2
        y = row * cell_h + (cell_h - image.height) // 2
        canvas.paste(image, (x, y))
    canvas.save(path)


def _plot_periodicity(analysis, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    height, width = analysis.height, analysis.width
    axes[0].imshow(
        analysis.autocorrelation,
        cmap="viridis",
        extent=(-width // 2, width // 2, height // 2, -height // 2),
    )
    axes[0].set_title("Latent autocorrelation")
    axes[1].plot(np.arange(width) - width // 2, analysis.profile_x)
    axes[1].set_title(f"Horizontal period: {analysis.period_x}")
    axes[2].plot(np.arange(height) - height // 2, analysis.profile_y)
    axes[2].set_title(f"Vertical period: {analysis.period_y}")
    for axis in axes[1:]:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_search(history: list, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    iteration = [entry["iteration"] for entry in history]
    axis.plot(
        iteration,
        [entry["best_loss"] for entry in history],
        linewidth=2,
        label="best combined",
    )
    for key in (
        "pixel_appearance",
        "pixel_offset",
        "seam_appearance",
        "seam_offset",
    ):
        axis.plot(
            iteration,
            [entry[key] for entry in history],
            alpha=0.65,
            label=key.replace("_", " "),
        )
    axis.set_yscale("log")
    axis.set_xlabel("Greedy proposal")
    axis.set_ylabel("Weighted latent error")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _visualize_tiles(
    *,
    tiles,
    artifact,
    config,
    device: torch.device,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    individual = output_dir / "tiles"
    individual.mkdir(parents=True, exist_ok=True)
    model = load_source_model(artifact, device)
    light_xy, view_xy = preview_directions(artifact)
    clean_images = []
    seam_images = []
    offset_images = []

    for tile in tiles:
        preview_h, preview_w = _fit_size(
            tile.latent.shape[0],
            tile.latent.shape[1],
            config.wang_tiles.visualization.preview_size,
        )
        rendered = render_latent_stack(
            model,
            tile.latent,
            artifact.layout,
            light_xy=light_xy,
            view_xy=view_xy,
            output_height=preview_h,
            output_width=preview_w,
            uv_scale=source_uv_scale(
                artifact,
                domain_height=tile.latent.shape[0],
                domain_width=tile.latent.shape[1],
            ),
            device=device,
        )
        stem = f"tile_{tile.index:02d}"
        write_png(str(individual / f"{stem}_rgb.png"), rendered.rgb)
        clean = to_display(rendered.rgb)
        clean_images.append(_edge_frame(clean, tile.edges))

        seam = clean.copy()
        resized_mask = _resize_mask(tile.seam, preview_h, preview_w)
        seam[resized_mask] = np.array([255, 0, 255], dtype=np.uint8)
        seam_images.append(_edge_frame(seam, tile.edges))

        if config.wang_tiles.visualization.save_offset:
            write_png(
                str(individual / f"{stem}_offset.png"),
                rendered.offset,
                srgb=False,
            )
            offset = np.round(
                np.clip(rendered.offset, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            offset_images.append(_edge_frame(offset, tile.edges))
        if config.wang_tiles.visualization.save_exr:
            write_exr(str(individual / f"{stem}_rgb.exr"), rendered.rgb)

    _save_grid(clean_images, output_dir / "raw_tiles_rgb.png")
    _save_grid(seam_images, output_dir / "raw_tiles_rgb_seams.png")
    if offset_images:
        _save_grid(offset_images, output_dir / "raw_tiles_offset.png")


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
        "--overwrite",
        action="store_true",
        help="replace known files in an existing Wang-tile stage",
    )
    args = parser.parse_args(argv)

    config_path = str(Path(args.config).resolve())
    config = load_synthesis_config(config_path)
    output_root = resolve_output_root(
        config,
        config_path=config_path,
        override=args.output_dir,
    )
    paths = synthesis_stage_paths(output_root, "wang_tiles")
    tile_set_path = paths.artifacts / "tile_set.json"
    metadata_path = paths.artifacts / "metadata.json"
    prepare_stage(
        paths.artifacts,
        (tile_set_path, metadata_path),
        overwrite=args.overwrite,
    )
    device = resolve_device(args.device or config.device)
    artifact = load_latent_artifact(output_root)
    latent = artifact.load(mmap_mode="r")
    started = time.perf_counter()

    print("--- Raw Wang tiles ---")
    print(f"latent:       {artifact.latent_path}")
    print(f"output:       {paths.artifacts}")
    print(f"source shape: {latent.shape}")
    print(f"device:       {device}")

    period_config = config.wang_tiles.periodicity
    analysis = None
    if period_config.enabled or config.wang_tiles.tile.mode == "auto":
        print("Analyzing latent periodicity...")
        analysis = analyze_periodicity(
            latent,
            prominence_ratio=period_config.prominence_ratio,
            minimum_peak_distance=period_config.minimum_peak_distance,
        )
        write_json(paths.artifacts / "periodicity.json", analysis.summary())

    selection = select_tile_shape(
        analysis,
        config.wang_tiles.tile,
        source_height=latent.shape[0],
        source_width=latent.shape[1],
    )
    tile_h = int(selection["height"])
    tile_w = int(selection["width"])
    print(
        f"tile shape:   {tile_w}x{tile_h} "
        f"({selection['strategy']}, periods "
        f"{selection.get('period_x')}/{selection.get('period_y')})"
    )

    print("Optimizing four source diamonds...")
    diamonds, history = optimize_diamonds(
        latent,
        height=tile_h,
        width=tile_w,
        layout=artifact.layout,
        config=config.wang_tiles.search,
        seed=config.wang_tiles.seed,
    )
    del latent

    diamonds_dir = paths.artifacts / "diamonds"
    diamonds_dir.mkdir(parents=True, exist_ok=True)
    for key, value in diamonds.items():
        np.save(diamonds_dir / f"{key}.npy", value)
    write_json(paths.artifacts / "search_history.json", {"history": history})

    print("Cutting corners and assembling 16 tiles...")
    lookup = build_corner_lookup(
        diamonds,
        artifact.layout,
        config.wang_tiles.search,
    )
    tiles = build_tile_set(
        lookup,
        height=tile_h,
        width=tile_w,
        channels=artifact.layout.total_channels,
    )

    tile_dir = paths.artifacts / "tiles"
    seam_dir = paths.artifacts / "seams"
    tile_dir.mkdir(parents=True, exist_ok=True)
    seam_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for tile in tiles:
        latent_path = tile_dir / f"tile_{tile.index:02d}.npy"
        seam_path = seam_dir / f"seam_{tile.index:02d}.npy"
        np.save(latent_path, tile.latent.astype(np.float32, copy=False))
        np.save(seam_path, tile.seam)
        entries.append(
            {
                "index": tile.index,
                "edges": tile.edges.as_dict(),
                "latent": str(latent_path.relative_to(paths.artifacts)),
                "seam": str(seam_path.relative_to(paths.artifacts)),
            }
        )
    write_json(
        tile_set_path,
        {
            "schema_version": 1,
            "complete": True,
            "tile_height": tile_h,
            "tile_width": tile_w,
            "channels": artifact.layout.total_channels,
            "tiles": entries,
        },
    )

    if config.wang_tiles.visualization.enabled:
        paths.visualization.mkdir(parents=True, exist_ok=True)
        if analysis is not None:
            _plot_periodicity(
                analysis, paths.visualization / "periodicity.png"
            )
        _plot_search(history, paths.visualization / "search_loss.png")
        _visualize_tiles(
            tiles=tiles,
            artifact=artifact,
            config=config,
            device=device,
            output_dir=paths.visualization,
        )

    elapsed = time.perf_counter() - started
    resolved = config.as_dict()
    resolved["output_dir"] = str(output_root)
    write_json(paths.artifacts / "config.resolved.json", resolved)
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "source_latent": str(artifact.latent_path),
            "source_metadata": str(artifact.metadata_path),
            "tile_set": tile_set_path.name,
            "tile_selection": selection,
            "search_iterations": config.wang_tiles.search.iterations,
            "elapsed_seconds": elapsed,
        },
    )
    print(f"[saved] {tile_set_path}")
    print(f"[done] {elapsed:.2f}s")


if __name__ == "__main__":
    main()
