"""Stable on-disk artifact contracts shared by synthesis stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..config import load_model

from .latent import LatentLayout


@dataclass(frozen=True)
class LatentArtifact:
    synthesis_root: Path
    latent_path: Path
    metadata_path: Path
    metadata: Dict[str, Any]
    layout: LatentLayout

    def load(self, mmap_mode: Optional[str] = None) -> np.ndarray:
        latent = np.load(self.latent_path, mmap_mode=mmap_mode)
        expected = (
            int(self.metadata["height"]),
            int(self.metadata["width"]),
            int(self.metadata["channels"]),
        )
        if latent.shape != expected:
            raise ValueError(
                f"latent shape {latent.shape} does not match metadata {expected}"
            )
        return latent


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return values


def write_json(path: Path, values: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2)
        handle.write("\n")


def load_latent_artifact(synthesis_root: Path) -> LatentArtifact:
    root = synthesis_root.resolve()
    stage = root / "latent"
    latent_path = stage / "latent.npy"
    metadata_path = stage / "metadata.json"
    if not latent_path.is_file():
        raise FileNotFoundError(
            f"missing extracted latent: {latent_path}; run extract_latents.py first"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing latent metadata: {metadata_path}")

    metadata = read_json(metadata_path)
    if "layout" not in metadata:
        raise ValueError(f"latent metadata has no channel layout: {metadata_path}")
    layout = LatentLayout.from_dict(metadata["layout"])
    if int(metadata.get("channels", -1)) != layout.total_channels:
        raise ValueError("latent metadata channel count disagrees with its layout")
    return LatentArtifact(root, latent_path, metadata_path, metadata, layout)


def source_checkpoint(artifact: LatentArtifact) -> Path:
    value = artifact.metadata.get("checkpoint")
    if not value:
        raise ValueError(f"checkpoint is missing from {artifact.metadata_path}")
    checkpoint = Path(str(value)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"source BTF checkpoint not found: {checkpoint}")
    return checkpoint


def load_source_model(
    artifact: LatentArtifact,
    device: torch.device,
):
    return load_model(str(source_checkpoint(artifact)), device=device)[0]


def preview_directions(
    artifact: LatentArtifact,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    preview = artifact.metadata.get("preview", {})
    light = preview.get("light_xy")
    view = preview.get("view_xy")
    if light is None or view is None:
        raise ValueError(
            f"fixed preview directions are missing from {artifact.metadata_path}"
        )
    return (
        (float(light[0]), float(light[1])),
        (float(view[0]), float(view[1])),
    )


def source_uv_scale(
    artifact: LatentArtifact,
    *,
    domain_height: int,
    domain_width: int,
) -> Tuple[float, float]:
    return (
        float(domain_width) / float(artifact.metadata["width"]),
        float(domain_height) / float(artifact.metadata["height"]),
    )


def prepare_stage(
    directory: Path,
    known_outputs: Tuple[Path, ...],
    *,
    overwrite: bool,
) -> None:
    existing = [path for path in known_outputs if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"stage outputs already exist: {names}; pass --overwrite")
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TileSetEntry:
    index: int
    edges: Dict[str, str]
    latent_path: Path
    seam_path: Optional[Path]


@dataclass(frozen=True)
class TileSetArtifact:
    root: Path
    manifest_path: Path
    tile_height: int
    tile_width: int
    channels: int
    complete: bool
    entries: Tuple[TileSetEntry, ...]

    def load_latents(self) -> list[np.ndarray]:
        result = []
        for entry in self.entries:
            latent = np.load(entry.latent_path)
            expected = (self.tile_height, self.tile_width, self.channels)
            if latent.shape != expected:
                raise ValueError(
                    f"tile {entry.index} shape {latent.shape} does not match {expected}"
                )
            result.append(np.asarray(latent, dtype=np.float32))
        return result


def load_tile_set(stage: Path) -> TileSetArtifact:
    """Load a raw or repainted Wang tile manifest with resolved artifact paths."""
    root = stage.resolve()
    manifest_path = root / "tile_set.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Wang tile manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported Wang tile schema in {manifest_path}")
    records = manifest.get("tiles")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Wang tile manifest has no tiles: {manifest_path}")

    entries = []
    seen = set()
    required_edges = {"north", "east", "south", "west"}
    for record in records:
        index = int(record["index"])
        if index in seen:
            raise ValueError(f"duplicate Wang tile index {index}")
        seen.add(index)
        edges = dict(record["edges"])
        if set(edges) != required_edges:
            raise ValueError(f"tile {index} has invalid edge labels")
        latent_path = (root / record["latent"]).resolve()
        if not latent_path.is_file():
            raise FileNotFoundError(f"missing Wang tile latent: {latent_path}")
        seam_value = record.get("seam")
        seam_path = (root / seam_value).resolve() if seam_value else None
        if seam_path is not None and not seam_path.is_file():
            raise FileNotFoundError(f"missing Wang tile seam: {seam_path}")
        entries.append(TileSetEntry(index, edges, latent_path, seam_path))

    entries.sort(key=lambda entry: entry.index)
    complete = bool(manifest.get("complete", len(entries) == 16))
    if complete and [entry.index for entry in entries] != list(range(16)):
        raise ValueError("a complete Wang tile set must contain indices 0 through 15")
    return TileSetArtifact(
        root=root,
        manifest_path=manifest_path,
        tile_height=int(manifest["tile_height"]),
        tile_width=int(manifest["tile_width"]),
        channels=int(manifest["channels"]),
        complete=complete,
        entries=tuple(entries),
    )
