"""Latent-space synthesis for trained neural BTF materials."""

from .artifacts import (
    LatentArtifact,
    TileSetArtifact,
    load_latent_artifact,
    load_tile_set,
)
from .config import (
    StagePaths,
    SynthesisConfig,
    load_synthesis_config,
    synthesis_stage_paths,
)
from .latent import LatentLayout, extract_latent_stack
from .periodicity import PeriodAnalysis, analyze_periodicity, select_tile_shape
from .rendering import LatentRender, render_latent_stack, render_model
from .repaint import build_cross_mask, repaint_latent
from .wang import (
    TileEdges,
    WangTile,
    build_corner_lookup,
    build_tile_set,
    generate_tile_grid,
    optimize_diamonds,
    write_tile_grid,
)

__all__ = [
    "LatentArtifact",
    "LatentLayout",
    "LatentRender",
    "PeriodAnalysis",
    "StagePaths",
    "SynthesisConfig",
    "TileEdges",
    "TileSetArtifact",
    "WangTile",
    "analyze_periodicity",
    "build_corner_lookup",
    "build_cross_mask",
    "build_tile_set",
    "extract_latent_stack",
    "generate_tile_grid",
    "load_latent_artifact",
    "load_synthesis_config",
    "load_tile_set",
    "optimize_diamonds",
    "render_latent_stack",
    "render_model",
    "repaint_latent",
    "select_tile_shape",
    "synthesis_stage_paths",
    "write_tile_grid",
]
