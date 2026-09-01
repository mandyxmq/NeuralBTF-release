"""Typed configuration and path conventions for synthesis stages."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PreviewConfig:
    enabled: bool = True
    light_xy: Tuple[float, float] = (0.48712474, 0.14440002)
    view_xy: Tuple[float, float] = (-0.6001169, 0.1504723)
    chunk_size: int = 262144
    save_exr: bool = False

    def validate(self) -> None:
        _direction_xy(self.light_xy, "extract.preview.light_xy")
        _direction_xy(self.view_xy, "extract.preview.view_xy")
        _positive(self.chunk_size, "extract.preview.chunk_size")


@dataclass(frozen=True)
class ExtractConfig:
    height: Optional[int] = None
    width: Optional[int] = None
    preview: PreviewConfig = PreviewConfig()

    def validate(self) -> None:
        for name, value in (("height", self.height), ("width", self.width)):
            if value is not None:
                _positive(value, f"extract.{name}")
        self.preview.validate()


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = True
    preview_size: int = 512
    save_offset: bool = True
    save_exr: bool = False

    def validate(self, section: str) -> None:
        _positive(self.preview_size, f"{section}.preview_size")


@dataclass(frozen=True)
class DiffusionDataConfig:
    crop_size: int = 320
    samples_per_epoch: int = 30000
    validation_samples: int = 600
    batch_size: int = 6
    zoom_range: Tuple[float, float] = (1.0, 5.0)
    extremes_probability: float = 0.3
    workers: int = 4

    def validate(self) -> None:
        _positive(self.crop_size, "diffusion.data.crop_size")
        _positive(self.samples_per_epoch, "diffusion.data.samples_per_epoch")
        if self.validation_samples < 0:
            raise ValueError("diffusion.data.validation_samples must be non-negative")
        _positive(self.batch_size, "diffusion.data.batch_size")
        if len(self.zoom_range) != 2:
            raise ValueError("diffusion.data.zoom_range must contain two values")
        if self.zoom_range[0] < 1.0 or self.zoom_range[1] < self.zoom_range[0]:
            raise ValueError("diffusion.data.zoom_range must satisfy 1 <= min <= max")
        if not 0.0 <= self.extremes_probability <= 0.5:
            raise ValueError(
                "diffusion.data.extremes_probability must be between 0 and 0.5"
            )
        if self.workers < 0:
            raise ValueError("diffusion.data.workers must be non-negative")


@dataclass(frozen=True)
class DiffusionScheduleConfig:
    train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "v_prediction"
    clip_sample: bool = False
    set_alpha_to_one: bool = False
    steps_offset: int = 1

    def validate(self) -> None:
        _positive(self.train_timesteps, "diffusion.schedule.train_timesteps")
        if self.prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError(
                "diffusion.schedule.prediction_type must be epsilon or v_prediction"
            )


@dataclass(frozen=True)
class DiffusionModelConfig:
    base_channels: int = 128
    channel_multipliers: Tuple[int, ...] = (1, 2, 2, 4)
    layers_per_block: int = 2
    norm_groups: int = 8
    attention: bool = False
    attention_head_dim: int = 32
    attention_last_only: bool = False

    def validate(self) -> None:
        _positive(self.base_channels, "diffusion.model.base_channels")
        if not self.channel_multipliers:
            raise ValueError("diffusion.model.channel_multipliers cannot be empty")
        for value in self.channel_multipliers:
            _positive(value, "diffusion.model.channel_multipliers")
        _positive(self.layers_per_block, "diffusion.model.layers_per_block")
        _positive(self.norm_groups, "diffusion.model.norm_groups")
        _positive(self.attention_head_dim, "diffusion.model.attention_head_dim")
        for channels in self.block_channels:
            if channels % self.norm_groups:
                raise ValueError(
                    "every diffusion model block must be divisible by norm_groups"
                )
            if self.attention and channels < self.attention_head_dim:
                raise ValueError(
                    "attention_head_dim cannot exceed a diffusion block width"
                )

    @property
    def block_channels(self) -> Tuple[int, ...]:
        return tuple(self.base_channels * value for value in self.channel_multipliers)


@dataclass(frozen=True)
class DiffusionTrainConfig:
    epochs: int = 3
    max_steps: Optional[int] = None
    learning_rate: float = 8e-5
    weight_decay: float = 1e-2
    min_snr_gamma: float = 1.0
    ema_decay: float = 0.999
    gradient_clip: float = 1.0
    print_every: int = 100

    def validate(self) -> None:
        _positive(self.epochs, "diffusion.train.epochs")
        if self.max_steps is not None:
            _positive(self.max_steps, "diffusion.train.max_steps")
        _positive(self.learning_rate, "diffusion.train.learning_rate")
        if self.weight_decay < 0.0:
            raise ValueError("diffusion.train.weight_decay must be non-negative")
        _positive(self.min_snr_gamma, "diffusion.train.min_snr_gamma")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("diffusion.train.ema_decay must lie between 0 and 1")
        _positive(self.gradient_clip, "diffusion.train.gradient_clip")
        _positive(self.print_every, "diffusion.train.print_every")


@dataclass(frozen=True)
class DiffusionPreviewConfig:
    enabled: bool = True
    every_epochs: int = 1
    size: int = 512
    sampling_steps: int = 50
    seed: int = 0
    eta: float = 0.0
    roll_during_sampling: bool = True
    save_offset: bool = True

    def validate(self) -> None:
        _positive(self.every_epochs, "diffusion.preview.every_epochs")
        _positive(self.size, "diffusion.preview.size")
        _positive(self.sampling_steps, "diffusion.preview.sampling_steps")
        if self.eta < 0.0:
            raise ValueError("diffusion.preview.eta must be non-negative")


@dataclass(frozen=True)
class DiffusionConfig:
    seed: int = 0
    data: DiffusionDataConfig = DiffusionDataConfig()
    schedule: DiffusionScheduleConfig = DiffusionScheduleConfig()
    model: DiffusionModelConfig = DiffusionModelConfig()
    train: DiffusionTrainConfig = DiffusionTrainConfig()
    preview: DiffusionPreviewConfig = DiffusionPreviewConfig()

    def validate(self) -> None:
        self.data.validate()
        self.schedule.validate()
        self.model.validate()
        self.train.validate()
        self.preview.validate()


@dataclass(frozen=True)
class PeriodicityConfig:
    enabled: bool = True
    prominence_ratio: float = 0.1
    minimum_peak_distance: int = 20

    def validate(self) -> None:
        if not 0.0 < self.prominence_ratio < 1.0:
            raise ValueError("wang_tiles.periodicity.prominence_ratio must lie in (0, 1)")
        _positive(
            self.minimum_peak_distance,
            "wang_tiles.periodicity.minimum_peak_distance",
        )


@dataclass(frozen=True)
class WangTileShapeConfig:
    mode: str = "auto"
    target_size: Optional[int] = None
    target_size_ratio: float = 0.75
    fallback_size: int = 1024
    divisor: int = 8
    enforce_square: bool = False
    height: Optional[int] = None
    width: Optional[int] = None

    def validate(self) -> None:
        if self.mode not in ("auto", "manual"):
            raise ValueError("wang_tiles.tile.mode must be auto or manual")
        if self.target_size is not None:
            _positive(self.target_size, "wang_tiles.tile.target_size")
        _positive(self.target_size_ratio, "wang_tiles.tile.target_size_ratio")
        _positive(self.fallback_size, "wang_tiles.tile.fallback_size")
        _positive(self.divisor, "wang_tiles.tile.divisor")
        if self.mode == "manual":
            if self.height is None or self.width is None:
                raise ValueError(
                    "wang_tiles.tile manual mode requires both height and width"
                )
            _positive(self.height, "wang_tiles.tile.height")
            _positive(self.width, "wang_tiles.tile.width")


@dataclass(frozen=True)
class WangSearchConfig:
    iterations: int = 300
    pixel_appearance: float = 1.0
    pixel_offset: float = 2.0
    seam_appearance: float = 1.0
    seam_offset: float = 2.0

    def validate(self) -> None:
        if self.iterations < 0:
            raise ValueError("wang_tiles.search.iterations must be non-negative")
        for name, value in (
            ("pixel_appearance", self.pixel_appearance),
            ("pixel_offset", self.pixel_offset),
            ("seam_appearance", self.seam_appearance),
            ("seam_offset", self.seam_offset),
        ):
            if value < 0.0:
                raise ValueError(f"wang_tiles.search.{name} must be non-negative")


@dataclass(frozen=True)
class WangTilesConfig:
    seed: int = 0
    periodicity: PeriodicityConfig = PeriodicityConfig()
    tile: WangTileShapeConfig = WangTileShapeConfig()
    search: WangSearchConfig = WangSearchConfig()
    visualization: VisualizationConfig = VisualizationConfig(preview_size=320)

    def validate(self) -> None:
        self.periodicity.validate()
        self.tile.validate()
        self.search.validate()
        self.visualization.validate("wang_tiles.visualization")


@dataclass(frozen=True)
class RepaintMaskConfig:
    core_width: int = 48
    taper_power: float = 0.3
    flat_ratio: float = 0.8

    def validate(self) -> None:
        _positive(self.core_width, "repaint.mask.core_width")
        _positive(self.taper_power, "repaint.mask.taper_power")
        if not 0.0 < self.flat_ratio <= 1.0:
            raise ValueError("repaint.mask.flat_ratio must lie in (0, 1]")


@dataclass(frozen=True)
class RepaintProcessConfig:
    steps: int = 50
    jump: int = 5
    repeats: int = 1
    coarse_zoom: float = 5.0
    fine_zoom: float = 1.0
    fine_strength: float = 0.65
    feather: int = 8
    precision: str = "auto"
    preview_tile: int = 0

    def validate(self) -> None:
        _positive(self.steps, "repaint.process.steps")
        _positive(self.jump, "repaint.process.jump")
        _positive(self.repeats, "repaint.process.repeats")
        _positive(self.coarse_zoom, "repaint.process.coarse_zoom")
        _positive(self.fine_zoom, "repaint.process.fine_zoom")
        if not 0.0 < self.fine_strength <= 1.0:
            raise ValueError("repaint.process.fine_strength must lie in (0, 1]")
        if self.feather < 0:
            raise ValueError("repaint.process.feather must be non-negative")
        if self.precision not in ("auto", "float32", "float16"):
            raise ValueError(
                "repaint.process.precision must be auto, float32, or float16"
            )
        if self.preview_tile < 0 or self.preview_tile >= 16:
            raise ValueError("repaint.process.preview_tile must lie in [0, 15]")


@dataclass(frozen=True)
class RepaintConfig:
    seed: int = 0
    mask: RepaintMaskConfig = RepaintMaskConfig()
    process: RepaintProcessConfig = RepaintProcessConfig()
    visualization: VisualizationConfig = VisualizationConfig()

    def validate(self) -> None:
        self.mask.validate()
        self.process.validate()
        self.visualization.validate("repaint.visualization")


@dataclass(frozen=True)
class TilingConfig:
    source: str = "repaint"
    rows: int = 4
    columns: int = 4
    seed: int = 20
    periodic: bool = True
    visualization: VisualizationConfig = VisualizationConfig(preview_size=2048)

    def validate(self) -> None:
        if self.source not in ("repaint", "raw"):
            raise ValueError("tiling.source must be repaint or raw")
        _positive(self.rows, "tiling.rows")
        _positive(self.columns, "tiling.columns")
        self.visualization.validate("tiling.visualization")


@dataclass(frozen=True)
class SynthesisConfig:
    output_dir: str = ""
    device: str = "auto"
    extract: ExtractConfig = ExtractConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    wang_tiles: WangTilesConfig = WangTilesConfig()
    repaint: RepaintConfig = RepaintConfig()
    tiling: TilingConfig = TilingConfig()

    def validate(self) -> None:
        self.extract.validate()
        self.diffusion.validate()
        self.wang_tiles.validate()
        self.repaint.validate()
        self.tiling.validate()

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def synchronize_repaint_zooms(config: SynthesisConfig) -> SynthesisConfig:
    """Copy diffusion zoom endpoints into the explicit RePaint settings."""
    fine_zoom, coarse_zoom = config.diffusion.data.zoom_range
    process = config.repaint.process
    if (
        process.coarse_zoom == coarse_zoom
        and process.fine_zoom == fine_zoom
    ):
        return config
    return replace(
        config,
        repaint=replace(
            config.repaint,
            process=replace(
                process,
                coarse_zoom=coarse_zoom,
                fine_zoom=fine_zoom,
            ),
        ),
    )


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _only_known(values: Dict[str, Any], known: set[str], section: str) -> None:
    unknown = set(values) - known
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown {section} configuration key(s): {names}")


def _mapping(values: Dict[str, Any], key: str, section: str) -> Dict[str, Any]:
    result = values.get(key, {})
    if not isinstance(result, dict):
        raise ValueError(f"{section} must be a JSON object")
    return result


def _direction_xy(value: Sequence[float], name: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    x, y = float(value[0]), float(value[1])
    if x * x + y * y > 1.0 + 1e-6:
        raise ValueError(f"{name} must lie on the projected upper hemisphere")
    return x, y


def _float_pair(value: Sequence[float], name: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    return float(value[0]), float(value[1])


def _int_tuple(value: Sequence[int], name: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return tuple(int(item) for item in value)


def _construct(cls, values: Dict[str, Any], section: str):
    names = set(cls.__dataclass_fields__)
    _only_known(values, names, section)
    return cls(**values)


def _visualization(
    values: Dict[str, Any],
    section: str,
    defaults: VisualizationConfig,
) -> VisualizationConfig:
    _only_known(values, set(VisualizationConfig.__dataclass_fields__), section)
    return VisualizationConfig(
        enabled=values.get("enabled", defaults.enabled),
        preview_size=values.get("preview_size", defaults.preview_size),
        save_offset=values.get("save_offset", defaults.save_offset),
        save_exr=values.get("save_exr", defaults.save_exr),
    )


def load_synthesis_config(path: Optional[str] = None) -> SynthesisConfig:
    """Load an immutable project config and reject misspelled keys."""
    values: Dict[str, Any] = {}
    if path:
        with open(path, encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("the synthesis config root must be a JSON object")

    _only_known(
        values,
        {
            "output_dir",
            "device",
            "extract",
            "diffusion",
            "wang_tiles",
            "repaint",
            "tiling",
        },
        "root",
    )

    extract_values = _mapping(values, "extract", "extract")
    _only_known(extract_values, {"height", "width", "preview"}, "extract")
    preview_values = _mapping(extract_values, "preview", "extract.preview")
    _only_known(
        preview_values,
        {"enabled", "light_xy", "view_xy", "chunk_size", "save_exr"},
        "extract.preview",
    )
    preview_defaults = PreviewConfig()
    preview = PreviewConfig(
        enabled=preview_values.get("enabled", preview_defaults.enabled),
        light_xy=_direction_xy(
            preview_values.get("light_xy", preview_defaults.light_xy),
            "extract.preview.light_xy",
        ),
        view_xy=_direction_xy(
            preview_values.get("view_xy", preview_defaults.view_xy),
            "extract.preview.view_xy",
        ),
        chunk_size=preview_values.get("chunk_size", preview_defaults.chunk_size),
        save_exr=preview_values.get("save_exr", preview_defaults.save_exr),
    )
    extract = ExtractConfig(
        height=extract_values.get("height"),
        width=extract_values.get("width"),
        preview=preview,
    )

    diffusion_values = _mapping(values, "diffusion", "diffusion")
    _only_known(
        diffusion_values,
        {"seed", "data", "schedule", "model", "train", "preview"},
        "diffusion",
    )
    data_values = _mapping(diffusion_values, "data", "diffusion.data")
    data_defaults = DiffusionDataConfig()
    _only_known(data_values, set(DiffusionDataConfig.__dataclass_fields__), "diffusion.data")
    data = DiffusionDataConfig(
        crop_size=data_values.get("crop_size", data_defaults.crop_size),
        samples_per_epoch=data_values.get(
            "samples_per_epoch", data_defaults.samples_per_epoch
        ),
        validation_samples=data_values.get(
            "validation_samples", data_defaults.validation_samples
        ),
        batch_size=data_values.get("batch_size", data_defaults.batch_size),
        zoom_range=_float_pair(
            data_values.get("zoom_range", data_defaults.zoom_range),
            "diffusion.data.zoom_range",
        ),
        extremes_probability=data_values.get(
            "extremes_probability", data_defaults.extremes_probability
        ),
        workers=data_values.get("workers", data_defaults.workers),
    )
    schedule = _construct(
        DiffusionScheduleConfig,
        _mapping(diffusion_values, "schedule", "diffusion.schedule"),
        "diffusion.schedule",
    )
    model_values = _mapping(diffusion_values, "model", "diffusion.model")
    model_defaults = DiffusionModelConfig()
    _only_known(
        model_values,
        set(DiffusionModelConfig.__dataclass_fields__),
        "diffusion.model",
    )
    model = DiffusionModelConfig(
        base_channels=model_values.get("base_channels", model_defaults.base_channels),
        channel_multipliers=_int_tuple(
            model_values.get(
                "channel_multipliers", model_defaults.channel_multipliers
            ),
            "diffusion.model.channel_multipliers",
        ),
        layers_per_block=model_values.get(
            "layers_per_block", model_defaults.layers_per_block
        ),
        norm_groups=model_values.get("norm_groups", model_defaults.norm_groups),
        attention=model_values.get("attention", model_defaults.attention),
        attention_head_dim=model_values.get(
            "attention_head_dim", model_defaults.attention_head_dim
        ),
        attention_last_only=model_values.get(
            "attention_last_only", model_defaults.attention_last_only
        ),
    )
    train = _construct(
        DiffusionTrainConfig,
        _mapping(diffusion_values, "train", "diffusion.train"),
        "diffusion.train",
    )
    diffusion_preview = _construct(
        DiffusionPreviewConfig,
        _mapping(diffusion_values, "preview", "diffusion.preview"),
        "diffusion.preview",
    )
    diffusion = DiffusionConfig(
        seed=diffusion_values.get("seed", 0),
        data=data,
        schedule=schedule,
        model=model,
        train=train,
        preview=diffusion_preview,
    )

    wang_values = _mapping(values, "wang_tiles", "wang_tiles")
    _only_known(
        wang_values,
        {"seed", "periodicity", "tile", "search", "visualization"},
        "wang_tiles",
    )
    wang_visual_defaults = VisualizationConfig(preview_size=320)
    wang = WangTilesConfig(
        seed=wang_values.get("seed", 0),
        periodicity=_construct(
            PeriodicityConfig,
            _mapping(wang_values, "periodicity", "wang_tiles.periodicity"),
            "wang_tiles.periodicity",
        ),
        tile=_construct(
            WangTileShapeConfig,
            _mapping(wang_values, "tile", "wang_tiles.tile"),
            "wang_tiles.tile",
        ),
        search=_construct(
            WangSearchConfig,
            _mapping(wang_values, "search", "wang_tiles.search"),
            "wang_tiles.search",
        ),
        visualization=_visualization(
            _mapping(wang_values, "visualization", "wang_tiles.visualization"),
            "wang_tiles.visualization",
            wang_visual_defaults,
        ),
    )

    repaint_values = _mapping(values, "repaint", "repaint")
    _only_known(
        repaint_values,
        {"seed", "mask", "process", "visualization"},
        "repaint",
    )
    repaint = RepaintConfig(
        seed=repaint_values.get("seed", 0),
        mask=_construct(
            RepaintMaskConfig,
            _mapping(repaint_values, "mask", "repaint.mask"),
            "repaint.mask",
        ),
        process=_construct(
            RepaintProcessConfig,
            _mapping(repaint_values, "process", "repaint.process"),
            "repaint.process",
        ),
        visualization=_visualization(
            _mapping(repaint_values, "visualization", "repaint.visualization"),
            "repaint.visualization",
            VisualizationConfig(),
        ),
    )

    tiling_values = _mapping(values, "tiling", "tiling")
    _only_known(
        tiling_values,
        {"source", "rows", "columns", "seed", "periodic", "visualization"},
        "tiling",
    )
    tiling = TilingConfig(
        source=tiling_values.get("source", "repaint"),
        rows=tiling_values.get("rows", 4),
        columns=tiling_values.get("columns", 4),
        seed=tiling_values.get("seed", 20),
        periodic=tiling_values.get("periodic", True),
        visualization=_visualization(
            _mapping(tiling_values, "visualization", "tiling.visualization"),
            "tiling.visualization",
            VisualizationConfig(preview_size=2048),
        ),
    )

    config = SynthesisConfig(
        output_dir=values.get("output_dir", ""),
        device=values.get("device", "auto"),
        extract=extract,
        diffusion=diffusion,
        wang_tiles=wang,
        repaint=repaint,
        tiling=tiling,
    )
    config.validate()
    return config


def resolve_config_path(value: str, config_path: Optional[str]) -> Path:
    """Resolve a user path relative to its JSON file, or the current directory."""
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if path.is_absolute():
        return path.resolve()
    base = Path(config_path).resolve().parent if config_path else Path.cwd()
    return (base / path).resolve()


def default_synthesis_root(checkpoint: Path) -> Path:
    """Place synthesis beside a release training run when its layout is recognised."""
    if checkpoint.parent.name == "result":
        return checkpoint.parent.parent / "synthesis"
    return checkpoint.parent / "synthesis"


def resolve_output_root(
    config: SynthesisConfig,
    *,
    config_path: Optional[str],
    override: str = "",
) -> Path:
    """Resolve the synthesis root from CLI or an extraction project config."""
    if override:
        return resolve_config_path(override, None)
    if config.output_dir:
        return resolve_config_path(config.output_dir, config_path)
    raise ValueError(
        "set --output-dir or use the project.json generated by extraction"
    )


@dataclass(frozen=True)
class StagePaths:
    """Artifact and visualization directories for one synthesis stage."""

    root: Path
    artifacts: Path
    visualization: Path


def synthesis_stage_paths(output_root: Path, stage: str) -> StagePaths:
    """Return the stable public output paths for a synthesis stage."""
    if not stage or Path(stage).name != stage:
        raise ValueError("stage must be one directory name")
    root = output_root.resolve()
    return StagePaths(
        root=root,
        artifacts=root / stage,
        visualization=root / "visualization" / stage,
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)
