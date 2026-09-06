#!/usr/bin/env python3
"""CPU checks for the public latent-synthesis API; no pytest required."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralbtf import ModelConfig, MultiResNeuralTextureModel  # noqa: E402
from neuralbtf.synthesis import (  # noqa: E402
    extract_latent_stack,
    load_synthesis_config,
    render_latent_stack,
    render_model,
    synthesis_stage_paths,
)
from neuralbtf.synthesis.config import (  # noqa: E402
    DiffusionConfig,
    DiffusionDataConfig,
    RepaintConfig,
    RepaintProcessConfig,
    SynthesisConfig,
    resolve_output_root,
    synchronize_repaint_zooms,
)


def check(name, function) -> None:
    function()
    print(f"ok    {name}")


def tiny_model(
    offset_mode: str = "2d", use_neural_offset: bool = True
) -> MultiResNeuralTextureModel:
    torch.manual_seed(4)
    config = ModelConfig(
        texture_res=16,
        min_res=8,
        texture_channels=2,
        off_texture_res=4,
        off_texture_channels=2,
        off_neurons=8,
        off_hidden_layers=1,
        n_neurons=8,
        n_hidden_layers=2,
        offset_mode=offset_mode,
        use_neural_offset=use_neural_offset,
        sh_degree=3,
    )
    model = MultiResNeuralTextureModel(config).cpu().eval()
    with torch.no_grad():
        if use_neural_offset:
            model.offset_gate.zero_()
    return model


def latent_layout_and_roundtrip() -> None:
    model = tiny_model()
    latent, layout = extract_latent_stack(model, height=12, width=20, device=torch.device("cpu"))
    assert latent.shape == (12, 20, 6)
    assert layout.appearance_channels == 4
    assert layout.offset_channels == 2
    assert layout.appearance_resolutions == (8, 16)

    light = (0.2, -0.1)
    view = (-0.15, 0.25)
    decoded = render_latent_stack(
        model, latent, layout, light, view, device=torch.device("cpu"), chunk_size=37
    )
    direct = render_model(
        model, 12, 20, light, view, device=torch.device("cpu"), chunk_size=41
    )
    assert decoded.rgb.shape == direct.shape == (12, 20, 3)
    assert decoded.offset.shape == (12, 20, 3)
    assert np.max(np.abs(decoded.rgb - direct)) < 2e-6


def offset_visualization_is_ungated() -> None:
    model = tiny_model()
    with torch.no_grad():
        output_layer = model.offset_mlp[-1]
        output_layer.weight.zero_()
        output_layer.bias.copy_(torch.atanh(torch.tensor([0.4, -0.4])))
        model.offset_gate.zero_()

    latent, layout = extract_latent_stack(model, height=7, width=9)
    decoded = render_latent_stack(
        model,
        latent,
        layout,
        light_xy=(0.1, 0.2),
        view_xy=(-0.2, 0.1),
        uv_scale=(2.0, 4.0),
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(decoded.offset[..., 0], 0.6, atol=1e-6)
    np.testing.assert_allclose(decoded.offset[..., 1], 0.45, atol=1e-6)
    np.testing.assert_allclose(decoded.offset[..., 2], 0.5, atol=1e-6)


def offset_variants_are_supported() -> None:
    for mode, use_offset, channels in (("1d", True, 6), ("2d", False, 4)):
        model = tiny_model(offset_mode=mode, use_neural_offset=use_offset)
        latent, layout = extract_latent_stack(model, height=11, width=13)
        expected_mode = mode if use_offset else "none"
        assert latent.shape == (11, 13, channels)
        assert layout.offset_mode == expected_mode
        decoded = render_latent_stack(
            model,
            latent,
            layout,
            light_xy=(0.1, 0.2),
            view_xy=(-0.2, 0.1),
            device=torch.device("cpu"),
            chunk_size=29,
        )
        assert np.isfinite(decoded.rgb).all()
        assert np.isfinite(decoded.offset).all()


def config_is_strict_and_immutable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        valid_path = os.path.join(directory, "valid.json")
        with open(valid_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "extract": {
                        "preview": {
                            "light_xy": [0.1, 0.2],
                            "view_xy": [-0.3, 0.1],
                        }
                    }
                },
                handle,
            )
        config = load_synthesis_config(valid_path)
        assert config.extract.preview.light_xy == (0.1, 0.2)
        assert config.extract.preview.view_xy == (-0.3, 0.1)

        invalid_path = os.path.join(directory, "invalid.json")
        with open(invalid_path, "w", encoding="utf-8") as handle:
            json.dump({"extract": {"widht": 64}}, handle)
        try:
            load_synthesis_config(invalid_path)
        except ValueError as error:
            assert "widht" in str(error)
        else:
            raise AssertionError("unknown config keys must be rejected")

        invalid_direction_path = os.path.join(directory, "invalid_direction.json")
        with open(invalid_direction_path, "w", encoding="utf-8") as handle:
            json.dump({"extract": {"preview": {"view_xy": [1.0, 1.0]}}}, handle)
        try:
            load_synthesis_config(invalid_direction_path)
        except ValueError as error:
            assert "view_xy" in str(error)
        else:
            raise AssertionError("preview directions must lie on the hemisphere")


def synthesis_paths_are_stable() -> None:
    paths = synthesis_stage_paths(Path("/tmp/material/synthesis"), "latent")
    assert paths.artifacts == Path("/tmp/material/synthesis/latent")
    assert paths.visualization == Path(
        "/tmp/material/synthesis/visualization/latent"
    )
    try:
        synthesis_stage_paths(Path("/tmp/material/synthesis"), "../latent")
    except ValueError:
        pass
    else:
        raise AssertionError("stage names must not escape the synthesis root")


def repaint_zooms_follow_diffusion_endpoints() -> None:
    config = SynthesisConfig(
        diffusion=DiffusionConfig(
            data=DiffusionDataConfig(zoom_range=(1.0, 6.0))
        ),
        repaint=RepaintConfig(
            process=RepaintProcessConfig(
                coarse_zoom=3.0,
                fine_zoom=2.0,
            )
        ),
    )
    synchronized = synchronize_repaint_zooms(config)
    assert config.repaint.process.coarse_zoom == 3.0
    assert config.repaint.process.fine_zoom == 2.0
    assert synchronized.repaint.process.coarse_zoom == 6.0
    assert synchronized.repaint.process.fine_zoom == 1.0


def output_root_uses_extraction_project() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_dir = root / "synthesis"
        config_dir.mkdir()
        project_path = config_dir / "project.json"
        project_path.touch()

        configured = resolve_output_root(
            SynthesisConfig(output_dir="."),
            config_path=str(project_path),
        )
        assert configured == config_dir

        explicit = resolve_output_root(
            SynthesisConfig(output_dir="."),
            config_path=str(project_path),
            override=str(root / "custom"),
        )
        assert explicit == root / "custom"

        try:
            resolve_output_root(SynthesisConfig(), config_path=str(project_path))
        except ValueError as error:
            assert "project.json" in str(error)
        else:
            raise AssertionError("later stages require a synthesis root")


if __name__ == "__main__":
    checks = [
        ("latent_layout_and_roundtrip", latent_layout_and_roundtrip),
        ("offset_visualization_is_ungated", offset_visualization_is_ungated),
        ("offset_variants_are_supported", offset_variants_are_supported),
        ("config_is_strict_and_immutable", config_is_strict_and_immutable),
        ("synthesis_paths_are_stable", synthesis_paths_are_stable),
        (
            "repaint_zooms_follow_diffusion_endpoints",
            repaint_zooms_follow_diffusion_endpoints,
        ),
        (
            "output_root_uses_extraction_project",
            output_root_uses_extraction_project,
        ),
    ]
    for check_name, check_function in checks:
        check(check_name, check_function)
    print(f"\n{len(checks)}/{len(checks)} checks passed")
