"""Model arguments, shared by the trainer and the evaluator.

Training writes ``<prefix>_model_config.json`` next to its checkpoints; evaluation
reads it back, so a run is reproducible from the checkpoint alone.  Checkpoints
that predate that file (or that came from the original research scripts) still
load: the architecture is then recovered from the tensor shapes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from .models import ModelConfig, MultiResNeuralTextureModel

CONFIG_SUFFIX = "_model_config.json"


def add_model_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add the ``model`` argument group; defaults are the published configuration."""
    model = parser.add_argument_group("model")
    model.add_argument("--config", default="", help="JSON file with a 'network' section")
    model.add_argument("--texture_res", type=int, default=512, help="finest texture resolution")
    model.add_argument("--min_res", type=int, default=128, help="coarsest texture resolution")
    model.add_argument("--texture_channels", type=int, default=8,
                       help="features per texture level")
    model.add_argument("--n_neurons", type=int, default=512, help="decoder width")
    model.add_argument("--n_hidden_layers", type=int, default=3, help="decoder depth")
    model.add_argument("--sh_degree", type=int, default=3, help="spherical harmonics bands")
    model.add_argument("--offset_mode", default="2d", choices=["2d", "1d"],
                       help="2d: free uv offset; 1d: displacement along the view ray")
    model.add_argument("--no_offset", action="store_true", help="disable the neural offset")
    model.add_argument("--off_texture_res", type=int, default=32,
                       help="offset texture resolution")
    model.add_argument("--off_texture_channels", type=int, default=8,
                       help="offset texture features")
    model.add_argument("--off_neurons", type=int, default=512, help="offset MLP width")
    model.add_argument("--off_hidden", type=int, default=1, help="offset MLP depth")
    return model


def model_config_from_args(args) -> ModelConfig:
    """Build a :class:`ModelConfig` from parsed arguments.

    Only ``network.n_neurons``, ``network.n_hidden_layers`` and the SH degree are
    read from ``--config``; the rest of the original tiny-cuda-nn JSON described
    encodings this model does not use.
    """
    n_neurons, n_hidden_layers, sh_degree = args.n_neurons, args.n_hidden_layers, args.sh_degree

    if args.config:
        with open(args.config) as fh:
            cfg = json.load(fh)
        network = cfg.get("network", {})
        n_neurons = network.get("n_neurons", n_neurons)
        n_hidden_layers = network.get("n_hidden_layers", n_hidden_layers)
        for nested in cfg.get("encoding", {}).get("nested", []):
            if nested.get("otype") == "SphericalHarmonics":
                sh_degree = nested.get("degree", sh_degree)
                break

    return ModelConfig(
        texture_res=args.texture_res,
        min_res=args.min_res,
        texture_channels=args.texture_channels,
        use_neural_offset=not args.no_offset,
        offset_mode=args.offset_mode,
        off_texture_res=args.off_texture_res,
        off_texture_channels=args.off_texture_channels,
        off_neurons=args.off_neurons,
        off_hidden_layers=args.off_hidden,
        n_neurons=n_neurons,
        n_hidden_layers=n_hidden_layers,
        sh_degree=sh_degree,
    )


def find_model_config(checkpoint: str) -> str | None:
    """The ``*_model_config.json`` written next to ``checkpoint``, if there is one."""
    directory = os.path.dirname(os.path.abspath(checkpoint))
    matches = sorted(glob.glob(os.path.join(directory, f"*{CONFIG_SUFFIX}")))
    return matches[0] if matches else None


def load_model_config(path: str) -> ModelConfig:
    with open(path) as fh:
        return ModelConfig.from_dict(json.load(fh))


def drop_empty_placeholders(state: dict, model: torch.nn.Module) -> dict:
    """Remove zero-sized entries the model does not have.

    Checkpoints trained against tiny-cuda-nn carry a ``dir_encoder.params`` tensor
    of size 0: its ``Encoding`` module registers a parameter even when, as for
    spherical harmonics, the encoding has nothing to learn.  Dropping it lets those
    checkpoints load here, and everything that does hold weights is still matched
    strictly.
    """
    extra = [k for k in state if k not in model.state_dict()]
    empty = [k for k in extra if state[k].numel() == 0]
    if not empty:
        return state
    print(f"[model] ignoring empty placeholder tensor(s) from tiny-cuda-nn: {', '.join(empty)}")
    return {k: v for k, v in state.items() if k not in empty}


def load_model(checkpoint: str, device: str | torch.device = "cuda",
               config_path: str | None = None) -> tuple[MultiResNeuralTextureModel, ModelConfig]:
    """Load a checkpoint into an evaluation-ready model.

    ``config_path`` overrides the search for a sibling ``*_model_config.json``;
    without either, the architecture is inferred from the checkpoint itself.  The
    state dict is always loaded with ``strict=True``.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    path = config_path or find_model_config(checkpoint)
    if path:
        cfg = load_model_config(path)
        print(f"[model] {os.path.basename(path)}")
    else:
        cfg = ModelConfig.from_state_dict(state)
        print("[model] no *_model_config.json found; architecture inferred from the checkpoint")

    model = MultiResNeuralTextureModel(cfg).to(device)
    model.load_state_dict(drop_empty_placeholders(state, model), strict=True)
    model.eval()
    print(f"[model] {checkpoint}: {cfg.offset_mode} offset, texture pyramid "
          f"{model.resolutions} x {cfg.texture_channels}ch, decoder "
          f"{cfg.n_neurons}x{cfg.n_hidden_layers}")
    return model, cfg
