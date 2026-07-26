"""Neural BTF model: multi-resolution neural texture + neural offset + MLP decoder.

Parameter names match the reference implementation (``util.py``:
``MultiResNeuralTextureModel`` / ``MultiResNeuralTextureModel_1d``), so existing
checkpoints load with ``strict=True`` and vice versa.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encodings import SphericalHarmonicsEncoding


class PyTorchMLP(nn.Module):
    """Fully connected ReLU decoder (the ``FullyFusedMLP`` stand-in).

    ``n_hidden_layers`` counts the input layer, matching tiny-cuda-nn's convention.
    """

    def __init__(self, n_input_dims: int, n_output_dims: int, n_neurons: int = 64,
                 n_hidden_layers: int = 2):
        super().__init__()

        layers = [nn.Linear(n_input_dims, n_neurons), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.ReLU()]
        layers.append(nn.Linear(n_neurons, n_output_dims))

        self.model = nn.Sequential(*layers)
        self.model.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class ModelConfig:
    """Everything needed to rebuild the model for evaluation/rendering."""

    # Texture pyramid
    texture_res: int = 512          # finest level
    min_res: int = 128              # coarsest level
    texture_channels: int = 8       # features per level

    # Neural offset
    use_neural_offset: bool = True
    offset_mode: str = "2d"         # "2d": free uv shift, "1d": shift along the view ray
    off_texture_res: int = 32
    off_texture_channels: int = 8
    off_neurons: int = 512
    off_hidden_layers: int = 1

    # Decoder / directional encoding
    n_neurons: int = 512
    n_hidden_layers: int = 3
    sh_degree: int = 3
    n_out: int = 3

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_state_dict(cls, state: dict) -> "ModelConfig":
        """Recover the architecture from a checkpoint's tensor shapes.

        Lets a checkpoint be evaluated without its ``_model_config.json`` -- useful
        for the models trained before this release wrote one.
        """
        resolutions = sorted(state[k].shape[-1] for k in state if k.startswith("textures."))
        if not resolutions:
            raise ValueError("checkpoint has no 'textures.*' entries; is it a BTF model?")
        channels = state["textures.0"].shape[1]

        # PyTorchMLP: n_hidden_layers + 1 Linear layers, hence one weight each.
        mlp_weights = sorted(k for k in state if k.startswith("mlp.model.") and k.endswith(".weight"))
        n_neurons = state[mlp_weights[0]].shape[0]
        n_hidden_layers = len(mlp_weights) - 1

        # The decoder sees the texture features plus degree^2 coefficients per direction.
        n_dir_feats = state[mlp_weights[0]].shape[1] - len(resolutions) * channels
        sh_degree = int(round((n_dir_feats / 2) ** 0.5))
        if sh_degree**2 * 2 != n_dir_feats:
            raise ValueError(f"cannot infer the SH degree from {n_dir_feats} directional inputs")

        use_offset = "offset_texture" in state
        cfg = cls(
            texture_res=resolutions[-1],
            min_res=resolutions[0],
            texture_channels=channels,
            use_neural_offset=use_offset,
            n_neurons=n_neurons,
            n_hidden_layers=n_hidden_layers,
            sh_degree=sh_degree,
            n_out=state[mlp_weights[-1]].shape[0],
        )
        if not use_offset:
            return cfg

        off_weights = sorted(k for k in state
                             if k.startswith("offset_mlp.") and k.endswith(".weight"))
        return replace(
            cfg,
            offset_mode="1d" if state[off_weights[-1]].shape[0] == 1 else "2d",
            off_texture_res=state["offset_texture"].shape[-1],
            off_texture_channels=state["offset_texture"].shape[1],
            off_neurons=state[off_weights[0]].shape[0],
            off_hidden_layers=len(off_weights) - 1,
        )


class MultiResNeuralTextureModel(nn.Module):
    """Maps ``[u, v, light_x, light_y, view_x, view_y]`` to RGB.

    1. a small "offset" texture + MLP warps the uv lookup by a view-dependent
       amount (NeuMIP-style parallax), gated by a learned scalar that starts at 0;
    2. a pyramid of neural textures is sampled bilinearly at the warped uv;
    3. the concatenated features and the SH-encoded directions are decoded by an MLP.

    ``offset_mode``:
        ``"2d"`` -- the offset MLP predicts a free 2-D uv shift.
        ``"1d"`` -- the offset MLP predicts a scalar displacement that is applied
        along the projected view direction ``view_xy / max(view_z, 0.6)``.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.offset_mode not in ("1d", "2d"):
            raise ValueError(f"offset_mode must be '1d' or '2d', got {cfg.offset_mode!r}")

        self.channels_per_level = cfg.texture_channels
        self.use_neural_offset = cfg.use_neural_offset
        self.offset_mode = cfg.offset_mode

        # --- 1. Neural offset ------------------------------------------------
        if self.use_neural_offset:
            self.off_channels = cfg.off_texture_channels
            # Zero init => identity mapping at step 0, no warping artifacts.
            self.offset_texture = nn.Parameter(
                torch.zeros(1, cfg.off_texture_channels, cfg.off_texture_res, cfg.off_texture_res)
            )

            n_offset_out = 1 if cfg.offset_mode == "1d" else 2
            layers = [nn.Linear(cfg.off_texture_channels + 3, cfg.off_neurons), nn.ReLU()]
            for _ in range(cfg.off_hidden_layers - 1):
                layers += [nn.Linear(cfg.off_neurons, cfg.off_neurons), nn.ReLU()]
            layers.append(nn.Linear(cfg.off_neurons, n_offset_out))
            self.offset_mlp = nn.Sequential(*layers)

            # Gate starts closed, so training begins from the un-warped model.
            self.offset_gate = nn.Parameter(torch.tensor(0.0))

        # --- 2. Multi-resolution texture pyramid -----------------------------
        self.resolutions = []
        res = cfg.min_res
        while res <= cfg.texture_res:
            self.resolutions.append(res)
            res *= 2
        if not self.resolutions:
            raise ValueError(f"min_res ({cfg.min_res}) exceeds texture_res ({cfg.texture_res})")

        self.textures = nn.ParameterList([
            nn.Parameter(torch.empty(1, self.channels_per_level, r, r).uniform_(-1e-4, 1e-4))
            for r in self.resolutions
        ])

        # --- 3. Directional encoding -----------------------------------------
        self.dir_encoder = SphericalHarmonicsEncoding(n_dirs=2, degree=cfg.sh_degree)

        # --- 4. Decoder -------------------------------------------------------
        mlp_input_dim = len(self.resolutions) * self.channels_per_level + self.dir_encoder.n_output_dims
        self.mlp = PyTorchMLP(
            n_input_dims=mlp_input_dim,
            n_output_dims=cfg.n_out,
            n_neurons=cfg.n_neurons,
            n_hidden_layers=cfg.n_hidden_layers,
        )

    @staticmethod
    def _reconstruct_z(xy: torch.Tensor) -> torch.Tensor:
        """Recover z of a unit direction from its xy projection."""
        sq = torch.clamp(1.0 - (xy * xy).sum(-1), min=0.0)
        return torch.sqrt(sq)

    def _sample(self, texture: torch.Tensor, grid: torch.Tensor, channels: int) -> torch.Tensor:
        sampled = F.grid_sample(texture, grid, align_corners=False, padding_mode="border")
        return sampled.view(channels, -1).permute(1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(N, 6)`` = ``[u, v, light_x, light_y, view_x, view_y]`` -> ``(N, 3)``."""
        uv = x[:, :2]
        light_xy = x[:, 2:4]
        view_xy = x[:, 4:6]
        view_z = self._reconstruct_z(view_xy)

        # --- Neural offset ---------------------------------------------------
        if self.use_neural_offset:
            view_dir_3d = torch.cat([view_xy, view_z.unsqueeze(-1)], dim=-1)
            grid_off = uv.view(1, -1, 1, 2) * 2.0 - 1.0
            off_feats = self._sample(self.offset_texture, grid_off, self.off_channels)

            offset = torch.tanh(self.offset_mlp(torch.cat([off_feats, view_dir_3d], dim=-1)))
            if self.offset_mode == "1d":
                # Displace along the view ray projected onto the surface; the z clamp
                # keeps grazing angles from blowing the displacement up.
                shift = view_xy / torch.clamp(view_z, min=0.6).unsqueeze(-1)
                offset = shift * offset

            uv_shifted = uv + self.offset_gate * offset
            uv_shifted = uv_shifted - torch.floor(uv_shifted)  # wrap into [0, 1)
        else:
            uv_shifted = uv

        # --- Texture pyramid --------------------------------------------------
        grid = uv_shifted.view(1, -1, 1, 2) * 2.0 - 1.0
        spatial_feats = torch.cat(
            [self._sample(tex, grid, self.channels_per_level) for tex in self.textures], dim=-1
        )

        # --- Directional features + decode ------------------------------------
        light_z = self._reconstruct_z(light_xy)
        dirs = torch.cat([
            light_xy, light_z.unsqueeze(-1),
            view_xy, view_z.unsqueeze(-1),
        ], dim=-1)
        dir_feats = self.dir_encoder(dirs)

        out = self.mlp(torch.cat([spatial_feats, dir_feats], dim=-1))
        return out.float()

    def parameter_groups(self, lr: float, texture_lr_scale: float = 10.0) -> list[dict]:
        """Textures train faster than the decoder; same split as the reference code."""
        texture_params, mlp_params = [], []
        for name, param in self.named_parameters():
            (texture_params if "texture" in name else mlp_params).append(param)
        return [
            {"params": texture_params, "lr": lr * texture_lr_scale},
            {"params": mlp_params, "lr": lr},
        ]
