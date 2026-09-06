"""Directional encoding.

The reference implementation encoded light/view directions with tiny-cuda-nn's
``SphericalHarmonics`` encoding.  This module reimplements that encoding in plain
PyTorch so the release has no CUDA-extension dependency.  The polynomials and the
input convention are the ones tiny-cuda-nn uses, so weights trained with either
implementation are interchangeable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Real spherical harmonics, degree <= 4.  Same closed forms (and the same sign
# convention) as tiny-cuda-nn's spherical_harmonics.h.
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.48860251190291987
_SH_C2 = (1.0925484305920792, 0.94617469575755997, 0.31539156525251999, 0.54627421529603959)
_SH_C3 = (0.59004358992664352, 2.8906114426405538, 0.45704579946446572, 0.3731763325901154,
          1.4453057213202769)


def spherical_harmonics(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate the first ``degree`` SH bands for ``dirs``.

    Args:
        dirs: ``(..., 3)`` direction vectors, expected to be unit length.
        degree: number of bands (``degree**2`` output coefficients).

    Returns:
        ``(..., degree**2)`` coefficients.
    """
    if not 1 <= degree <= 4:
        raise ValueError(f"degree must be in [1, 4], got {degree}")

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    out = [torch.full_like(x, _SH_C0)]

    if degree >= 2:
        out += [-_SH_C1 * y, _SH_C1 * z, -_SH_C1 * x]

    if degree >= 3:
        xy, yz, xz = x * y, y * z, x * z
        x2, y2, z2 = x * x, y * y, z * z
        out += [
            _SH_C2[0] * xy,
            -_SH_C2[0] * yz,
            _SH_C2[1] * z2 - _SH_C2[2],
            -_SH_C2[0] * xz,
            _SH_C2[3] * (x2 - y2),
        ]

    if degree >= 4:
        out += [
            _SH_C3[0] * y * (y2 - 3.0 * x2),
            _SH_C3[1] * xy * z,
            _SH_C3[2] * y * (1.0 - 5.0 * z2),
            _SH_C3[3] * z * (5.0 * z2 - 3.0),
            _SH_C3[2] * x * (1.0 - 5.0 * z2),
            _SH_C3[4] * z * (x2 - y2),
            _SH_C3[0] * x * (3.0 * y2 - x2),
        ]

    return torch.stack(out, dim=-1)


class SphericalHarmonicsEncoding(nn.Module):
    """SH encoding of one or more direction vectors, tiny-cuda-nn compatible.

    tiny-cuda-nn's SH encoding documents its input as a direction remapped to
    ``[0, 1]`` and undoes that remapping internally (``d = 2 * input - 1``).  The
    reference training code fed *raw* direction vectors to it, so the network was
    trained on SH evaluated at ``2 * d - 1`` rather than at ``d``.  ``rescale_input``
    (on by default) reproduces that exactly; it matters only for checkpoint
    compatibility, since the encoding is just a fixed smooth feature map either way.
    """

    def __init__(self, n_dirs: int = 2, degree: int = 3, rescale_input: bool = True):
        super().__init__()
        self.n_dirs = n_dirs
        self.degree = degree
        self.rescale_input = rescale_input

    @property
    def n_input_dims(self) -> int:
        return 3 * self.n_dirs

    @property
    def n_output_dims(self) -> int:
        return self.degree**2 * self.n_dirs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(N, 3 * n_dirs)`` -> ``(N, degree**2 * n_dirs)``."""
        if x.shape[-1] != self.n_input_dims:
            raise ValueError(f"expected {self.n_input_dims} input dims, got {x.shape[-1]}")

        dirs = x.reshape(*x.shape[:-1], self.n_dirs, 3)
        if self.rescale_input:
            dirs = dirs * 2.0 - 1.0
        feats = spherical_harmonics(dirs, self.degree)
        return feats.reshape(*x.shape[:-1], self.n_output_dims)

    def extra_repr(self) -> str:
        return (f"n_dirs={self.n_dirs}, degree={self.degree}, "
                f"rescale_input={self.rescale_input}, n_output_dims={self.n_output_dims}")
