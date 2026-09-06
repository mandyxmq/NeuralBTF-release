"""Training and validation losses.

Mode 0 (relative L2) is what all published results use; the others are kept
because they are referenced by the ablation configs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LOSS_MODES = {
    0: "relative L2 (prediction-normalised)",
    1: "L2 in log1p space",
    2: "Charbonnier in log1p space",
    3: "relative L2 + Charbonnier in log1p space",
    4: "smooth L1 (Huber)",
}


def _log1p_safe(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0) + eps)


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def btf_loss(pred: torch.Tensor, target: torch.Tensor, mode: int = 0,
             denominator: torch.Tensor | None = None) -> torch.Tensor:
    """Reconstruction loss between predicted and measured radiance.

    Args:
        pred: ``(N, 3)`` predicted radiance.
        target: ``(N, 3)`` measured radiance.
        mode: key of :data:`LOSS_MODES`.
        denominator: values used to normalise mode 0; defaults to ``pred.detach()``.
    """
    if mode == 0:
        den = pred.detach() if denominator is None else denominator
        return ((pred - target) ** 2 / (den**2 + 0.01)).mean()

    if mode == 1:
        o = torch.log1p(torch.clamp(pred, min=-0.1))
        t = torch.log1p(torch.clamp(target, min=-0.1))
        return ((o - t) ** 2).mean()

    if mode == 2:
        return _charbonnier(_log1p_safe(pred) - _log1p_safe(target)).mean()

    if mode == 3:
        den = pred.detach() if denominator is None else denominator
        rel = ((pred - target) ** 2 / (den**2 + 0.01)).mean()
        return rel + _charbonnier(_log1p_safe(pred) - _log1p_safe(target)).mean()

    if mode == 4:
        return F.smooth_l1_loss(pred, target, beta=1.0)

    raise ValueError(f"unknown loss mode {mode}; valid modes: {sorted(LOSS_MODES)}")


# Ground-truth lookups use align_corners=True, as in the reference implementation:
# uv 0 and 1 land on the *centres* of the first and last pixel. Note this differs
# from the model's own texture sampling (align_corners=False, uv spans texel edges),
# so the learned texture sits half a pixel off the capture grid. Kept as-is because
# the published checkpoints were trained this way; flip it to False for an exactly
# pixel-centred convention, but retrain if you do.
GT_ALIGN_CORNERS = True


def sample_images(images: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``images`` at ``uv``.

    Args:
        images: ``(B, H, W, C)``.
        uv: ``(B * P, 2)`` in ``[0, 1]``, image-major (matching
            :func:`neuralbtf.data.make_model_input`).

    Returns:
        ``(B * P, C)``.
    """
    b, _, _, c = images.shape
    grid = uv.view(b, 1, -1, 2) * 2.0 - 1.0          # (B, 1, P, 2) in [-1, 1]
    nchw = images.permute(0, 3, 1, 2)                 # (B, C, H, W)
    sampled = F.grid_sample(nchw, grid, mode="bilinear", padding_mode="border",
                            align_corners=GT_ALIGN_CORNERS)   # (B, C, 1, P)
    return sampled.squeeze(2).permute(0, 2, 1).reshape(-1, c)


def training_loss(pred: torch.Tensor, gt_images: torch.Tensor, uv: torch.Tensor,
                  mode: int = 0) -> torch.Tensor:
    """Loss against ground-truth images sampled at the same uv as ``pred``."""
    target = sample_images(gt_images, uv).to(pred.dtype)
    return btf_loss(pred, target, mode)
