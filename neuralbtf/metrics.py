"""Image metrics for BTF reconstruction.

Every metric takes linear-radiance images, ``(H, W, 3)`` or ``(N, H, W, 3)``, and
returns a scalar tensor.  Negative values are clamped away first (a prediction can
undershoot, and the measured data has a little sensor noise below zero), which is
what the reference evaluation scripts did.

``COLUMNS`` fixes the order metrics are reported in.
"""

from __future__ import annotations

import torch

COLUMNS = ("psnr", "log_psnr", "rel_l2", "lpips")


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB, with a peak of 1.0 (``-10 log10 MSE``)."""
    mse = ((pred - gt) ** 2).mean()
    return -10.0 * torch.log10(mse)


def log_psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """PSNR after ``log1p``, which weights the dark end of the range like the eye does."""
    return psnr(torch.log1p(pred), torch.log1p(gt))


def relative_l2(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """``mean((pred - gt)^2 / (pred^2 + 0.01))`` -- the training loss, mode 0."""
    return ((pred - gt) ** 2 / (pred**2 + 0.01)).mean()


def tonemap(hdr: torch.Tensor, exposure_stops: float = 0.0, gamma: float = 2.2) -> torch.Tensor:
    """Reinhard tone map + gamma, mapping linear radiance into ``[0, 1]``."""
    x = torch.clamp(hdr, min=0.0) * (2.0**exposure_stops)
    return torch.clamp((x / (1.0 + x)) ** (1.0 / gamma), 0.0, 1.0)


class LPIPS:
    """Perceptual distance of tone-mapped images (lower is better).

    Needs the optional ``lpips`` package (``pip install lpips``); it downloads a
    small AlexNet on first use.  The release does not depend on it -- pass
    ``--lpips`` to the evaluator only if you want this column.
    """

    def __init__(self, net: str = "alex", device: str | torch.device = "cuda"):
        try:
            import lpips as _lpips
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "the --lpips column needs the optional 'lpips' package: pip install lpips"
            ) from exc

        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):  # the package is chatty on load
            self.fn = _lpips.LPIPS(net=net).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor,
                 exposure_stops: float = 0.0) -> torch.Tensor:
        """``pred``/``gt``: ``(H, W, 3)`` linear radiance."""
        def prepare(x: torch.Tensor) -> torch.Tensor:
            x = tonemap(x.to(self.device), exposure_stops) * 2.0 - 1.0  # LPIPS wants [-1, 1]
            return x.permute(2, 0, 1).unsqueeze(0)                      # (1, 3, H, W)

        return self.fn(prepare(pred), prepare(gt)).mean()


def image_metrics(pred: torch.Tensor, gt: torch.Tensor,
                  lpips: LPIPS | None = None) -> dict[str, float]:
    """All metrics for one image pair, keyed by :data:`COLUMNS`."""
    pred = torch.clamp(pred, min=0.0)
    gt = torch.clamp(gt, min=0.0)
    out = {
        "psnr": psnr(pred, gt).item(),
        "log_psnr": log_psnr(pred, gt).item(),
        "rel_l2": relative_l2(pred, gt).item(),
    }
    if lpips is not None:
        out["lpips"] = lpips(pred, gt).item()
    return out
