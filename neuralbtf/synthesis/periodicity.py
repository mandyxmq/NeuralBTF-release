"""Period estimation and period-aware Wang tile sizing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from .config import WangTileShapeConfig


@dataclass(frozen=True)
class PeriodAnalysis:
    height: int
    width: int
    period_x: Optional[int]
    period_y: Optional[int]
    autocorrelation: np.ndarray
    profile_x: np.ndarray
    profile_y: np.ndarray
    prominence_ratio: float
    minimum_peak_distance: int

    def summary(self) -> dict:
        return {
            "height": self.height,
            "width": self.width,
            "period_x": self.period_x,
            "period_y": self.period_y,
            "prominence_ratio": self.prominence_ratio,
            "minimum_peak_distance": self.minimum_peak_distance,
        }


def analyze_periodicity(
    latent: np.ndarray,
    *,
    prominence_ratio: float,
    minimum_peak_distance: int,
) -> PeriodAnalysis:
    if latent.ndim != 3:
        raise ValueError(f"expected an HWC latent, got {latent.shape}")
    height, width, _ = latent.shape
    magnitude = np.mean(
        np.abs(np.asarray(latent, dtype=np.float32)),
        axis=2,
        dtype=np.float32,
    )
    centered = magnitude - magnitude.mean(dtype=np.float32)
    spectrum = np.fft.fft2(centered)
    autocorrelation = np.fft.fftshift(
        np.fft.ifft2(np.abs(spectrum) ** 2).real
    ).astype(np.float32)
    profile_x = autocorrelation[height // 2].copy()
    profile_y = autocorrelation[:, width // 2].copy()

    def closest(profile: np.ndarray, center: int) -> Optional[int]:
        maximum = float(np.max(profile))
        if not np.isfinite(maximum) or maximum <= 0.0:
            return None
        peaks, _ = find_peaks(
            profile,
            prominence=maximum * float(prominence_ratio),
            distance=int(minimum_peak_distance),
        )
        peaks = peaks[peaks != center]
        if not peaks.size:
            return None
        distances = np.abs(peaks - center)
        return int(distances[np.argmin(distances)])

    return PeriodAnalysis(
        height=int(height),
        width=int(width),
        period_x=closest(profile_x, width // 2),
        period_y=closest(profile_y, height // 2),
        autocorrelation=autocorrelation,
        profile_x=profile_x,
        profile_y=profile_y,
        prominence_ratio=float(prominence_ratio),
        minimum_peak_distance=int(minimum_peak_distance),
    )


def _multiple_near_target(period: int, target: int, limit: int) -> Optional[int]:
    if period <= 0 or period > limit:
        return None
    multiplier = max(1, int(round(float(target) / float(period))))
    multiplier = min(multiplier, limit // period)
    return period * multiplier if multiplier > 0 else None


def _divisible(value: int, divisor: int, limit: int) -> int:
    value = max(1, min(int(value), int(limit)))
    value -= value % int(divisor)
    if value <= 0:
        value = int(divisor) if divisor <= limit else int(limit)
    if value % 2:
        value -= 1
    if value <= 0:
        raise ValueError("source latent is too small for an even Wang tile")
    return int(value)


def select_tile_shape(
    analysis: Optional[PeriodAnalysis],
    config: WangTileShapeConfig,
    *,
    source_height: int,
    source_width: int,
) -> dict:
    if config.mode == "manual":
        return {
            "height": _divisible(config.height, config.divisor, source_height),
            "width": _divisible(config.width, config.divisor, source_width),
            "strategy": "manual",
            "period_x": None,
            "period_y": None,
        }
    if analysis is None:
        raise ValueError("automatic tile sizing requires periodicity analysis")

    target = config.target_size
    if target is None:
        target = int(
            round(min(source_height, source_width) * config.target_size_ratio)
        )
    fallback_h = min(config.fallback_size, source_height)
    fallback_w = min(config.fallback_size, source_width)
    period_x = analysis.period_x
    period_y = analysis.period_y
    strategy = "fallback"
    raw_h = fallback_h
    raw_w = fallback_w

    if period_x and period_y:
        common = math.lcm(int(period_x), int(period_y))
        if common <= min(source_height, source_width):
            candidate = _multiple_near_target(
                common, target, min(source_height, source_width)
            )
            if candidate is not None:
                raw_h = candidate
                raw_w = candidate
                strategy = "period_lcm"
        elif not config.enforce_square:
            candidate_h = _multiple_near_target(
                int(period_y), target, source_height
            )
            candidate_w = _multiple_near_target(
                int(period_x), target, source_width
            )
            if candidate_h and candidate_w:
                raw_h = candidate_h
                raw_w = candidate_w
                strategy = "independent_periods"
    elif period_y:
        candidate = _multiple_near_target(
            int(period_y), target, min(source_height, source_width)
        )
        if candidate:
            raw_h = raw_w = candidate
            strategy = "vertical_period"
    elif period_x:
        candidate = _multiple_near_target(
            int(period_x), target, min(source_height, source_width)
        )
        if candidate:
            raw_h = raw_w = candidate
            strategy = "horizontal_period"

    too_coarse = (
        (period_y and period_y > min(target, source_height // 2))
        or (period_x and period_x > min(target, source_width // 2))
    )
    if too_coarse:
        raw_h = fallback_h
        raw_w = fallback_w
        strategy = "coarse_period_fallback"

    if config.enforce_square:
        raw_h = raw_w = min(raw_h, raw_w, source_height, source_width)
    return {
        "height": _divisible(raw_h, config.divisor, source_height),
        "width": _divisible(raw_w, config.divisor, source_width),
        "strategy": strategy,
        "target_size": int(target),
        "fallback_size": int(config.fallback_size),
        "period_x": period_x,
        "period_y": period_y,
    }
