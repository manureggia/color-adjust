from __future__ import annotations

import math

import numpy as np


def _as_float_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError("Expected an image with shape HxW or HxWxC.")
    return np.clip(array, 0.0, 1.0)


def psnr(reference: np.ndarray, candidate: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio for images in [0, 1]."""
    ref = _as_float_image(reference)
    cand = _as_float_image(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"Image shapes differ: {ref.shape} != {cand.shape}")
    mse = float(np.mean((ref - cand) ** 2))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _uniform_filter(image: np.ndarray, window_size: int) -> np.ndarray:
    pad = window_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0), (0, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    total = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )
    return total / float(window_size * window_size)


def ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
    data_range: float = 1.0,
    window_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural similarity index with a local box window.

    This implementation avoids optional dependencies and returns the mean SSIM
    over all pixels and channels.
    """
    ref = _as_float_image(reference)
    cand = _as_float_image(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"Image shapes differ: {ref.shape} != {cand.shape}")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
    if min(ref.shape[:2]) < window_size:
        window_size = min(ref.shape[:2])
        if window_size % 2 == 0:
            window_size -= 1
        if window_size < 3:
            return _global_ssim(ref, cand, data_range, k1, k2)

    mu_x = _uniform_filter(ref, window_size)
    mu_y = _uniform_filter(cand, window_size)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = np.maximum(_uniform_filter(ref * ref, window_size) - mu_x2, 0.0)
    sigma_y2 = np.maximum(_uniform_filter(cand * cand, window_size) - mu_y2, 0.0)
    sigma_xy = _uniform_filter(ref * cand, window_size) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def _global_ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
    data_range: float,
    k1: float,
    k2: float,
) -> float:
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    scores: list[float] = []
    for channel in range(reference.shape[-1]):
        x = reference[..., channel]
        y = candidate[..., channel]
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        var_x = float(np.var(x))
        var_y = float(np.var(y))
        cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        scores.append(numerator / max(denominator, 1e-12))
    return float(np.mean(scores))

