from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualRefinement:
    """Result of a masked residual refinement pass."""

    residual: np.ndarray
    mask: np.ndarray
    output: np.ndarray


def compute_residual(target: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Return the RGB residual that the baseline does not explain."""
    target = _validate_rgb_image(target)
    baseline = _validate_rgb_image(baseline)
    if target.shape != baseline.shape:
        raise ValueError(f"Image shapes differ: {target.shape} != {baseline.shape}")
    return target - baseline


def compute_residual_mask(
    target: np.ndarray,
    baseline: np.ndarray,
    threshold: float = 0.08,
    softness: float = 0.08,
    blur_radius: int = 4,
    positive_luma_only: bool = False,
) -> np.ndarray:
    """Build a soft confidence mask for local residual corrections.

    The mask is high where the target differs enough from the LUT baseline to
    represent an effect the global LUT could not model. It does not identify
    semantic objects; it only gates residual transfer conservatively.
    """
    residual = compute_residual(target, baseline)
    if threshold < 0:
        raise ValueError("threshold must be >= 0.")
    if softness < 0:
        raise ValueError("softness must be >= 0.")
    if blur_radius < 0:
        raise ValueError("blur_radius must be >= 0.")

    color_delta = np.linalg.norm(residual, axis=-1) / np.sqrt(3.0)
    mask = _soft_threshold(color_delta, threshold, softness)

    if positive_luma_only:
        luma_delta = _luminance(target) - _luminance(baseline)
        mask *= _soft_threshold(luma_delta, 0.0, max(softness, 1e-6))

    if blur_radius > 0:
        mask = _box_blur(mask, blur_radius)

    return np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)


def apply_masked_residual(
    baseline: np.ndarray,
    residual: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    max_residual: float | None = 0.35,
) -> np.ndarray:
    """Apply a soft masked residual correction to a baseline image."""
    baseline = _validate_rgb_image(baseline)
    residual = np.asarray(residual, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if residual.shape != baseline.shape:
        raise ValueError(f"Residual shape differs: {residual.shape} != {baseline.shape}")
    if mask.shape != baseline.shape[:2]:
        raise ValueError(f"Mask shape differs: {mask.shape} != {baseline.shape[:2]}")
    if alpha < 0:
        raise ValueError("alpha must be >= 0.")
    if max_residual is not None:
        if max_residual < 0:
            raise ValueError("max_residual must be >= 0.")
        residual = np.clip(residual, -max_residual, max_residual)

    refined = baseline + float(alpha) * mask[..., None] * residual
    return np.clip(refined, 0.0, 1.0).astype(np.float32, copy=False)


def refine_with_masked_residual(
    target: np.ndarray,
    baseline: np.ndarray,
    alpha: float = 0.5,
    threshold: float = 0.08,
    softness: float = 0.08,
    blur_radius: int = 4,
    max_residual: float | None = 0.35,
    positive_luma_only: bool = False,
) -> ResidualRefinement:
    """Refine a LUT baseline with a masked local residual from the target."""
    residual = compute_residual(target, baseline)
    mask = compute_residual_mask(
        target,
        baseline,
        threshold=threshold,
        softness=softness,
        blur_radius=blur_radius,
        positive_luma_only=positive_luma_only,
    )
    output = apply_masked_residual(
        baseline,
        residual,
        mask,
        alpha=alpha,
        max_residual=max_residual,
    )
    return ResidualRefinement(residual=residual, mask=mask, output=output)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert a single-channel mask to RGB for saving and comparison sheets."""
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError("Expected a 2D mask.")
    mask = np.clip(mask, 0.0, 1.0)
    return np.repeat(mask[..., None], 3, axis=-1)


def _soft_threshold(values: np.ndarray, threshold: float, softness: float) -> np.ndarray:
    if softness == 0:
        return (values >= threshold).astype(np.float32)
    scaled = np.clip((values - threshold) / softness, 0.0, 1.0)
    return (scaled * scaled * (3.0 - 2.0 * scaled)).astype(np.float32)


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.astype(np.float32, copy=False)
    size = radius * 2 + 1
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return (total / float(size * size)).astype(np.float32)


def _luminance(image: np.ndarray) -> np.ndarray:
    image = _validate_rgb_image(image)
    return (
        image[..., 0] * 0.2126
        + image[..., 1] * 0.7152
        + image[..., 2] * 0.0722
    )


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape HxWx3.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("Expected a numeric RGB image array.")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("RGB image contains NaN or infinite values.")
    min_value = float(array.min(initial=0.0))
    max_value = float(array.max(initial=0.0))
    if min_value < 0.0:
        raise ValueError("RGB image values must be non-negative.")
    if max_value > 1.0:
        if max_value > 255.0:
            raise ValueError("RGB image values must be in [0, 1] or [0, 255].")
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)
