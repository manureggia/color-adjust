from __future__ import annotations

from typing import Literal

import numpy as np


GradeStyle = Literal[
    "warm-sunset",
    "cool-winter",
    "teal-orange",
    "vintage",
    "autumn",
    "high-contrast",
]


STYLES: tuple[GradeStyle, ...] = (
    "warm-sunset",
    "cool-winter",
    "teal-orange",
    "vintage",
    "autumn",
    "high-contrast",
)


def apply_color_grade(
    image: np.ndarray,
    style: GradeStyle,
    strength: float = 1.0,
) -> np.ndarray:
    """Create deterministic recoloring targets for testing LUT fitting."""
    if not (0.0 <= strength <= 1.0):
        raise ValueError("strength must be in [0, 1].")
    source = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape HxWx3.")

    if style == "warm-sunset":
        graded = _warm_sunset(source)
    elif style == "cool-winter":
        graded = _cool_winter(source)
    elif style == "teal-orange":
        graded = _teal_orange(source)
    elif style == "vintage":
        graded = _vintage(source)
    elif style == "autumn":
        graded = _autumn(source)
    elif style == "high-contrast":
        graded = _high_contrast(source)
    else:
        raise ValueError(f"Unknown grade style: {style}")

    return np.clip(source * (1.0 - strength) + graded * strength, 0.0, 1.0)


def _warm_sunset(image: np.ndarray) -> np.ndarray:
    result = _contrast(image, 1.08)
    result = _gamma(result, np.array([0.94, 0.98, 1.08], dtype=np.float32))
    result = result * np.array([1.13, 1.02, 0.86], dtype=np.float32)
    result += np.array([0.035, 0.012, -0.018], dtype=np.float32)
    return _lift_shadows(result, np.array([0.025, 0.008, -0.015], dtype=np.float32))


def _cool_winter(image: np.ndarray) -> np.ndarray:
    result = _contrast(image, 1.04)
    result = _gamma(result, np.array([1.06, 1.02, 0.94], dtype=np.float32))
    result = result * np.array([0.88, 0.98, 1.15], dtype=np.float32)
    result += np.array([-0.018, 0.0, 0.035], dtype=np.float32)
    return _lift_shadows(result, np.array([-0.01, 0.004, 0.025], dtype=np.float32))


def _teal_orange(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)[..., None]
    shadows = 1.0 - smoothstep(0.25, 0.70, luminance)
    highlights = smoothstep(0.35, 0.85, luminance)
    result = _contrast(image, 1.12)
    result += shadows * np.array([-0.035, 0.045, 0.055], dtype=np.float32)
    result += highlights * np.array([0.070, 0.025, -0.045], dtype=np.float32)
    return result


def _vintage(image: np.ndarray) -> np.ndarray:
    sepia_matrix = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        dtype=np.float32,
    )
    sepia = image @ sepia_matrix.T
    result = image * 0.45 + sepia * 0.55
    result = _contrast(result, 0.88)
    return result + np.array([0.025, 0.015, -0.010], dtype=np.float32)


def _autumn(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)[..., None]
    mids = smoothstep(0.18, 0.75, luminance) * (1.0 - smoothstep(0.70, 0.98, luminance))
    result = _contrast(image, 1.05)
    result = result * np.array([1.12, 0.94, 0.78], dtype=np.float32)
    result += mids * np.array([0.060, 0.018, -0.055], dtype=np.float32)
    return result


def _high_contrast(image: np.ndarray) -> np.ndarray:
    result = _contrast(image, 1.22)
    result = _gamma(result, np.array([0.98, 1.0, 1.03], dtype=np.float32))
    return result


def _luminance(image: np.ndarray) -> np.ndarray:
    return (
        image[..., 0] * 0.2126
        + image[..., 1] * 0.7152
        + image[..., 2] * 0.0722
    )


def _contrast(image: np.ndarray, amount: float) -> np.ndarray:
    return np.clip((image - 0.5) * amount + 0.5, 0.0, 1.0)


def _gamma(image: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0) ** gamma


def _lift_shadows(image: np.ndarray, color: np.ndarray) -> np.ndarray:
    luminance = _luminance(np.clip(image, 0.0, 1.0))[..., None]
    shadows = 1.0 - smoothstep(0.18, 0.65, luminance)
    return image + shadows * color


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    x = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

