from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


FitMethod = Literal["weighted", "nearest-mean", "nearest-median"]


@dataclass(frozen=True)
class LUT3D:
    """RGB 3D LUT with trilinear interpolation."""

    table: np.ndarray

    def __post_init__(self) -> None:
        table = np.asarray(self.table, dtype=np.float32)
        if table.ndim != 4 or table.shape[-1] != 3:
            raise ValueError("LUT table must have shape NxNxNx3.")
        if len(set(table.shape[:3])) != 1:
            raise ValueError("LUT grid must be cubic: NxNxN.")
        object.__setattr__(self, "table", np.clip(table, 0.0, 1.0))

    @property
    def size(self) -> int:
        return int(self.table.shape[0])

    @classmethod
    def identity(cls, size: int = 17) -> "LUT3D":
        if size < 2:
            raise ValueError("LUT size must be >= 2.")
        axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
        r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
        return cls(np.stack([r, g, b], axis=-1))

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the LUT to an RGB image in [0, 1]."""
        image = _validate_rgb_image(image)
        flat = image.reshape(-1, 3)
        mapped = _trilinear_lookup(self.table, flat)
        return mapped.reshape(image.shape)

    def save_cube(self, path: str | Path, title: str = "color-adjust") -> None:
        """Save the LUT in common .cube text format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f'TITLE "{title}"\n')
            handle.write(f"LUT_3D_SIZE {self.size}\n")
            handle.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            handle.write("DOMAIN_MAX 1.0 1.0 1.0\n")
            for r in range(self.size):
                for g in range(self.size):
                    for b in range(self.size):
                        value = self.table[r, g, b]
                        handle.write(
                            f"{value[0]:.8f} {value[1]:.8f} {value[2]:.8f}\n"
                        )

    @classmethod
    def load_cube(cls, path: str | Path) -> "LUT3D":
        """Load a .cube LUT saved by this project."""
        size: int | None = None
        values: list[list[float]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                upper = stripped.upper()
                if upper.startswith("TITLE") or upper.startswith("DOMAIN_"):
                    continue
                if upper.startswith("LUT_3D_SIZE"):
                    size = int(stripped.split()[1])
                    continue
                parts = stripped.split()
                if len(parts) == 3:
                    values.append([float(part) for part in parts])
        if size is None:
            raise ValueError(f"Missing LUT_3D_SIZE in {path}.")
        expected = size * size * size
        if len(values) != expected:
            raise ValueError(f"Expected {expected} LUT rows, found {len(values)}.")
        table = np.asarray(values, dtype=np.float32).reshape(size, size, size, 3)
        return cls(table)


def fit_lut(
    source: np.ndarray,
    target: np.ndarray,
    size: int = 17,
    max_samples: int | None = 200_000,
    seed: int | None = 0,
    method: FitMethod = "weighted",
    smooth_iterations: int = 24,
    identity_weight: float = 0.02,
    luma_tolerance: float | None = None,
) -> LUT3D:
    """Fit a 3D LUT from corresponding source and target RGB images."""
    source = _validate_rgb_image(source)
    target = _validate_rgb_image(target)
    if source.shape != target.shape:
        raise ValueError(f"Image shapes differ: {source.shape} != {target.shape}")
    if size < 2:
        raise ValueError("LUT size must be >= 2.")
    if identity_weight < 0:
        raise ValueError("identity_weight must be >= 0.")

    src, tgt = _sample_pairs(
        source,
        target,
        max_samples=max_samples,
        seed=seed,
        luma_tolerance=luma_tolerance,
    )
    if method == "weighted":
        table, weights = _fit_weighted(src, tgt, size, identity_weight)
    elif method in {"nearest-mean", "nearest-median"}:
        table, weights = _fit_nearest(src, tgt, size, method, identity_weight)
    else:
        raise ValueError(f"Unknown fit method: {method}")

    if smooth_iterations > 0:
        table = _smooth_missing_and_weak_regions(
            table,
            weights,
            iterations=smooth_iterations,
            identity_weight=identity_weight,
        )
    return LUT3D(table)


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape HxWx3.")
    return np.clip(array, 0.0, 1.0)


def _sample_pairs(
    source: np.ndarray,
    target: np.ndarray,
    max_samples: int | None,
    seed: int | None,
    luma_tolerance: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    src = source.reshape(-1, 3)
    tgt = target.reshape(-1, 3)
    if luma_tolerance is not None:
        if luma_tolerance < 0:
            raise ValueError("luma_tolerance must be >= 0.")
        luma_diff = np.abs(_luminance(source).reshape(-1) - _luminance(target).reshape(-1))
        keep = luma_diff <= luma_tolerance
        kept = int(np.count_nonzero(keep))
        if kept == 0:
            raise ValueError("luma_tolerance rejected all samples.")
        src = src[keep]
        tgt = tgt[keep]
    if max_samples is None or max_samples <= 0 or max_samples >= len(src):
        return src, tgt
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(src), size=max_samples, replace=False)
    return src[indices], tgt[indices]


def _luminance(image: np.ndarray) -> np.ndarray:
    return (
        image[..., 0] * 0.2126
        + image[..., 1] * 0.7152
        + image[..., 2] * 0.0722
    )


def _fit_weighted(
    source: np.ndarray,
    target: np.ndarray,
    size: int,
    identity_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    identity = LUT3D.identity(size).table
    accum = identity * identity_weight
    weights = np.full((size, size, size), identity_weight, dtype=np.float32)

    position = np.clip(source, 0.0, 1.0) * (size - 1)
    low = np.floor(position).astype(np.int32)
    high = np.minimum(low + 1, size - 1)
    frac = position - low

    for dr in (0, 1):
        wr = frac[:, 0] if dr else 1.0 - frac[:, 0]
        ir = high[:, 0] if dr else low[:, 0]
        for dg in (0, 1):
            wg = frac[:, 1] if dg else 1.0 - frac[:, 1]
            ig = high[:, 1] if dg else low[:, 1]
            for db in (0, 1):
                wb = frac[:, 2] if db else 1.0 - frac[:, 2]
                ib = high[:, 2] if db else low[:, 2]
                sample_weight = (wr * wg * wb).astype(np.float32)
                np.add.at(weights, (ir, ig, ib), sample_weight)
                for channel in range(3):
                    np.add.at(
                        accum[..., channel],
                        (ir, ig, ib),
                        target[:, channel] * sample_weight,
                    )

    table = accum / np.maximum(weights[..., None], 1e-12)
    return np.clip(table, 0.0, 1.0), weights


def _fit_nearest(
    source: np.ndarray,
    target: np.ndarray,
    size: int,
    method: FitMethod,
    identity_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    identity = LUT3D.identity(size).table
    table = identity.copy()
    weights = np.full((size, size, size), identity_weight, dtype=np.float32)
    index = np.rint(np.clip(source, 0.0, 1.0) * (size - 1)).astype(np.int32)
    linear = np.ravel_multi_index((index[:, 0], index[:, 1], index[:, 2]), weights.shape)

    if method == "nearest-mean":
        accum = identity * identity_weight
        for channel in range(3):
            np.add.at(accum[..., channel], (index[:, 0], index[:, 1], index[:, 2]), target[:, channel])
        np.add.at(weights, (index[:, 0], index[:, 1], index[:, 2]), 1.0)
        table = accum / np.maximum(weights[..., None], 1e-12)
    else:
        for cell in np.unique(linear):
            mask = linear == cell
            r, g, b = np.unravel_index(cell, weights.shape)
            table[r, g, b] = np.median(target[mask], axis=0)
            weights[r, g, b] = float(np.count_nonzero(mask)) + identity_weight

    return np.clip(table, 0.0, 1.0), weights


def _smooth_missing_and_weak_regions(
    table: np.ndarray,
    weights: np.ndarray,
    iterations: int,
    identity_weight: float,
) -> np.ndarray:
    fixed = table.copy()
    confidence = weights / (weights + max(1.0, identity_weight))
    confidence = np.clip(confidence, 0.0, 1.0)[..., None]
    result = table.copy()
    for _ in range(iterations):
        average = _neighbor_average(result)
        result = confidence * fixed + (1.0 - confidence) * average
        result = np.clip(result, 0.0, 1.0)
    return result


def _neighbor_average(table: np.ndarray) -> np.ndarray:
    padded = np.pad(table, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="edge")
    return (
        padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
    ) / 6.0


def _trilinear_lookup(table: np.ndarray, colors: np.ndarray) -> np.ndarray:
    size = table.shape[0]
    position = np.clip(colors, 0.0, 1.0) * (size - 1)
    low = np.floor(position).astype(np.int32)
    high = np.minimum(low + 1, size - 1)
    frac = position - low

    c000 = table[low[:, 0], low[:, 1], low[:, 2]]
    c100 = table[high[:, 0], low[:, 1], low[:, 2]]
    c010 = table[low[:, 0], high[:, 1], low[:, 2]]
    c001 = table[low[:, 0], low[:, 1], high[:, 2]]
    c110 = table[high[:, 0], high[:, 1], low[:, 2]]
    c101 = table[high[:, 0], low[:, 1], high[:, 2]]
    c011 = table[low[:, 0], high[:, 1], high[:, 2]]
    c111 = table[high[:, 0], high[:, 1], high[:, 2]]

    wx = frac[:, 0:1]
    wy = frac[:, 1:2]
    wz = frac[:, 2:3]
    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return np.clip(c0 * (1.0 - wz) + c1 * wz, 0.0, 1.0).astype(np.float32)
