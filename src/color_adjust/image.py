from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


RGBArray = np.ndarray


def load_rgb(path: str | Path) -> RGBArray:
    """Load an image as float32 RGB in [0, 1]."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_rgb(path: str | Path, image: RGBArray) -> None:
    """Save a float RGB image, clipping to [0, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(np.rint(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def save_comparison(
    path: str | Path,
    panels: list[tuple[str, RGBArray]],
    panel_width: int = 420,
) -> None:
    """Save a contact sheet for quick visual comparison."""
    if not panels:
        raise ValueError("Expected at least one comparison panel.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    thumbnails: list[tuple[str, Image.Image]] = []
    for label, image in panels:
        array = np.clip(np.rint(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)
        pil = Image.fromarray(array, mode="RGB")
        width, height = pil.size
        scale = panel_width / float(width)
        size = (panel_width, max(1, int(round(height * scale))))
        thumbnails.append((label, pil.resize(size, Image.Resampling.LANCZOS)))

    padding = 14
    label_height = 28
    sheet_width = len(thumbnails) * panel_width + (len(thumbnails) + 1) * padding
    sheet_height = max(image.height for _, image in thumbnails) + label_height + 2 * padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    for index, (label, image) in enumerate(thumbnails):
        x = padding + index * (panel_width + padding)
        draw.text((x, padding), label, fill=(0, 0, 0))
        sheet.paste(image, (x, padding + label_height))

    sheet.save(path)


def resize_rgb(image: RGBArray, size: tuple[int, int]) -> RGBArray:
    """Resize a float RGB image to PIL size order: (width, height)."""
    array = np.clip(np.rint(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)
    pil = Image.fromarray(array, mode="RGB")
    return np.asarray(pil.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


def ensure_same_size(source: RGBArray, target: RGBArray) -> RGBArray:
    """Resize target to source spatial dimensions when needed."""
    if source.shape[:2] == target.shape[:2]:
        return target
    height, width = source.shape[:2]
    return resize_rgb(target, (width, height))
