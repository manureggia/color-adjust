from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RGBArray = np.ndarray


def load_rgb(path: str | Path) -> RGBArray:
    """Load an image as float32 RGB in [0, 1]."""
    image = Image.open(path).convert("RGB")
    return _as_float_rgb(np.asarray(image, dtype=np.uint8))


def save_rgb(path: str | Path, image: RGBArray) -> None:
    """Save a float RGB image, clipping to [0, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(np.rint(_as_float_rgb(image) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def save_comparison(
    path: str | Path,
    panels: list[tuple[str, RGBArray]],
    panel_width: int = 420,
    title: str | None = None,
    footer: str | None = None,
) -> None:
    """Save a contact sheet for quick visual comparison."""
    if not panels:
        raise ValueError("Expected at least one comparison panel.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    thumbnails: list[tuple[str, Image.Image]] = []
    for label, image in panels:
        array = np.clip(np.rint(_as_float_rgb(image) * 255.0), 0, 255).astype(np.uint8)
        pil = Image.fromarray(array, mode="RGB")
        width, height = pil.size
        scale = panel_width / float(width)
        size = (panel_width, max(1, int(round(height * scale))))
        thumbnails.append((label, pil.resize(size, Image.Resampling.LANCZOS)))

    padding = 14
    label_height = 28
    line_spacing = 4
    font = ImageFont.load_default()
    sheet_width = len(thumbnails) * panel_width + (len(thumbnails) + 1) * padding
    text_width = sheet_width - 2 * padding
    title_lines = _wrap_text(title, text_width, font) if title else []
    footer_lines = _wrap_text(footer, text_width, font) if footer else []
    text_line_height = _text_line_height(font)
    title_height = _text_block_height(title_lines, text_line_height, line_spacing)
    footer_height = _text_block_height(footer_lines, text_line_height, line_spacing)
    title_gap = padding if title_lines else 0
    footer_gap = padding if footer_lines else 0
    panels_top = padding + title_height + title_gap
    sheet_height = (
        panels_top
        + label_height
        + max(image.height for _, image in thumbnails)
        + footer_gap
        + footer_height
        + padding
    )
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)

    y = padding
    for line in title_lines:
        draw.text((padding, y), line, fill=(0, 0, 0), font=font)
        y += text_line_height + line_spacing

    for index, (label, image) in enumerate(thumbnails):
        x = padding + index * (panel_width + padding)
        draw.text((x, panels_top), label, fill=(0, 0, 0), font=font)
        sheet.paste(image, (x, panels_top + label_height))

    y = panels_top + label_height + max(image.height for _, image in thumbnails) + footer_gap
    for line in footer_lines:
        draw.text((padding, y), line, fill=(0, 0, 0), font=font)
        y += text_line_height + line_spacing

    sheet.save(path)


def _wrap_text(text: str | None, max_width: int, font: ImageFont.ImageFont) -> list[str]:
    if not text:
        return []
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[str] = []
    for raw_line in text.splitlines():
        words = raw_line.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if _text_width(draw, candidate, font) <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _text_line_height(font: ImageFont.ImageFont) -> int:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return bbox[3] - bbox[1]


def _text_block_height(lines: list[str], line_height: int, line_spacing: int) -> int:
    if not lines:
        return 0
    return len(lines) * line_height + (len(lines) - 1) * line_spacing


def resize_rgb(image: RGBArray, size: tuple[int, int]) -> RGBArray:
    """Resize a float RGB image to PIL size order: (width, height)."""
    if size[0] < 1 or size[1] < 1:
        raise ValueError("Resize dimensions must be positive.")
    array = np.clip(np.rint(_as_float_rgb(image) * 255.0), 0, 255).astype(np.uint8)
    pil = Image.fromarray(array, mode="RGB")
    return np.asarray(pil.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0


def ensure_same_size(source: RGBArray, target: RGBArray) -> RGBArray:
    """Resize target to source spatial dimensions when needed."""
    source = _as_float_rgb(source)
    target = _as_float_rgb(target)
    if source.shape[:2] == target.shape[:2]:
        return target
    height, width = source.shape[:2]
    return resize_rgb(target, (width, height))


def _as_float_rgb(image: RGBArray) -> RGBArray:
    """Return an HxWx3 RGB float32 array normalized to [0, 1]."""
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
