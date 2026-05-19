"""Crop helpers — center crop and rect-based crop."""

from __future__ import annotations

from PIL import Image

from ..models import Rect


def crop_center(image: Image.Image, margin_x: float = 0.18, margin_y: float = 0.18) -> Image.Image:
    """Crop the central region of *image*, removing *margin_x/y* fractions from each side."""
    width, height = image.size

    left = int(width * margin_x)
    top = int(height * margin_y)
    right = int(width * (1.0 - margin_x))
    bottom = int(height * (1.0 - margin_y))

    return image.crop((left, top, right, bottom))


def crop_rect(image: Image.Image, rect: Rect, padding: int = 8) -> Image.Image:
    """Crop *image* to the bounding box of *rect* with an optional pixel *padding*."""
    width, height = image.size

    left = max(0, rect.x - padding)
    top = max(0, rect.y - padding)
    right = min(width, rect.x + rect.w + padding)
    bottom = min(height, rect.y + rect.h + padding)

    return image.crop((left, top, right, bottom))
