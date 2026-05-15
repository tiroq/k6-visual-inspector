"""Image loading helpers."""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path

from PIL import Image


def load_image(path: Path) -> Image.Image:
    """Open *path* and return an RGB PIL image."""
    return Image.open(path).convert("RGB")


def pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a BGR OpenCV ndarray."""
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
