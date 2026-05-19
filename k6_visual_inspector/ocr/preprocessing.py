"""OCR image preprocessing — build multi-variant images for Tesseract."""

from __future__ import annotations

from typing import Any, List, Tuple

import cv2
import numpy as np
from PIL import Image


def build_ocr_image_variants(image: Image.Image, mode: str = "fast") -> List[Tuple[str, Any]]:
    """Return a list of (name, opencv_array) preprocessing variants.

    More variants are produced for 'accurate' mode, fewer for 'fast'.
    """
    variants: List[Tuple[str, Any]] = []

    image = image.convert("RGB")
    w, h = image.size

    scale = 2.0 if max(w, h) < 2500 else 1.5
    upscaled = image.resize(
        (int(w * scale), int(h * scale)),
        Image.Resampling.LANCZOS,
    )

    arr = np.array(upscaled)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    variants.append(("gray", gray))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    variants.append(("clahe", contrast))

    denoised = cv2.bilateralFilter(contrast, 5, 75, 75)
    variants.append(("denoised", denoised))

    blur = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    sharpened = cv2.addWeighted(denoised, 1.6, blur, -0.6, 0)
    variants.append(("sharpened", sharpened))

    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    variants.append(("otsu", otsu))

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )
    variants.append(("adaptive", adaptive))

    variants.append(("gray_inverted", cv2.bitwise_not(gray)))
    variants.append(("otsu_inverted", cv2.bitwise_not(otsu)))
    variants.append(("adaptive_inverted", cv2.bitwise_not(adaptive)))

    return variants
