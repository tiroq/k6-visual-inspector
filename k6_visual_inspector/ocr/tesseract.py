"""Tesseract OCR implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pytesseract
from PIL import Image

from .cleanup import count_useful_chars
from .preprocessing import build_ocr_image_variants


def _ocr_data_to_text_and_score(data: Dict[str, Any]) -> Tuple[str, float]:
    """Convert pytesseract image_to_data output into (text, quality_score)."""
    words: List[str] = []
    confidences: List[float] = []

    for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
        text = str(raw_text).strip()

        if not text:
            continue

        try:
            conf = float(raw_conf)
        except ValueError:
            conf = -1.0

        if conf < 0:
            continue

        words.append(text)
        confidences.append(conf)

    if not words:
        return "", 0.0

    joined = " ".join(words)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    alpha_ratio = count_useful_chars(joined) / max(len(joined), 1)
    length_bonus = min(len(words) / 30.0, 1.0)

    score = (avg_conf / 100.0) * 0.75 + alpha_ratio * 0.15 + length_bonus * 0.10

    return joined, score


def extract_ocr_text_tesseract(image: Image.Image, lang: str, mode: str = "fast") -> str:
    """Run OCR using multiple preprocessing strategies and choose the best result.

    This is slower than a single OCR pass, but much better for UI screenshots.
    """
    candidates = []
    prepared_images = build_ocr_image_variants(image, mode=mode)

    # 6  = assume a uniform block of text
    # 11 = sparse text
    # 12 = sparse text with OSD
    # 3  = fully automatic page segmentation
    if mode == "fast":
        psm_modes = [6]
    elif mode == "balanced":
        psm_modes = [6, 11]
    else:
        psm_modes = [6, 11, 12, 3]

    for variant_name, variant in prepared_images:
        for psm in psm_modes:
            config = f"--oem 3 --psm {psm}"

            try:
                data = pytesseract.image_to_data(
                    variant,
                    lang=lang,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
            except pytesseract.TesseractError:
                data = pytesseract.image_to_data(
                    variant,
                    lang="eng",
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )

            text, score = _ocr_data_to_text_and_score(data)

            if text.strip():
                candidates.append(
                    {
                        "variant": variant_name,
                        "psm": psm,
                        "text": text,
                        "score": score,
                    }
                )

    if not candidates:
        return ""

    candidates.sort(
        key=lambda x: (
            x["score"],
            min(len(x["text"]), 2000),
        ),
        reverse=True,
    )

    return candidates[0]["text"]
