"""Optional EasyOCR implementation.

EasyOCR is an optional dependency. This module imports it lazily so the
rest of the package works without it installed.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from PIL import Image

_EASYOCR_READER = None


def parse_easyocr_languages(ocr_lang: str) -> List[str]:
    """Map Tesseract-style language codes (e.g. 'eng+rus') to EasyOCR codes."""
    mapping = {
        "eng": "en",
        "rus": "ru",
    }

    parts = re.split(r"[+,]", ocr_lang)
    langs = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        langs.append(mapping.get(part, part))

    return langs or ["en"]


def extract_ocr_text_easyocr(image: Image.Image, languages: List[str]) -> str:
    """Run OCR using EasyOCR. Initializes the reader on first use."""
    global _EASYOCR_READER

    import easyocr  # noqa: PLC0415 — optional dependency, imported lazily

    if _EASYOCR_READER is None:
        _EASYOCR_READER = easyocr.Reader(languages, gpu=False)

    arr = np.array(image.convert("RGB"))

    results = _EASYOCR_READER.readtext(
        arr,
        detail=1,
        paragraph=True,
    )

    parts = []

    for item in results:
        if len(item) >= 2:
            text = str(item[1]).strip()
            if text:
                parts.append(text)

    return " ".join(parts)
