"""Optional EasyOCR implementation.

EasyOCR is an optional dependency. This module imports it lazily so the
rest of the package works without it installed.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from PIL import Image

# Readers are keyed by the sorted language tuple so that different language
# combinations each get their own initialised reader.
_EASYOCR_READERS: dict = {}


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
    """Run OCR using EasyOCR. Initializes and caches a reader per language set."""
    import easyocr  # noqa: PLC0415 — optional dependency, imported lazily

    lang_key = tuple(sorted(languages))
    if lang_key not in _EASYOCR_READERS:
        _EASYOCR_READERS[lang_key] = easyocr.Reader(list(lang_key), gpu=False)

    arr = np.array(image.convert("RGB"))

    results = _EASYOCR_READERS[lang_key].readtext(
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
