"""OCR engine dispatch — unified extract_ocr_text interface."""

from __future__ import annotations

from PIL import Image

from .tesseract import extract_ocr_text_tesseract
from .easyocr_engine import extract_ocr_text_easyocr, parse_easyocr_languages


def extract_ocr_text(
    image: Image.Image,
    lang: str,
    engine: str,
    mode: str = "fast",
) -> str:
    """Dispatch OCR to the selected engine.

    Parameters
    ----------
    image:  PIL image (RGB)
    lang:   Tesseract-style language code, e.g. ``"eng"`` or ``"eng+rus"``
    engine: ``"tesseract"`` or ``"easyocr"``
    mode:   ``"fast"``, ``"balanced"``, or ``"accurate"`` (Tesseract only)
    """
    if engine == "tesseract":
        return extract_ocr_text_tesseract(image, lang=lang, mode=mode)

    if engine == "easyocr":
        languages = parse_easyocr_languages(lang)
        return extract_ocr_text_easyocr(image, languages=languages)

    raise ValueError(f"Unsupported OCR engine: {engine!r}")
