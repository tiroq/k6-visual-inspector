"""OCR text normalization, tokenization, and filtering helpers."""

from __future__ import annotations

import re
from typing import List

from rapidfuzz import fuzz


def normalize_text(text: str) -> str:
    """Normalize raw OCR text by replacing variable tokens with placeholders."""
    text = text.lower()

    replacements = [
        (r"https?://\S+", " <url> "),
        (r"\b[\w.+-]+@[\w.-]+\.\w+\b", " <email> "),
        (
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            " <uuid> ",
        ),
        (r"\b[0-9a-f]{10,}\b", " <hex> "),
        (r"\b\d{4}-\d{2}-\d{2}[t\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?z?\b", " <datetime> "),
        (r"\b\d{4}-\d{2}-\d{2}\b", " <date> "),
        (r"\b\d{1,2}[/:]\d{2}(?::\d{2})?\b", " <time> "),
        (r"\b(order|trade|request|session|correlation|trace|span)[-_ ]?id[:= ]+[a-z0-9._:-]+\b", r" \1_id <id> "),
        (r"\b[a-z]{2,10}-\d{3,}\b", " <business_id> "),
        (r"\b\d+(?:[.,]\d+)?\b", " <num> "),
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r"\b5xx\b", " backend_error ", text)
    text = re.sub(r"\b4xx\b", " client_error ", text)

    text = re.sub(r"[^\w<>/.-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """Split normalized text into meaningful tokens, removing stopwords."""
    tokens = re.findall(r"[a-zа-яё0-9_<>/-]{2,}", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "you",
        "your",
        "this",
        "that",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "not",
        "but",
        "или",
        "это",
        "как",
        "что",
        "для",
        "при",
        "над",
        "под",
    }
    return [t for t in tokens if t not in stop]


def count_useful_chars(text: str) -> int:
    """Count alphanumeric and structurally meaningful characters in *text*."""
    return sum(1 for ch in text if ch.isalnum() or ch in "<>/_-:.")


def is_useful_ocr_text(text: str) -> bool:
    """Return True when *text* has enough content to be worth keeping."""
    if not text:
        return False

    if len(text) < 3:
        return False

    useful = count_useful_chars(text)
    ratio = useful / max(len(text), 1)

    if ratio < 0.45:
        return False

    return True


def deduplicate_texts(texts: List[str]) -> List[str]:
    """Remove near-duplicate strings from *texts* using fuzzy matching."""
    result: List[str] = []

    for text in texts:
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        duplicate = False

        for existing in result:
            if fuzz.token_set_ratio(text, existing) >= 92:
                duplicate = True
                break

        if not duplicate:
            result.append(text)

    return result
