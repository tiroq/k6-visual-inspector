"""Pairwise similarity computation for visual, text, layout, and rule dimensions."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from ..models import ScreenshotItem
from ..image.hashing import hamming_hash_similarity
from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Individual similarity functions
# ---------------------------------------------------------------------------

def _visual_similarity(a: ScreenshotItem, b: ScreenshotItem) -> float:
    ph = hamming_hash_similarity(a.phash, b.phash)
    dh = hamming_hash_similarity(a.dhash, b.dhash)
    return (ph * 0.7) + (dh * 0.3)


def _text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)

    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0

    score = float(np.dot(va, vb) / denom)
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _rule_similarity(a: ScreenshotItem, b: ScreenshotItem) -> float:
    la = a.rule_labels
    lb = b.rule_labels

    score = 0.0
    weight = 0.0

    checks = [
        ("visual_class", 0.35),
        ("text_class", 0.40),
        ("severity", 0.10),
        ("cluster_hint", 0.15),
    ]

    for key, w in checks:
        weight += w
        if la.get(key) == lb.get(key):
            score += w

    if weight <= 0:
        return 0.0

    return score / weight


def _combined_similarity(
    visual: float,
    text: float,
    layout: float,
    rule: float,
    visual_weight: float,
    text_weight: float,
    layout_weight: float,
    rule_weight: float,
) -> float:
    total = visual_weight + text_weight + layout_weight + rule_weight

    if total <= 0:
        raise ValueError("At least one weight must be positive")

    return (
        visual * visual_weight
        + text * text_weight
        + layout * layout_weight
        + rule * rule_weight
    ) / total


# ---------------------------------------------------------------------------
# Full similarity matrix computation
# ---------------------------------------------------------------------------

def compute_similarity_matrices(
    items: List[ScreenshotItem],
    visual_weight: float,
    text_weight: float,
    layout_weight: float,
    rule_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise similarity matrices for all four dimensions plus combined."""
    n = len(items)

    visual = np.eye(n, dtype=np.float32)
    text = np.eye(n, dtype=np.float32)
    layout = np.eye(n, dtype=np.float32)
    rule = np.eye(n, dtype=np.float32)
    combined = np.eye(n, dtype=np.float32)

    for i in tqdm(range(n), desc="Comparing screenshots"):
        for j in range(i + 1, n):
            v = _visual_similarity(items[i], items[j])

            text_a = " ".join([
                items[i].normalized_text,
                items[i].central_ocr_text,
                *items[i].region_ocr_texts,
            ])

            text_b = " ".join([
                items[j].normalized_text,
                items[j].central_ocr_text,
                *items[j].region_ocr_texts,
            ])

            t = _text_similarity(text_a, text_b)
            lo = _cosine_similarity(items[i].layout_signature, items[j].layout_signature)
            r = _rule_similarity(items[i], items[j])

            c = _combined_similarity(
                visual=v,
                text=t,
                layout=lo,
                rule=r,
                visual_weight=visual_weight,
                text_weight=text_weight,
                layout_weight=layout_weight,
                rule_weight=rule_weight,
            )

            visual[i, j] = visual[j, i] = v
            text[i, j] = text[j, i] = t
            layout[i, j] = layout[j, i] = lo
            rule[i, j] = rule[j, i] = r
            combined[i, j] = combined[j, i] = c

    return visual, text, layout, rule, combined
