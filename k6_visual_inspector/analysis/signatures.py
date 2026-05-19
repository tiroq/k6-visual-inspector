"""Semantic signature building from a screenshot item."""

from __future__ import annotations

from typing import Any, Dict, List

from ..models import Rect


def build_semantic_signature(item_like: Dict[str, Any]) -> str:
    """Build a human-readable semantic description for use in debugging and reports."""
    rects = item_like["rects"]
    rule_labels = item_like["rule_labels"]

    rect_kinds = []
    for r in rects[:10]:
        if isinstance(r, Rect):
            rect_kinds.append(r.kind)
        else:
            rect_kinds.append(r["kind"])

    parts = [
        "UI screenshot from k6/browser load test.",
        f"Visual class: {rule_labels.get('visual_class', 'unknown')}.",
        f"Text class: {rule_labels.get('text_class', 'unknown')}.",
        f"Severity: {rule_labels.get('severity', 'unknown')}.",
        f"Layout regions: {', '.join(rect_kinds)}.",
        f"Normalized OCR: {item_like.get('normalized_text', '')}",
        f"Central OCR: {item_like.get('central_ocr_text', '')}",
        f"Region OCR: {' | '.join(item_like.get('region_ocr_texts', []))}",
    ]

    return "\n".join(parts)
