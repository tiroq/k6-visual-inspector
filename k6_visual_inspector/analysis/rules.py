"""Rule-based label detection for UI screenshots."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..models import Rect


# ---------------------------------------------------------------------------
# Low-level text pattern helpers
# ---------------------------------------------------------------------------

def looks_like_error_text(text: str) -> bool:
    return bool(
        re.search(
            r"\b(error|failed|failure|exception|unable|cannot|could not|invalid|denied|unauthorized|forbidden|timeout|unavailable)\b",
            text,
        )
    )


def looks_like_browser_error(text: str) -> bool:
    return bool(
        re.search(
            r"\b(this site can.t be reached|connection refused|bad gateway|service unavailable|gateway timeout|http error|page crashed|aw snap)\b",
            text,
        )
    )


def looks_like_loading(text: str) -> bool:
    return bool(
        re.search(
            r"\b(loading|please wait|processing|spinner|in progress|загрузка|подождите)\b",
            text,
        )
    )


def looks_like_empty_state(text: str) -> bool:
    return bool(
        re.search(
            r"\b(no data|nothing found|empty|no records|no results|нет данных|ничего не найдено)\b",
            text,
        )
    )


# ---------------------------------------------------------------------------
# Main rule label detection
# ---------------------------------------------------------------------------

def detect_rule_labels(
    normalized_text: str,
    central_text: str,
    region_texts: List[str],
    rects: List[Rect],
    width: int,
    height: int,
) -> Dict[str, Any]:
    """Derive rule-based semantic labels from OCR text and detected layout rects."""
    joined = " ".join([normalized_text, central_text, *region_texts]).lower()
    rect_kinds = [r.kind for r in rects]

    has_modal = "modal_or_dialog" in rect_kinds
    has_content_frame = "content_frame" in rect_kinds
    has_topbar = "topbar_or_header" in rect_kinds
    has_sidebar = "sidebar_or_panel" in rect_kinds

    visual_class = "unknown"
    text_class = "unknown"
    severity = "medium"

    if has_modal:
        visual_class = "error_modal" if looks_like_error_text(joined) else "modal"
    elif looks_like_browser_error(joined):
        visual_class = "browser_error"
    elif looks_like_loading(joined):
        visual_class = "loading"
    elif looks_like_empty_state(joined):
        visual_class = "empty_state"
    elif has_content_frame or has_topbar or has_sidebar:
        visual_class = "normal_or_content_page"
    else:
        visual_class = "unknown"

    if re.search(r"\b(500|internal server error|server error|backend error|backend_error)\b", joined):
        text_class = "backend_500"
        severity = "high"
    elif re.search(r"\b(502|bad gateway)\b", joined):
        text_class = "bad_gateway_502"
        severity = "high"
    elif re.search(r"\b(503|service unavailable|temporarily unavailable)\b", joined):
        text_class = "service_unavailable_503"
        severity = "high"
    elif re.search(r"\b(504|gateway timeout|timeout|timed out|deadline exceeded)\b", joined):
        text_class = "timeout"
        severity = "high"
    elif re.search(r"\b(401|unauthorized|not authorized|login required)\b", joined):
        text_class = "unauthorized"
        severity = "high"
    elif re.search(r"\b(403|forbidden|access denied|permission denied)\b", joined):
        text_class = "forbidden"
        severity = "high"
    elif re.search(r"\b(404|not found)\b", joined):
        text_class = "not_found"
        severity = "medium"
    elif re.search(r"\b(429|too many requests|rate limit|rate_limited)\b", joined):
        text_class = "rate_limit"
        severity = "high"
    elif re.search(r"\b(validation|invalid|required|missing|incorrect)\b", joined):
        text_class = "validation_error"
        severity = "medium"
    elif looks_like_loading(joined):
        text_class = "loading_or_spinner"
        severity = "medium"
    elif looks_like_empty_state(joined):
        text_class = "empty_state"
        severity = "low"
    elif not joined.strip():
        text_class = "no_text"
        severity = "medium"

    if visual_class == "error_modal" and text_class == "unknown":
        text_class = "generic_error"
        severity = "high"

    cluster_hint = f"{visual_class}.{text_class}"

    return {
        "visual_class": visual_class,
        "text_class": text_class,
        "severity": severity,
        "cluster_hint": cluster_hint,
        "has_modal": has_modal,
        "has_content_frame": has_content_frame,
        "has_topbar": has_topbar,
        "has_sidebar": has_sidebar,
        "rect_kinds": rect_kinds,
    }
