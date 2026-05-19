"""OpenCV layout detection, layout signatures, and overlay image writing."""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from ..models import Rect
from .loading import pil_to_cv


# ---------------------------------------------------------------------------
# IoU helper (used only within layout detection for NMS-style deduplication)
# ---------------------------------------------------------------------------

def iou_rect(a: Rect, b: Rect) -> float:
    """Intersection-over-union for two Rect objects."""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h

    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter

    if union <= 0:
        return 0.0

    return inter / union


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def detect_layout_rects(image: Image.Image) -> List[Rect]:
    """Detect large UI regions: panels, windows, frames, cards, modals.

    This is not semantic UI detection. It is composition detection based on:
    - edges,
    - contours,
    - rectangular bounding boxes,
    - area ratio,
    - aspect ratio,
    - centrality.
    """
    cv_img = pil_to_cv(image)
    height, width = cv_img.shape[:2]
    total_area = width * height

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 40, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: List[Rect] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_ratio = area / total_area

        if area_ratio < 0.01:
            continue

        if area_ratio > 0.98:
            continue

        aspect = w / max(h, 1)
        cx = x + w / 2
        cy = y + h / 2

        center_dx = abs(cx - width / 2) / width
        center_dy = abs(cy - height / 2) / height

        kind = "region"

        if 0.08 <= area_ratio <= 0.65 and center_dx < 0.18 and center_dy < 0.18:
            kind = "modal_or_dialog"
        elif h > height * 0.55 and w < width * 0.35:
            kind = "sidebar_or_panel"
        elif w > width * 0.65 and h < height * 0.22 and y < height * 0.25:
            kind = "topbar_or_header"
        elif w > width * 0.65 and h < height * 0.25 and y > height * 0.65:
            kind = "footer_or_bottom_panel"
        elif w > width * 0.45 and h > height * 0.25:
            kind = "content_frame"

        rects.append(
            Rect(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area_ratio=float(area_ratio),
                aspect_ratio=float(aspect),
                kind=kind,
            )
        )

    rects.sort(key=lambda r: r.area_ratio, reverse=True)

    filtered: List[Rect] = []

    for rect in rects:
        if not any(iou_rect(rect, existing) > 0.80 for existing in filtered):
            filtered.append(rect)

    return filtered[:20]


# ---------------------------------------------------------------------------
# Layout signature
# ---------------------------------------------------------------------------

def build_layout_signature(image: Image.Image, rects: List[Rect]) -> List[float]:
    """Build a numeric representation of UI composition.

    The signature intentionally ignores exact text and focuses on:
    - global aspect,
    - brightness blocks,
    - edge density blocks,
    - detected rectangle positions and sizes,
    - counts of region kinds.
    """
    width, height = image.size
    cv_img = pil_to_cv(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    brightness_grid = (small.flatten() / 255.0).tolist()

    edges = cv2.Canny(gray, 40, 120)
    edge_small = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA)
    edge_grid = (edge_small.flatten() / 255.0).tolist()

    kind_counts = {
        "modal_or_dialog": 0,
        "sidebar_or_panel": 0,
        "topbar_or_header": 0,
        "footer_or_bottom_panel": 0,
        "content_frame": 0,
        "region": 0,
    }

    rect_features: List[float] = []

    for rect in rects[:10]:
        kind_counts[rect.kind] = kind_counts.get(rect.kind, 0) + 1

        rect_features.extend(
            [
                rect.x / width,
                rect.y / height,
                rect.w / width,
                rect.h / height,
                rect.area_ratio,
                min(rect.aspect_ratio / 10.0, 1.0),
            ]
        )

    while len(rect_features) < 10 * 6:
        rect_features.append(0.0)

    counts = [
        min(kind_counts["modal_or_dialog"] / 5.0, 1.0),
        min(kind_counts["sidebar_or_panel"] / 5.0, 1.0),
        min(kind_counts["topbar_or_header"] / 5.0, 1.0),
        min(kind_counts["footer_or_bottom_panel"] / 5.0, 1.0),
        min(kind_counts["content_frame"] / 10.0, 1.0),
        min(kind_counts["region"] / 20.0, 1.0),
    ]

    signature = [
        width / max(height, 1),
        height / max(width, 1),
        len(rects) / 20.0,
    ]

    signature.extend(counts)
    signature.extend(rect_features)
    signature.extend(brightness_grid)
    signature.extend(edge_grid)

    return [float(x) for x in signature]


# ---------------------------------------------------------------------------
# Overlay image writing
# ---------------------------------------------------------------------------

def rects_to_overlay_image(image_path: Path, rects: List[Rect], out_path: Path) -> None:
    """Draw colored bounding boxes for each detected rect onto a copy of the image."""
    image = cv2.imread(str(image_path))
    if image is None:
        return

    colors = {
        "modal_or_dialog": (0, 0, 255),
        "sidebar_or_panel": (255, 0, 0),
        "topbar_or_header": (0, 255, 255),
        "footer_or_bottom_panel": (255, 255, 0),
        "content_frame": (0, 255, 0),
        "region": (255, 0, 255),
    }

    for rect in rects:
        color = colors.get(rect.kind, (255, 255, 255))

        cv2.rectangle(
            image,
            (rect.x, rect.y),
            (rect.x + rect.w, rect.y + rect.h),
            color,
            2,
        )

        cv2.putText(
            image,
            rect.kind,
            (rect.x, max(20, rect.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), image)
