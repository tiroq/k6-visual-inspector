#!/usr/bin/env python3
"""
cluster_screenshots.py

Cluster UI screenshots by:
1. visual similarity,
2. OCR text similarity,
3. layout / composition similarity,
4. rule-based semantic labels.

Designed for post-processing k6/browser screenshots when no explicit metadata is available.

Outputs:
    - report.html
    - clusters.json
    - items.jsonl
    - cluster-summary.csv
    - similarity_*.npy
    - overlays/
    - clusters/
    - debug-ocr/ when --debug-ocr is enabled

Install:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install pillow opencv-python imagehash pytesseract rapidfuzz numpy scikit-learn tqdm

System dependency:
    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
    sudo apt-get install -y tesseract-ocr-rus  # optional

Optional EasyOCR:
    pip install easyocr

Usage:
    python3 cluster_screenshots.py ./screenshots ./analysis

Examples:
    python3 cluster_screenshots.py ./screenshots ./analysis --ocr-lang eng --debug-ocr

    python3 cluster_screenshots.py ./screenshots ./analysis \
      --visual-weight 0.30 \
      --text-weight 0.25 \
      --layout-weight 0.25 \
      --rule-weight 0.20 \
      --template-threshold 0.84 \
      --final-threshold 0.78

    python3 cluster_screenshots.py ./screenshots ./analysis \
      --ocr-engine easyocr \
      --ocr-lang eng \
      --debug-ocr

Notes:
    - Comments are intentionally in English.
    - OCR quality depends heavily on screenshot resolution and OCR engine.
    - For future k6/browser runs, prefer storing DOM visible text as metadata.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
import json
import math
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
import pytesseract
from PIL import Image
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

_EASYOCR_READER = None


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int
    area_ratio: float
    aspect_ratio: float
    kind: str


@dataclass
class ScreenshotItem:
    index: int
    path: str
    filename: str
    width: int
    height: int

    sha256: str
    phash: str
    dhash: str

    ocr_text: str
    normalized_text: str
    central_ocr_text: str
    region_ocr_texts: List[str]
    text_tokens: List[str]

    rects: List[Rect]
    layout_signature: List[float]
    semantic_signature: str

    rule_labels: Dict[str, Any]

    template_cluster_id: Optional[int] = None
    final_cluster_id: Optional[int] = None
    nearest_to_representative_score: Optional[float] = None


@dataclass
class Cluster:
    cluster_id: int
    count: int
    representative_index: int
    representative_path: str

    cluster_name: str
    visual_class: str
    text_class: str
    severity: str

    avg_visual_similarity: float
    avg_text_similarity: float
    avg_layout_similarity: float
    avg_rule_similarity: float
    avg_combined_similarity: float

    items: List[int] = field(default_factory=list)
    common_tokens: List[str] = field(default_factory=list)
    text_variants: List[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "unknown"


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_cv(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def normalize_text(text: str) -> str:
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
    return sum(1 for ch in text if ch.isalnum() or ch in "<>/_-:.")


def is_useful_ocr_text(text: str) -> bool:
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


def build_ocr_image_variants(image: Image.Image, mode: str = "fast") -> List[Tuple[str, Any]]:
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


def ocr_data_to_text_and_score(data: Dict[str, Any]) -> Tuple[str, float]:
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
    """
    Run OCR using multiple preprocessing strategies and choose the best result.

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

            text, score = ocr_data_to_text_and_score(data)

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


def parse_easyocr_languages(ocr_lang: str) -> List[str]:
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
    global _EASYOCR_READER

    import easyocr

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


def extract_ocr_text_with_engine(
    image: Image.Image,
    lang: str,
    engine: str,
    mode: str = "fast",
) -> str:
    if engine == "tesseract":
        return extract_ocr_text_tesseract(image, lang=lang, mode=mode)

    if engine == "easyocr":
        languages = parse_easyocr_languages(lang)
        return extract_ocr_text_easyocr(image, languages=languages)

    raise ValueError(f"Unsupported OCR engine: {engine}")


def crop_center(image: Image.Image, margin_x: float = 0.18, margin_y: float = 0.18) -> Image.Image:
    width, height = image.size

    left = int(width * margin_x)
    top = int(height * margin_y)
    right = int(width * (1.0 - margin_x))
    bottom = int(height * (1.0 - margin_y))

    return image.crop((left, top, right, bottom))


def crop_rect(image: Image.Image, rect: Rect, padding: int = 8) -> Image.Image:
    width, height = image.size

    left = max(0, rect.x - padding)
    top = max(0, rect.y - padding)
    right = min(width, rect.x + rect.w + padding)
    bottom = min(height, rect.y + rect.h + padding)

    return image.crop((left, top, right, bottom))


def iou_rect(a: Rect, b: Rect) -> float:
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter

    if union <= 0:
        return 0.0

    return inter / union


def detect_layout_rects(image: Image.Image) -> List[Rect]:
    """
    Detect large UI regions: panels, windows, frames, cards, modals.

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


def extract_region_ocr_texts(
    image: Image.Image,
    rects: List[Rect],
    lang: str,
    engine: str,
    mode: str,
) -> List[str]:
    """
    Run OCR on the most informative UI regions.

    Priorities:
    - modal/dialog;
    - large content frame;
    - top banner / toast-like area;
    - center crop.
    """
    texts: List[str] = []
    width, height = image.size
    selected: List[Rect] = []

    selected.extend([
        r for r in rects
        if r.kind == "modal_or_dialog" and 0.03 <= r.area_ratio <= 0.75
    ])

    selected.extend([
        r for r in rects
        if r.kind == "content_frame" and 0.08 <= r.area_ratio <= 0.85
    ])

    selected.extend([
        r for r in rects
        if r.kind in {"topbar_or_header", "region"} and r.y < height * 0.35 and r.area_ratio >= 0.015
    ])

    filtered: List[Rect] = []

    for rect in selected:
        if not any(iou_rect(rect, existing) > 0.70 for existing in filtered):
            filtered.append(rect)

    if mode == "fast":
        filtered = filtered[:2]
    elif mode == "balanced":
        filtered = filtered[:4]
    else:
        filtered = filtered[:8]

    for rect in filtered:
        try:
            cropped = crop_rect(image, rect, padding=12)
            raw = extract_ocr_text_with_engine(cropped, lang=lang, engine=engine, mode=mode)
            norm = normalize_text(raw)

            if is_useful_ocr_text(norm):
                texts.append(norm)

        except Exception:
            continue

    try:
        center = crop_center(image, margin_x=0.15, margin_y=0.15)
        raw = extract_ocr_text_with_engine(center, lang=lang, engine=engine, mode=mode)
        norm = normalize_text(raw)

        if is_useful_ocr_text(norm):
            texts.append(norm)

    except Exception:
        pass

    return deduplicate_texts(texts)


def build_layout_signature(image: Image.Image, rects: List[Rect]) -> List[float]:
    """
    Build a numeric representation of UI composition.

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


def detect_rule_labels(
    normalized_text: str,
    central_text: str,
    region_texts: List[str],
    rects: List[Rect],
    width: int,
    height: int,
) -> Dict[str, Any]:
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


def build_semantic_signature(item_like: Dict[str, Any]) -> str:
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


def save_ocr_debug_crops(
    image: Image.Image,
    rects: List[Rect],
    original_path: Path,
    out_dir: Path,
    index: int,
) -> None:
    base = safe_name(original_path.stem)
    item_dir = out_dir / f"{index:05d}__{base}"
    item_dir.mkdir(parents=True, exist_ok=True)

    image.save(item_dir / "full.png")

    center = crop_center(image, margin_x=0.15, margin_y=0.15)
    center.save(item_dir / "center.png")

    for i, rect in enumerate(rects[:10]):
        crop = crop_rect(image, rect, padding=12)
        crop.save(item_dir / f"rect-{i:02d}__{rect.kind}.png")


def analyze_screenshot(
    path: Path,
    index: int,
    ocr_lang: str,
    ocr_engine: str,
    ocr_mode: str,
    debug_ocr_dir: Optional[Path] = None,
) -> ScreenshotItem:
    image = load_image(path)
    width, height = image.size

    p_hash = imagehash.phash(image)
    d_hash = imagehash.dhash(image)

    rects = detect_layout_rects(image)

    if debug_ocr_dir is not None:
        save_ocr_debug_crops(
            image=image,
            rects=rects,
            original_path=path,
            out_dir=debug_ocr_dir,
            index=index,
        )

    ocr_text = extract_ocr_text_with_engine(image, lang=ocr_lang, engine=ocr_engine, mode=ocr_mode)
    normalized = normalize_text(ocr_text)

    central_img = crop_center(image)
    central_text_raw = extract_ocr_text_with_engine(central_img, lang=ocr_lang, engine=ocr_engine, mode=ocr_mode)
    central_text = normalize_text(central_text_raw)

    region_texts = extract_region_ocr_texts(
        image=image,
        rects=rects,
        lang=ocr_lang,
        engine=ocr_engine,
        mode=ocr_mode,
    )

    all_text_for_tokens = " ".join([normalized, central_text, *region_texts])
    tokens = tokenize_text(all_text_for_tokens)

    rule_labels = detect_rule_labels(
        normalized_text=normalized,
        central_text=central_text,
        region_texts=region_texts,
        rects=rects,
        width=width,
        height=height,
    )

    item_like = {
        "normalized_text": normalized,
        "central_ocr_text": central_text,
        "region_ocr_texts": region_texts,
        "rects": rects,
        "rule_labels": rule_labels,
    }

    semantic_signature = build_semantic_signature(item_like)
    layout_signature = build_layout_signature(image, rects)

    return ScreenshotItem(
        index=index,
        path=str(path),
        filename=path.name,
        width=width,
        height=height,
        sha256=sha256_file(path),
        phash=str(p_hash),
        dhash=str(d_hash),
        ocr_text=ocr_text,
        normalized_text=normalized,
        central_ocr_text=central_text,
        region_ocr_texts=region_texts,
        text_tokens=tokens,
        rects=rects,
        layout_signature=layout_signature,
        semantic_signature=semantic_signature,
        rule_labels=rule_labels,
    )


def hamming_hash_similarity(hash_a: str, hash_b: str, hash_bits: int = 64) -> float:
    a = imagehash.hex_to_hash(hash_a)
    b = imagehash.hex_to_hash(hash_b)
    distance = a - b
    return max(0.0, 1.0 - (distance / hash_bits))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)

    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0

    score = float(np.dot(va, vb) / denom)
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b) / 100.0


def rule_similarity(a: ScreenshotItem, b: ScreenshotItem) -> float:
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


def combined_similarity(
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


def find_images(input_dir: Path) -> List[Path]:
    files: List[Path] = []

    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


def compute_similarity_matrices(
    items: List[ScreenshotItem],
    visual_weight: float,
    text_weight: float,
    layout_weight: float,
    rule_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(items)

    visual = np.eye(n, dtype=np.float32)
    text = np.eye(n, dtype=np.float32)
    layout = np.eye(n, dtype=np.float32)
    rule = np.eye(n, dtype=np.float32)
    combined = np.eye(n, dtype=np.float32)

    for i in tqdm(range(n), desc="Comparing screenshots"):
        for j in range(i + 1, n):
            ph = hamming_hash_similarity(items[i].phash, items[j].phash)
            dh = hamming_hash_similarity(items[i].dhash, items[j].dhash)
            v = (ph * 0.7) + (dh * 0.3)

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

            t = text_similarity(text_a, text_b)
            l = cosine_similarity(items[i].layout_signature, items[j].layout_signature)
            r = rule_similarity(items[i], items[j])

            c = combined_similarity(
                visual=v,
                text=t,
                layout=l,
                rule=r,
                visual_weight=visual_weight,
                text_weight=text_weight,
                layout_weight=layout_weight,
                rule_weight=rule_weight,
            )

            visual[i, j] = visual[j, i] = v
            text[i, j] = text[j, i] = t
            layout[i, j] = layout[j, i] = l
            rule[i, j] = rule[j, i] = r
            combined[i, j] = combined[j, i] = c

    return visual, text, layout, rule, combined


def cluster_by_matrix(
    similarity_matrix: np.ndarray,
    threshold: float,
) -> List[int]:
    n = similarity_matrix.shape[0]

    if n == 0:
        return []

    if n == 1:
        return [0]

    distance_matrix = 1.0 - similarity_matrix
    distance_threshold = 1.0 - threshold

    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )

    return [int(x) for x in model.fit_predict(distance_matrix)]


def cluster_items_single_stage(
    items: List[ScreenshotItem],
    combined_similarity_matrix: np.ndarray,
    threshold: float,
) -> List[int]:
    return cluster_by_matrix(combined_similarity_matrix, threshold=threshold)


def two_stage_cluster_items(
    items: List[ScreenshotItem],
    visual_matrix: np.ndarray,
    layout_matrix: np.ndarray,
    text_matrix: np.ndarray,
    rule_matrix: np.ndarray,
    template_threshold: float,
    final_threshold: float,
) -> List[int]:
    n = len(items)

    template_matrix = (
        visual_matrix * 0.45
        + layout_matrix * 0.45
        + rule_matrix * 0.10
    )

    template_labels = cluster_by_matrix(template_matrix, threshold=template_threshold)

    for item, label in zip(items, template_labels):
        item.template_cluster_id = label

    template_to_indices: Dict[int, List[int]] = {}

    for idx, label in enumerate(template_labels):
        template_to_indices.setdefault(label, []).append(idx)

    final_labels = [-1] * n
    next_cluster_id = 0

    for _, indices in sorted(template_to_indices.items(), key=lambda kv: min(kv[1])):
        if len(indices) == 1:
            final_labels[indices[0]] = next_cluster_id
            next_cluster_id += 1
            continue

        sub_matrix = np.eye(len(indices), dtype=np.float32)

        for local_i, global_i in enumerate(indices):
            for local_j, global_j in enumerate(indices):
                if local_i >= local_j:
                    continue

                score = (
                    visual_matrix[global_i, global_j] * 0.20
                    + layout_matrix[global_i, global_j] * 0.20
                    + text_matrix[global_i, global_j] * 0.35
                    + rule_matrix[global_i, global_j] * 0.25
                )

                sub_matrix[local_i, local_j] = score
                sub_matrix[local_j, local_i] = score

        sub_labels = cluster_by_matrix(sub_matrix, threshold=final_threshold)
        local_to_global_cluster: Dict[int, int] = {}

        for local_idx, sub_label in enumerate(sub_labels):
            if sub_label not in local_to_global_cluster:
                local_to_global_cluster[sub_label] = next_cluster_id
                next_cluster_id += 1

            global_idx = indices[local_idx]
            final_labels[global_idx] = local_to_global_cluster[sub_label]

    for item, label in zip(items, final_labels):
        item.final_cluster_id = label

    return final_labels


def choose_representative(
    indices: List[int],
    combined_matrix: np.ndarray,
) -> int:
    if len(indices) == 1:
        return indices[0]

    best_index = indices[0]
    best_score = -1.0

    for idx in indices:
        scores = [combined_matrix[idx, other] for other in indices if other != idx]
        avg = float(np.mean(scores)) if scores else 1.0

        if avg > best_score:
            best_score = avg
            best_index = idx

    return best_index


def common_tokens_for_cluster(items: List[ScreenshotItem], indices: List[int], limit: int = 20) -> List[str]:
    freq: Dict[str, int] = {}

    for idx in indices:
        unique_tokens = set(items[idx].text_tokens)
        for token in unique_tokens:
            freq[token] = freq.get(token, 0) + 1

    min_count = max(1, math.ceil(len(indices) * 0.4))

    common = [
        token
        for token, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        if count >= min_count
    ]

    return common[:limit]


def text_variants_for_cluster(
    items: List[ScreenshotItem],
    indices: List[int],
    limit: int = 8,
) -> List[str]:
    seen = set()
    variants: List[str] = []

    for idx in indices:
        text = " ".join([
            items[idx].normalized_text,
            items[idx].central_ocr_text,
            *items[idx].region_ocr_texts,
        ]).strip()

        text = re.sub(r"\s+", " ", text)

        if not text:
            continue

        short = text[:500]

        if short in seen:
            continue

        seen.add(short)
        variants.append(short)

        if len(variants) >= limit:
            break

    return variants


def build_cluster_name(items: List[ScreenshotItem], indices: List[int]) -> Tuple[str, str, str, str]:
    visual_counts: Dict[str, int] = {}
    text_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}

    for idx in indices:
        labels = items[idx].rule_labels

        visual = labels.get("visual_class", "unknown")
        text = labels.get("text_class", "unknown")
        severity = labels.get("severity", "medium")

        visual_counts[visual] = visual_counts.get(visual, 0) + 1
        text_counts[text] = text_counts.get(text, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    visual_class = max(visual_counts.items(), key=lambda kv: kv[1])[0]
    text_class = max(text_counts.items(), key=lambda kv: kv[1])[0]
    severity = max(severity_counts.items(), key=lambda kv: kv[1])[0]

    name = safe_name(f"{visual_class}.{text_class}")

    return name, visual_class, text_class, severity


def build_clusters(
    items: List[ScreenshotItem],
    labels: List[int],
    visual_matrix: np.ndarray,
    text_matrix: np.ndarray,
    layout_matrix: np.ndarray,
    rule_matrix: np.ndarray,
    combined_matrix: np.ndarray,
) -> List[Cluster]:
    label_to_indices: Dict[int, List[int]] = {}

    for item, label in zip(items, labels):
        item.final_cluster_id = label
        label_to_indices.setdefault(label, []).append(item.index)

    clusters: List[Cluster] = []

    sorted_groups = sorted(
        label_to_indices.values(),
        key=lambda xs: (-len(xs), min(xs)),
    )

    remap: Dict[int, int] = {}

    for new_cluster_id, indices in enumerate(sorted_groups):
        old_label = items[indices[0]].final_cluster_id
        assert old_label is not None
        remap[old_label] = new_cluster_id

    for item in items:
        assert item.final_cluster_id is not None
        item.final_cluster_id = remap[item.final_cluster_id]

    new_groups: Dict[int, List[int]] = {}

    for item in items:
        assert item.final_cluster_id is not None
        new_groups.setdefault(item.final_cluster_id, []).append(item.index)

    for cluster_id, indices in sorted(new_groups.items()):
        representative = choose_representative(indices, combined_matrix)

        if len(indices) > 1:
            pairs = [
                (i, j)
                for pos, i in enumerate(indices)
                for j in indices[pos + 1:]
            ]

            avg_visual = float(np.mean([visual_matrix[i, j] for i, j in pairs]))
            avg_text = float(np.mean([text_matrix[i, j] for i, j in pairs]))
            avg_layout = float(np.mean([layout_matrix[i, j] for i, j in pairs]))
            avg_rule = float(np.mean([rule_matrix[i, j] for i, j in pairs]))
            avg_combined = float(np.mean([combined_matrix[i, j] for i, j in pairs]))
        else:
            avg_visual = avg_text = avg_layout = avg_rule = avg_combined = 1.0

        for idx in indices:
            items[idx].nearest_to_representative_score = float(
                combined_matrix[idx, representative]
            )

        cluster_name, visual_class, text_class, severity = build_cluster_name(
            items,
            indices,
        )

        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                count=len(indices),
                representative_index=representative,
                representative_path=items[representative].path,
                cluster_name=cluster_name,
                visual_class=visual_class,
                text_class=text_class,
                severity=severity,
                avg_visual_similarity=avg_visual,
                avg_text_similarity=avg_text,
                avg_layout_similarity=avg_layout,
                avg_rule_similarity=avg_rule,
                avg_combined_similarity=avg_combined,
                items=sorted(indices),
                common_tokens=common_tokens_for_cluster(items, indices),
                text_variants=text_variants_for_cluster(items, indices),
            )
        )

    return clusters


def rects_to_overlay_image(image_path: Path, rects: List[Rect], out_path: Path) -> None:
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def item_to_json_dict(item: ScreenshotItem) -> Dict[str, Any]:
    data = asdict(item)
    data["rects"] = [asdict(r) for r in item.rects]
    return data


def cluster_to_json_dict(cluster: Cluster) -> Dict[str, Any]:
    return asdict(cluster)


def copy_or_symlink(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if use_symlink:
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)


def create_cluster_directories(
    items: List[ScreenshotItem],
    clusters: List[Cluster],
    out_dir: Path,
    use_symlink: bool,
) -> None:
    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    for cluster in clusters:
        cluster_name = f"cluster-{cluster.cluster_id:03d}__{cluster.cluster_name}__count-{cluster.count}"
        cluster_dir = clusters_dir / cluster_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        for rank, idx in enumerate(cluster.items):
            item = items[idx]
            src = Path(item.path)
            prefix = "representative" if idx == cluster.representative_index else f"item-{rank:03d}"
            dst = cluster_dir / f"{prefix}__{src.name}"
            copy_or_symlink(src, dst, use_symlink=use_symlink)


def image_to_base64_data_uri(path: Path, max_width: int = 360) -> str:
    image = Image.open(path).convert("RGB")
    w, h = image.size

    if w > max_width:
        new_h = int(h * (max_width / w))
        image = image.resize((max_width, new_h))

    import io

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=80)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def write_overlay_images(items: List[ScreenshotItem], clusters: List[Cluster], out_dir: Path) -> None:
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for cluster in clusters:
        rep = items[cluster.representative_index]
        out_path = overlay_dir / f"cluster-{cluster.cluster_id:03d}.jpg"
        rects_to_overlay_image(Path(rep.path), rep.rects, out_path)


def write_cluster_summary_csv(clusters: List[Cluster], out_dir: Path) -> None:
    path = out_dir / "cluster-summary.csv"

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_id",
                "cluster_name",
                "count",
                "severity",
                "visual_class",
                "text_class",
                "avg_combined_similarity",
                "avg_visual_similarity",
                "avg_text_similarity",
                "avg_layout_similarity",
                "avg_rule_similarity",
                "representative_path",
            ],
        )

        writer.writeheader()

        for c in clusters:
            writer.writerow(
                {
                    "cluster_id": c.cluster_id,
                    "cluster_name": c.cluster_name,
                    "count": c.count,
                    "severity": c.severity,
                    "visual_class": c.visual_class,
                    "text_class": c.text_class,
                    "avg_combined_similarity": round(c.avg_combined_similarity, 4),
                    "avg_visual_similarity": round(c.avg_visual_similarity, 4),
                    "avg_text_similarity": round(c.avg_text_similarity, 4),
                    "avg_layout_similarity": round(c.avg_layout_similarity, 4),
                    "avg_rule_similarity": round(c.avg_rule_similarity, 4),
                    "representative_path": c.representative_path,
                }
            )


def generate_html_report(
    items: List[ScreenshotItem],
    clusters: List[Cluster],
    out_dir: Path,
    visual_weight: float,
    text_weight: float,
    layout_weight: float,
    rule_weight: float,
    threshold: float,
    template_threshold: float,
    final_threshold: float,
    single_stage: bool,
) -> None:
    rows: List[str] = []

    for cluster in clusters:
        rep = items[cluster.representative_index]
        img_uri = image_to_base64_data_uri(Path(rep.path))
        overlay_path = out_dir / "overlays" / f"cluster-{cluster.cluster_id:03d}.jpg"
        overlay_uri = image_to_base64_data_uri(overlay_path) if overlay_path.exists() else ""

        token_html = ", ".join(html.escape(t) for t in cluster.common_tokens[:20])
        sample_text = html.escape(rep.normalized_text[:1200])
        central_text = html.escape(rep.central_ocr_text[:1200])
        variants_html = "\n".join(
            f"<li><code>{html.escape(v)}</code></li>"
            for v in cluster.text_variants
        )

        rows.append(
            f"""
            <section class="cluster severity-{html.escape(cluster.severity)}">
              <div class="cluster-header">
                <div>
                  <h2>Cluster {cluster.cluster_id:03d}: {html.escape(cluster.cluster_name)}</h2>
                  <div class="subtle">
                    visual={html.escape(cluster.visual_class)} |
                    text={html.escape(cluster.text_class)} |
                    severity={html.escape(cluster.severity)}
                  </div>
                </div>
                <div class="count">{cluster.count} screenshots</div>
              </div>

              <div class="metrics">
                <span>combined: {cluster.avg_combined_similarity:.3f}</span>
                <span>visual: {cluster.avg_visual_similarity:.3f}</span>
                <span>text: {cluster.avg_text_similarity:.3f}</span>
                <span>layout: {cluster.avg_layout_similarity:.3f}</span>
                <span>rule: {cluster.avg_rule_similarity:.3f}</span>
              </div>

              <div class="images">
                <div>
                  <h3>Representative</h3>
                  <img src="{img_uri}" />
                </div>
                <div>
                  <h3>Detected layout</h3>
                  <img src="{overlay_uri}" />
                </div>
              </div>

              <div class="details">
                <p><b>Representative file:</b> {html.escape(rep.filename)}</p>
                <p><b>Common OCR tokens:</b> {token_html}</p>

                <h3>Normalized OCR sample</h3>
                <pre>{sample_text}</pre>

                <h3>Central OCR sample</h3>
                <pre>{central_text}</pre>

                <h3>Text variants</h3>
                <ol>{variants_html}</ol>
              </div>
            </section>
            """
        )

    mode_text = "single-stage" if single_stage else "two-stage"

    doc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Screenshot Cluster Report</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f9fafb;
    }}
    h1 {{
      margin-bottom: 4px;
    }}
    .summary {{
      margin-bottom: 24px;
      padding: 16px;
      background: white;
      border-radius: 12px;
      border: 1px solid #e5e7eb;
    }}
    .cluster {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    .cluster-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .cluster-header h2 {{
      margin: 0;
    }}
    .subtle {{
      color: #6b7280;
      margin-top: 4px;
      font-size: 14px;
    }}
    .count {{
      font-weight: 700;
      background: #eef2ff;
      padding: 6px 10px;
      border-radius: 999px;
      white-space: nowrap;
    }}
    .metrics {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 12px 0;
    }}
    .metrics span {{
      background: #f3f4f6;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 13px;
    }}
    .images {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-top: 12px;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #111827;
      color: #f9fafb;
      border-radius: 8px;
      padding: 12px;
      max-height: 260px;
      overflow: auto;
    }}
    code {{
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .severity-high {{
      border-left: 6px solid #ef4444;
    }}
    .severity-critical {{
      border-left: 6px solid #7f1d1d;
    }}
    .severity-medium {{
      border-left: 6px solid #f59e0b;
    }}
    .severity-low {{
      border-left: 6px solid #22c55e;
    }}
  </style>
</head>
<body>
  <h1>Screenshot Cluster Report</h1>

  <div class="summary">
    <p><b>Total screenshots:</b> {len(items)}</p>
    <p><b>Total clusters:</b> {len(clusters)}</p>
    <p><b>Mode:</b> {html.escape(mode_text)}</p>
    <p><b>Single-stage threshold:</b> {threshold:.3f}</p>
    <p><b>Template threshold:</b> {template_threshold:.3f}</p>
    <p><b>Final threshold:</b> {final_threshold:.3f}</p>
    <p><b>Weights:</b> visual={visual_weight}, text={text_weight}, layout={layout_weight}, rule={rule_weight}</p>
  </div>

  {''.join(rows)}
</body>
</html>
"""

    (out_dir / "report.html").write_text(doc, encoding="utf-8")



def analyze_screenshot_worker(task: Tuple[str, int, str, str, str, Optional[str]]) -> ScreenshotItem:
    """
    Worker entrypoint for multiprocessing.

    Keep it top-level so it is picklable on macOS/Windows.
    """
    path_str, index, ocr_lang, ocr_engine, ocr_mode, debug_ocr_dir_str = task

    debug_ocr_dir = Path(debug_ocr_dir_str) if debug_ocr_dir_str else None

    return analyze_screenshot(
        path=Path(path_str),
        index=index,
        ocr_lang=ocr_lang,
        ocr_engine=ocr_engine,
        ocr_mode=ocr_mode,
        debug_ocr_dir=debug_ocr_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster UI screenshots by visual, OCR text, layout, and rule-based semantic similarity."
    )

    parser.add_argument(
        "input_dir",
        help="Directory with screenshots.",
    )

    parser.add_argument(
        "output_dir",
        help="Directory where reports and clusters will be written.",
    )

    parser.add_argument(
        "--ocr-engine",
        choices=["tesseract", "easyocr"],
        default="tesseract",
        help="OCR engine. Default: tesseract.",
    )

    parser.add_argument(
        "--ocr-mode",
        choices=["fast", "balanced", "accurate"],
        default="fast",
        help="OCR speed/quality mode. Default: fast.",
    )

    parser.add_argument(
        "--ocr-lang",
        default="eng",
        help="OCR language, e.g. eng, rus, eng+rus. Default: eng.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Combined similarity threshold for single-stage clustering. Default: 0.82.",
    )

    parser.add_argument(
        "--template-threshold",
        type=float,
        default=0.84,
        help="Similarity threshold for first-stage visual/layout template clustering. Default: 0.84.",
    )

    parser.add_argument(
        "--final-threshold",
        type=float,
        default=0.78,
        help="Similarity threshold for second-stage text/symptom clustering inside templates. Default: 0.78.",
    )

    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Use single-stage clustering instead of two-stage clustering.",
    )

    parser.add_argument(
        "--visual-weight",
        type=float,
        default=0.30,
        help="Weight of visual hash similarity. Default: 0.30.",
    )

    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.25,
        help="Weight of OCR text similarity. Default: 0.25.",
    )

    parser.add_argument(
        "--layout-weight",
        type=float,
        default=0.25,
        help="Weight of layout/composition similarity. Default: 0.25.",
    )

    parser.add_argument(
        "--rule-weight",
        type=float,
        default=0.20,
        help="Weight of rule-based semantic similarity. Default: 0.20.",
    )

    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images into cluster folders instead of creating symlinks.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of screenshots for debugging. 0 means no limit.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel screenshot analysis workers. Default: 1.",
    )

    parser.add_argument(
        "--debug-ocr",
        action="store_true",
        help="Save OCR debug crops.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(input_dir)

    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise SystemExit(f"No supported images found in: {input_dir}")

    print(f"Found screenshots: {len(image_paths)}")
    print(f"Output directory: {out_dir}")
    print(f"OCR engine: {args.ocr_engine}")
    print(f"OCR language: {args.ocr_lang}")

    items: List[ScreenshotItem] = []
    debug_ocr_dir = out_dir / "debug-ocr" if args.debug_ocr else None

    workers = max(1, int(args.workers))

    # Tesseract/OpenCV/NumPy may use internal threading.
    # When using multiple Python worker processes, limiting nested native threads
    # prevents CPU oversubscription and usually improves throughput.
    if workers > 1:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if args.ocr_engine == "easyocr" and workers > 1:
        print("WARNING: EasyOCR loads a heavy torch model per worker. For EasyOCR, workers=1 is usually safer.")
        print("         Use multiprocessing primarily with --ocr-engine tesseract.")

    tasks = [
        (
            str(path),
            index,
            args.ocr_lang,
            args.ocr_engine,
            args.ocr_mode,
            str(debug_ocr_dir) if debug_ocr_dir is not None else None,
        )
        for index, path in enumerate(image_paths)
    ]

    if workers == 1:
        for task in tqdm(tasks, desc="Analyzing screenshots"):
            path_str = task[0]
            try:
                item = analyze_screenshot_worker(task)
                items.append(item)
            except Exception as e:
                print(f"Failed to analyze {path_str}: {e}")
    else:
        print(f"Using parallel screenshot analysis workers: {workers}")

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(analyze_screenshot_worker, task): task
                for task in tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing screenshots"):
                task = futures[future]
                path_str = task[0]

                try:
                    item = future.result()
                    items.append(item)
                except Exception as e:
                    print(f"Failed to analyze {path_str}: {e}")

        # Restore deterministic item order after parallel execution.
        items.sort(key=lambda item: item.index)

    if not items:
        raise SystemExit("No screenshots were successfully analyzed.")

    visual_matrix, text_matrix, layout_matrix, rule_matrix, combined_matrix = compute_similarity_matrices(
        items,
        visual_weight=args.visual_weight,
        text_weight=args.text_weight,
        layout_weight=args.layout_weight,
        rule_weight=args.rule_weight,
    )

    if args.single_stage:
        labels = cluster_items_single_stage(
            items,
            combined_similarity_matrix=combined_matrix,
            threshold=args.threshold,
        )
    else:
        labels = two_stage_cluster_items(
            items=items,
            visual_matrix=visual_matrix,
            layout_matrix=layout_matrix,
            text_matrix=text_matrix,
            rule_matrix=rule_matrix,
            template_threshold=args.template_threshold,
            final_threshold=args.final_threshold,
        )

    clusters = build_clusters(
        items=items,
        labels=labels,
        visual_matrix=visual_matrix,
        text_matrix=text_matrix,
        layout_matrix=layout_matrix,
        rule_matrix=rule_matrix,
        combined_matrix=combined_matrix,
    )

    clusters.sort(key=lambda c: (-c.count, c.cluster_id))

    write_jsonl(out_dir / "items.jsonl", (item_to_json_dict(item) for item in items))

    (out_dir / "clusters.json").write_text(
        json.dumps([cluster_to_json_dict(c) for c in clusters], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_cluster_summary_csv(clusters, out_dir)

    np.save(out_dir / "similarity_visual.npy", visual_matrix)
    np.save(out_dir / "similarity_text.npy", text_matrix)
    np.save(out_dir / "similarity_layout.npy", layout_matrix)
    np.save(out_dir / "similarity_rule.npy", rule_matrix)
    np.save(out_dir / "similarity_combined.npy", combined_matrix)

    write_overlay_images(items, clusters, out_dir)

    create_cluster_directories(
        items=items,
        clusters=clusters,
        out_dir=out_dir,
        use_symlink=not args.copy,
    )

    generate_html_report(
        items=items,
        clusters=clusters,
        out_dir=out_dir,
        visual_weight=args.visual_weight,
        text_weight=args.text_weight,
        layout_weight=args.layout_weight,
        rule_weight=args.rule_weight,
        threshold=args.threshold,
        template_threshold=args.template_threshold,
        final_threshold=args.final_threshold,
        single_stage=args.single_stage,
    )

    print()
    print("Done.")
    print(f"Total screenshots: {len(items)}")
    print(f"Total clusters: {len(clusters)}")
    print(f"HTML report: {out_dir / 'report.html'}")
    print(f"Clusters JSON: {out_dir / 'clusters.json'}")
    print(f"Items JSONL: {out_dir / 'items.jsonl'}")
    print(f"Cluster CSV: {out_dir / 'cluster-summary.csv'}")


if __name__ == "__main__":
    main()
