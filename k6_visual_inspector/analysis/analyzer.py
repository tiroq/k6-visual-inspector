"""Per-screenshot analysis — main analyzer and multiprocessing worker."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from ..models import Rect, ScreenshotItem
from ..image.loading import load_image
from ..image.hashing import compute_hashes
from ..image.layout import detect_layout_rects, build_layout_signature, iou_rect
from ..image.crops import crop_center, crop_rect
from ..ocr.engines import extract_ocr_text
from ..ocr.cleanup import normalize_text, tokenize_text, is_useful_ocr_text, deduplicate_texts
from ..fileio.filesystem import safe_name
from .rules import detect_rule_labels
from .signatures import build_semantic_signature


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Region OCR
# ---------------------------------------------------------------------------

def _extract_region_ocr_texts(
    image: Image.Image,
    rects: List[Rect],
    lang: str,
    engine: str,
    mode: str,
) -> List[str]:
    """Run OCR on the most informative UI regions.

    Priorities:
    - modal/dialog;
    - large content frame;
    - top banner / toast-like area;
    - center crop.
    """
    texts: List[str] = []
    _, height = image.size
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
            raw = extract_ocr_text(cropped, lang=lang, engine=engine, mode=mode)
            norm = normalize_text(raw)

            if is_useful_ocr_text(norm):
                texts.append(norm)

        except Exception:
            continue

    try:
        center = crop_center(image, margin_x=0.15, margin_y=0.15)
        raw = extract_ocr_text(center, lang=lang, engine=engine, mode=mode)
        norm = normalize_text(raw)

        if is_useful_ocr_text(norm):
            texts.append(norm)

    except Exception:
        pass

    return deduplicate_texts(texts)


# ---------------------------------------------------------------------------
# OCR debug crops
# ---------------------------------------------------------------------------

def save_ocr_debug_crops(
    image: Image.Image,
    rects: List[Rect],
    original_path: Path,
    out_dir: Path,
    index: int,
) -> None:
    """Save the full image, center crop, and per-rect crops for OCR debugging."""
    base = safe_name(original_path.stem)
    item_dir = out_dir / f"{index:05d}__{base}"
    item_dir.mkdir(parents=True, exist_ok=True)

    image.save(item_dir / "full.png")

    center = crop_center(image, margin_x=0.15, margin_y=0.15)
    center.save(item_dir / "center.png")

    for i, rect in enumerate(rects[:10]):
        crop = crop_rect(image, rect, padding=12)
        crop.save(item_dir / f"rect-{i:02d}__{rect.kind}.png")


# ---------------------------------------------------------------------------
# Main analysis entry-point
# ---------------------------------------------------------------------------

def analyze_screenshot(
    path: Path,
    index: int,
    ocr_lang: str,
    ocr_engine: str,
    ocr_mode: str,
    debug_ocr_dir: Optional[Path] = None,
) -> ScreenshotItem:
    """Fully analyse a single screenshot and return a ScreenshotItem."""
    image = load_image(path)
    width, height = image.size

    phash_str, dhash_str = compute_hashes(image)

    rects = detect_layout_rects(image)

    if debug_ocr_dir is not None:
        save_ocr_debug_crops(
            image=image,
            rects=rects,
            original_path=path,
            out_dir=debug_ocr_dir,
            index=index,
        )

    ocr_text = extract_ocr_text(image, lang=ocr_lang, engine=ocr_engine, mode=ocr_mode)
    normalized = normalize_text(ocr_text)

    central_img = crop_center(image)
    central_text_raw = extract_ocr_text(central_img, lang=ocr_lang, engine=ocr_engine, mode=ocr_mode)
    central_text = normalize_text(central_text_raw)

    region_texts = _extract_region_ocr_texts(
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
        phash=phash_str,
        dhash=dhash_str,
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


# ---------------------------------------------------------------------------
# Multiprocessing worker (must be top-level to be picklable)
# ---------------------------------------------------------------------------

def analyze_screenshot_worker(
    task: Tuple[str, int, str, str, str, Optional[str]],
) -> ScreenshotItem:
    """Worker entry-point for ProcessPoolExecutor.

    Kept at module level so it is picklable on macOS/Windows.
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
