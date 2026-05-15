#!/usr/bin/env python3
"""
Cluster UI screenshots by:
1. visual similarity,
2. OCR text similarity,
3. layout / composition similarity.

The script is designed for post-processing k6/browser screenshots
when no explicit metadata is available.

Usage:
    python3 cluster_screenshots.py ./screenshots ./out

Example:
    python3 cluster_screenshots.py artifacts/run-001/screenshots artifacts/run-001/analysis

Notes:
    - Comments are intentionally in English.
    - OCR quality depends heavily on screenshot resolution and Tesseract installation.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
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
    text_tokens: List[str]

    rects: List[Rect]
    layout_signature: List[float]

    cluster_id: Optional[int] = None
    nearest_to_representative_score: Optional[float] = None


@dataclass
class Cluster:
    cluster_id: int
    count: int
    representative_index: int
    representative_path: str

    avg_visual_similarity: float
    avg_text_similarity: float
    avg_layout_similarity: float
    avg_combined_similarity: float

    items: List[int] = field(default_factory=list)
    common_tokens: List[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: str) -> str:
    value = value.strip().lower()
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

    # Normalize URLs, emails, ids, dates, times, numbers.
    text = re.sub(r"https?://\S+", " <url> ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " <email> ", text)

    # UUIDs.
    text = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        " <uuid> ",
        text,
    )

    # Long hex / hashes / ids.
    text = re.sub(r"\b[0-9a-f]{10,}\b", " <hex> ", text)

    # ISO-like dates and times.
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[t\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?z?\b", " <datetime> ", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " <date> ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " <time> ", text)

    # Numbers, prices, counts.
    text = re.sub(r"\b\d+(?:[.,]\d+)?\b", " <num> ", text)

    # Normalize whitespace and punctuation noise.
    text = re.sub(r"[^\w<>/.-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9_<>/-]{2,}", text.lower())
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
        "the",
    }
    return [t for t in tokens if t not in stop]


def extract_ocr_text(image: Image.Image, lang: str) -> str:
    # Upscale a little to improve OCR for browser screenshots.
    w, h = image.size
    scale = 1.5 if max(w, h) < 1800 else 1.0
    if scale != 1.0:
        image = image.resize((int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Mild thresholding improves text extraction for many UI screenshots.
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--psm 6"
    try:
        return pytesseract.image_to_string(thresh, lang=lang, config=config)
    except pytesseract.TesseractError:
        # Fallback to English if requested language pack is not installed.
        return pytesseract.image_to_string(thresh, lang="eng", config=config)


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

    # Smooth small texture while preserving larger edges.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection.
    edges = cv2.Canny(blur, 40, 120)

    # Close gaps so UI frames become complete contours.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: List[Rect] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_ratio = area / total_area

        # Ignore tiny artifacts.
        if area_ratio < 0.01:
            continue

        # Ignore almost full-screen border.
        if area_ratio > 0.98:
            continue

        aspect = w / max(h, 1)

        cx = x + w / 2
        cy = y + h / 2

        center_dx = abs(cx - width / 2) / width
        center_dy = abs(cy - height / 2) / height

        kind = "region"

        # Heuristic modal detection: centered, medium-sized rectangle.
        if 0.08 <= area_ratio <= 0.65 and center_dx < 0.18 and center_dy < 0.18:
            kind = "modal_or_dialog"

        # Heuristic sidebar.
        elif h > height * 0.55 and w < width * 0.35:
            kind = "sidebar_or_panel"

        # Heuristic top bar.
        elif w > width * 0.65 and h < height * 0.22 and y < height * 0.25:
            kind = "topbar_or_header"

        # Heuristic bottom area.
        elif w > width * 0.65 and h < height * 0.25 and y > height * 0.65:
            kind = "footer_or_bottom_panel"

        # Heuristic table/card/content area.
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

    # Sort by area descending, keep a bounded number for stable signature.
    rects.sort(key=lambda r: r.area_ratio, reverse=True)

    # Remove strongly overlapping rectangles.
    filtered: List[Rect] = []
    for rect in rects:
        if not any(iou_rect(rect, existing) > 0.80 for existing in filtered):
            filtered.append(rect)

    return filtered[:20]


def iou_rect(a: Rect, b: Rect) -> float:
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

    # Resize to fixed grid.
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

    # Pad to fixed length.
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

    # Convert from [-1, 1] to [0, 1].
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    # token_set_ratio is robust when one screenshot contains extra UI text.
    return fuzz.token_set_ratio(a, b) / 100.0


def combined_similarity(
    visual: float,
    text: float,
    layout: float,
    visual_weight: float,
    text_weight: float,
    layout_weight: float,
) -> float:
    total = visual_weight + text_weight + layout_weight
    if total <= 0:
        raise ValueError("At least one weight must be positive")

    return (
        visual * visual_weight
        + text * text_weight
        + layout * layout_weight
    ) / total


def analyze_screenshot(path: Path, index: int, ocr_lang: str) -> ScreenshotItem:
    image = load_image(path)
    width, height = image.size

    p_hash = imagehash.phash(image)
    d_hash = imagehash.dhash(image)

    ocr_text = extract_ocr_text(image, ocr_lang)
    normalized = normalize_text(ocr_text)
    tokens = tokenize_text(normalized)

    rects = detect_layout_rects(image)
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
        text_tokens=tokens,
        rects=rects,
        layout_signature=layout_signature,
    )


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(items)

    visual = np.eye(n, dtype=np.float32)
    text = np.eye(n, dtype=np.float32)
    layout = np.eye(n, dtype=np.float32)
    combined = np.eye(n, dtype=np.float32)

    for i in tqdm(range(n), desc="Comparing screenshots"):
        for j in range(i + 1, n):
            # Combine pHash and dHash for more stable visual similarity.
            ph = hamming_hash_similarity(items[i].phash, items[j].phash)
            dh = hamming_hash_similarity(items[i].dhash, items[j].dhash)
            v = (ph * 0.7) + (dh * 0.3)

            t = text_similarity(items[i].normalized_text, items[j].normalized_text)

            l = cosine_similarity(items[i].layout_signature, items[j].layout_signature)

            c = combined_similarity(
                visual=v,
                text=t,
                layout=l,
                visual_weight=visual_weight,
                text_weight=text_weight,
                layout_weight=layout_weight,
            )

            visual[i, j] = visual[j, i] = v
            text[i, j] = text[j, i] = t
            layout[i, j] = layout[j, i] = l
            combined[i, j] = combined[j, i] = c

    return visual, text, layout, combined


def cluster_items(
    items: List[ScreenshotItem],
    combined_similarity_matrix: np.ndarray,
    threshold: float,
) -> List[int]:
    """
    Agglomerative clustering with a precomputed distance matrix.

    threshold is a similarity threshold.
    distance = 1 - similarity.
    """
    n = len(items)

    if n == 0:
        return []

    if n == 1:
        return [0]

    distance_matrix = 1.0 - combined_similarity_matrix
    distance_threshold = 1.0 - threshold

    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        # Compatibility with older scikit-learn versions.
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
        )

    labels = model.fit_predict(distance_matrix)
    return [int(x) for x in labels]


def choose_representative(
    indices: List[int],
    combined_matrix: np.ndarray,
) -> int:
    """
    Choose the most central item in the cluster.
    """
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


def build_clusters(
    items: List[ScreenshotItem],
    labels: List[int],
    visual_matrix: np.ndarray,
    text_matrix: np.ndarray,
    layout_matrix: np.ndarray,
    combined_matrix: np.ndarray,
) -> List[Cluster]:
    label_to_indices: Dict[int, List[int]] = {}

    for item, label in zip(items, labels):
        item.cluster_id = label
        label_to_indices.setdefault(label, []).append(item.index)

    clusters: List[Cluster] = []

    # Normalize cluster ids by size desc.
    sorted_groups = sorted(
        label_to_indices.values(),
        key=lambda xs: (-len(xs), min(xs)),
    )

    old_to_new: Dict[int, int] = {}

    for new_cluster_id, indices in enumerate(sorted_groups):
        old_label = items[indices[0]].cluster_id
        assert old_label is not None
        old_to_new[old_label] = new_cluster_id

    for item in items:
        assert item.cluster_id is not None
        item.cluster_id = old_to_new[item.cluster_id]

    new_groups: Dict[int, List[int]] = {}
    for item in items:
        assert item.cluster_id is not None
        new_groups.setdefault(item.cluster_id, []).append(item.index)

    for cluster_id, indices in sorted(new_groups.items()):
        representative = choose_representative(indices, combined_matrix)

        if len(indices) > 1:
            pairs = [(i, j) for pos, i in enumerate(indices) for j in indices[pos + 1 :]]

            avg_visual = float(np.mean([visual_matrix[i, j] for i, j in pairs]))
            avg_text = float(np.mean([text_matrix[i, j] for i, j in pairs]))
            avg_layout = float(np.mean([layout_matrix[i, j] for i, j in pairs]))
            avg_combined = float(np.mean([combined_matrix[i, j] for i, j in pairs]))
        else:
            avg_visual = avg_text = avg_layout = avg_combined = 1.0

        for idx in indices:
            items[idx].nearest_to_representative_score = float(combined_matrix[idx, representative])

        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                count=len(indices),
                representative_index=representative,
                representative_path=items[representative].path,
                avg_visual_similarity=avg_visual,
                avg_text_similarity=avg_text,
                avg_layout_similarity=avg_layout,
                avg_combined_similarity=avg_combined,
                items=sorted(indices),
                common_tokens=common_tokens_for_cluster(items, indices),
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
        cluster_name = f"cluster-{cluster.cluster_id:03d}__count-{cluster.count}"
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


def generate_html_report(
    items: List[ScreenshotItem],
    clusters: List[Cluster],
    out_dir: Path,
    visual_weight: float,
    text_weight: float,
    layout_weight: float,
    threshold: float,
) -> None:
    rows: List[str] = []

    for cluster in clusters:
        rep = items[cluster.representative_index]
        img_uri = image_to_base64_data_uri(Path(rep.path))
        overlay_path = out_dir / "overlays" / f"cluster-{cluster.cluster_id:03d}.jpg"
        overlay_uri = image_to_base64_data_uri(overlay_path) if overlay_path.exists() else ""

        token_html = ", ".join(html.escape(t) for t in cluster.common_tokens[:20])
        sample_text = html.escape(rep.normalized_text[:1200])

        rows.append(
            f"""
            <section class="cluster">
              <div class="cluster-header">
                <h2>Cluster {cluster.cluster_id:03d}</h2>
                <div class="count">{cluster.count} screenshots</div>
              </div>

              <div class="metrics">
                <span>combined: {cluster.avg_combined_similarity:.3f}</span>
                <span>visual: {cluster.avg_visual_similarity:.3f}</span>
                <span>text: {cluster.avg_text_similarity:.3f}</span>
                <span>layout: {cluster.avg_layout_similarity:.3f}</span>
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
                <pre>{sample_text}</pre>
              </div>
            </section>
            """
        )

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
    .count {{
      font-weight: 700;
      background: #eef2ff;
      padding: 6px 10px;
      border-radius: 999px;
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
  </style>
</head>
<body>
  <h1>Screenshot Cluster Report</h1>

  <div class="summary">
    <p><b>Total screenshots:</b> {len(items)}</p>
    <p><b>Total clusters:</b> {len(clusters)}</p>
    <p><b>Threshold:</b> {threshold:.3f}</p>
    <p><b>Weights:</b> visual={visual_weight}, text={text_weight}, layout={layout_weight}</p>
  </div>

  {''.join(rows)}
</body>
</html>
"""

    (out_dir / "report.html").write_text(doc, encoding="utf-8")


def write_overlay_images(items: List[ScreenshotItem], clusters: List[Cluster], out_dir: Path) -> None:
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for cluster in clusters:
        rep = items[cluster.representative_index]
        out_path = overlay_dir / f"cluster-{cluster.cluster_id:03d}.jpg"
        rects_to_overlay_image(Path(rep.path), rep.rects, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster UI screenshots by visual, OCR text, and layout similarity."
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
        "--ocr-lang",
        default="eng",
        help="Tesseract language, e.g. eng, rus, eng+rus. Default: eng.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Combined similarity threshold for clustering. Higher means stricter clusters. Default: 0.82.",
    )

    parser.add_argument(
        "--visual-weight",
        type=float,
        default=0.40,
        help="Weight of visual hash similarity. Default: 0.40.",
    )

    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.35,
        help="Weight of OCR text similarity. Default: 0.35.",
    )

    parser.add_argument(
        "--layout-weight",
        type=float,
        default=0.25,
        help="Weight of layout/composition similarity. Default: 0.25.",
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

    items: List[ScreenshotItem] = []

    for index, path in enumerate(tqdm(image_paths, desc="Analyzing screenshots")):
        try:
            item = analyze_screenshot(path, index=index, ocr_lang=args.ocr_lang)
            items.append(item)
        except Exception as e:
            print(f"Failed to analyze {path}: {e}")

    if not items:
        raise SystemExit("No screenshots were successfully analyzed.")

    visual_matrix, text_matrix, layout_matrix, combined_matrix = compute_similarity_matrices(
        items,
        visual_weight=args.visual_weight,
        text_weight=args.text_weight,
        layout_weight=args.layout_weight,
    )

    labels = cluster_items(
        items,
        combined_similarity_matrix=combined_matrix,
        threshold=args.threshold,
    )

    clusters = build_clusters(
        items=items,
        labels=labels,
        visual_matrix=visual_matrix,
        text_matrix=text_matrix,
        layout_matrix=layout_matrix,
        combined_matrix=combined_matrix,
    )

    # Sort clusters by size descending.
    clusters.sort(key=lambda c: (-c.count, c.cluster_id))

    write_jsonl(out_dir / "items.jsonl", (item_to_json_dict(item) for item in items))

    (out_dir / "clusters.json").write_text(
        json.dumps([cluster_to_json_dict(c) for c in clusters], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    np.save(out_dir / "similarity_visual.npy", visual_matrix)
    np.save(out_dir / "similarity_text.npy", text_matrix)
    np.save(out_dir / "similarity_layout.npy", layout_matrix)
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
        threshold=args.threshold,
    )

    print()
    print("Done.")
    print(f"Total screenshots: {len(items)}")
    print(f"Total clusters: {len(clusters)}")
    print(f"HTML report: {out_dir / 'report.html'}")
    print(f"Clusters JSON: {out_dir / 'clusters.json'}")
    print(f"Items JSONL: {out_dir / 'items.jsonl'}")


if __name__ == "__main__":
    main()