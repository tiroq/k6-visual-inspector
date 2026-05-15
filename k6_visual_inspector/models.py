"""Shared data-model dataclasses used across the whole package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
