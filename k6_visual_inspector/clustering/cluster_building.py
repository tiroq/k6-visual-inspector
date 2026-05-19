"""Cluster building — representative selection, token aggregation, and Cluster construction."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

import numpy as np

from ..models import Cluster, ScreenshotItem
from ..fileio.filesystem import safe_name


def choose_representative(
    indices: List[int],
    combined_matrix: np.ndarray,
) -> int:
    """Return the index with the highest average similarity to all other cluster members."""
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
    """Return tokens that appear in at least 40% of cluster members."""
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
    """Return up to *limit* distinct text samples from cluster members."""
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


def _build_cluster_name(
    items: List[ScreenshotItem],
    indices: List[int],
) -> Tuple[str, str, str, str]:
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
    """Construct Cluster objects from per-item labels and similarity matrices."""
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

        cluster_name, visual_class, text_class, severity = _build_cluster_name(
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
